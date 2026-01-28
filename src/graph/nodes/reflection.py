import logging
import json
import re
from typing import Dict, Any, List, Optional
from src.graph.state import GraphState
from src.agents.prompt_builder import PromptBuilder, ReflectionView
from src.rag.strategies.finmem import FinMemRAG
from langchain_openai import ChatOpenAI
from src.config import settings

logger = logging.getLogger(__name__)

def reflection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that performs self-reflection and updates memory importance.
    Now optimized for a single factor (parallel fan-out).
    """
    factor_name = state.get("factor_name")
    actual_return = state.get("actual_return")
    view = state.get("view")
    target_date = state.get("target_date")
    context = state.get("context", [])

    if not factor_name or actual_return is None or not view:
        logger.warning(f"Missing data for reflection on {factor_name}. Skipping.")
        return {}

    logger.info(f"Executing Reflection Node for factor: {factor_name}...")

    # 1. Prepare Context with IDs
    context_text = ""
    doc_map = {}
    for doc in context:
        doc_id = doc.metadata.get("temp_id")
        if doc_id:
            doc_map[doc_id] = doc
            context_text += f"[{doc_id}] (Source: {doc.metadata.get('filename', 'Unknown')}, Date: {doc.metadata.get('date', 'Unknown')}):\n{doc.page_content}\n\n"

    # 2. Run Reflection LLM with Structured Output
    builder = PromptBuilder()
    builder.set_target_date(target_date)\
           .set_factor_expertise({"name": factor_name})\
           .set_actual_return(actual_return)
    
    prompt = builder.build_reflection_prompt(context_text)
    
    llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY, temperature=0)
    structured_llm = llm.with_structured_output(ReflectionView)
    
    response = structured_llm.invoke([
        ("system", "You are a senior financial analyst performing a post-investment review."),
        ("human", prompt)
    ])
    
    reflection_data = response.model_dump()
    q_value = view.get("q_value", 0.0)

    # 3. Calculate Feedback (F)
    feedback = 0.0
    if q_value > 0 and actual_return > 0:
        feedback = 1.0
    elif q_value < 0 and actual_return < 0:
        feedback = 1.0
    elif q_value > 0 and actual_return < 0:
        feedback = -1.0
    elif q_value < 0 and actual_return > 0:
        feedback = -1.0
    elif q_value == 0 and abs(actual_return) > 0.01:
        feedback = -0.5
    
    # 4. Handle Misleading Cases
    cited_ids = reflection_data.get("cited_doc_ids", [])
    if not cited_ids and feedback < 0:
        misleading_prompt = builder.build_misleading_reflection_prompt(context_text)
        response = structured_llm.invoke([
            ("system", "You are a senior financial analyst performing a post-investment review."),
            ("human", misleading_prompt)
        ])
        reflection_data = response.model_dump()
        cited_ids = reflection_data.get("cited_doc_ids", [])

    # 5. Update Memory (FAISS)
    if cited_ids and feedback != 0:
        _update_faiss_importance(cited_ids, doc_map, feedback)

    return {"reflections": {factor_name: reflection_data}}

def _update_faiss_importance(cited_ids: List[str], doc_map: Dict[str, Any], feedback: float):
    """
    Updates the access_counter and importance in the FAISS index for cited documents.
    """
    strategy = FinMemRAG()
    indices_updated = set()

    for doc_id in cited_ids:
        doc = doc_map.get(doc_id)
        if not doc:
            continue
        
        tier = doc.metadata.get("tier")
        if not tier or tier not in strategy.indices:
            continue
            
        index = strategy.indices[tier]
        
        target_found = False
        # We need to find the document in the docstore to update its metadata
        for internal_id, stored_doc in index.docstore._dict.items():
            # Match by content and metadata to ensure uniqueness
            if stored_doc.page_content == doc.page_content and \
               stored_doc.metadata.get("source_id") == doc.metadata.get("source_id"):
                
                # Found the document! Apply FinMem logic
                ac = stored_doc.metadata.get("access_counter", 0)
                importance = stored_doc.metadata.get("importance", 40.0)
                
                new_ac = ac + feedback
                new_importance = importance + (new_ac * 5.0)
                new_importance = max(0.0, min(100.0, new_importance))
                
                stored_doc.metadata["access_counter"] = new_ac
                stored_doc.metadata["importance"] = float(new_importance)
                
                logger.info(f"Updated doc {doc_id} in tier {tier}: AC {ac}->{new_ac}, Importance {importance}->{new_importance:.2f}")
                target_found = True
                indices_updated.add(tier)
                break
        
        if not target_found:
            logger.warning(f"Could not find document {doc_id} in FAISS docstore for update.")

    # Persist changes
    for tier in indices_updated:
        tier_path = strategy.vector_db_root / tier
        strategy.indices[tier].save_local(str(tier_path))
        logger.info(f"Persisted updated FAISS index for tier: {tier}")
