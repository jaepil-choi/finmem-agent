import logging
import json
import re
from typing import Dict, Any, List
from src.graph.state import GraphState
from src.agents.prompt_builder import PromptBuilder
from src.rag.strategies.finmem import FinMemRAG
from langchain_openai import ChatOpenAI
from src.config import settings

logger = logging.getLogger(__name__)

def reflection_node(state: GraphState) -> GraphState:
    """
    Node that performs self-reflection and updates memory importance.
    Only active in training mode.
    """
    if not state.get("is_training", False):
        logger.info("Skipping Reflection Node: Not in training mode.")
        return state

    actual_return = state.get("actual_return")
    if actual_return is None:
        logger.warning("No actual_return provided in state. Cannot perform reflection.")
        return state

    logger.info("Executing Reflection Node...")

    # 1. Identify the committee view to reflect on
    # For now, we assume the first committee in committee_views
    if not state.get("committee_views"):
        logger.warning("No committee views found to reflect on.")
        return state
    
    factor_name = list(state["committee_views"].keys())[0]
    view = state["committee_views"][factor_name]
    q_value = view.get("q_value", 0.0)

    # 2. Prepare Context with IDs (already done in analyst node)
    context_text = ""
    doc_map = {}
    for doc in state["context"]:
        doc_id = doc.metadata.get("temp_id")
        if doc_id:
            doc_map[doc_id] = doc
            context_text += f"[{doc_id}] (Source: {doc.metadata.get('filename', 'Unknown')}, Date: {doc.metadata.get('date', 'Unknown')}):\n{doc.page_content}\n\n"

    # 3. Run Reflection LLM
    builder = PromptBuilder()
    builder.set_target_date(state["target_date"])\
           .set_factor_expertise({"name": factor_name})\
           .set_actual_return(actual_return)
    
    prompt = builder.build_reflection_prompt(context_text)
    
    llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY, temperature=0)
    response = llm.invoke([("system", "You are a senior financial analyst performing a post-investment review."), ("human", prompt)])
    
    reflection_data = _parse_reflection_response(response.content)
    logger.info(f"Reflection Summary: {reflection_data.get('summary_reason')}")
    
    # 4. Calculate Feedback (F)
    # F = 1 if success, -1 if failure
    # Neutral (Q=0) penalty logic:
    # If market moved significantly (|return| > 0.01) but agent was neutral, F = -0.5
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
        # Penalize for missing a significant market move
        feedback = -0.5
        logger.info(f"Applying soft penalty (F={feedback}) for neutral view on significant market move ({actual_return:.4f})")
    
    # 5. Handle Citations and Multi-step Reflection if needed
    cited_ids = reflection_data.get("cited_doc_ids", [])
    
    # If no citations and prediction was incorrect or missed a signal, try misleading prompt
    if not cited_ids and feedback < 0:
        logger.info("No citations found for incorrect/neutral prediction. Retrying with misleading prompt.")
        misleading_prompt = builder.build_misleading_reflection_prompt(context_text)
        response = llm.invoke([("system", "You are a senior financial analyst performing a post-investment review."), ("human", misleading_prompt)])
        reflection_data = _parse_reflection_response(response.content)
        cited_ids = reflection_data.get("cited_doc_ids", [])
        logger.info(f"Misleading Reflection Summary: {reflection_data.get('summary_reason')}")

    # 6. Update Memory (FAISS) - Skip if no citations
    if not cited_ids:
        logger.info("No documents cited in reflection. Skipping importance updates.")
    elif feedback != 0:
        _update_faiss_importance(cited_ids, doc_map, feedback)

    # 7. Update State
    reflections = state.get("reflections", {})
    reflections[factor_name] = reflection_data
    
    return {"reflections": reflections}

def _parse_reflection_response(content: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"summary_reason": "Failed to parse reflection.", "cited_doc_ids": []}
    except Exception as e:
        logger.error(f"Error parsing reflection: {e}")
        return {"summary_reason": f"Error: {e}", "cited_doc_ids": []}

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
