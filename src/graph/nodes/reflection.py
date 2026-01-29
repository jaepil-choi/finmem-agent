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
    Node that performs self-reflection and suggests memory importance updates.
    Optimized for parallel execution (returns updates instead of applying them).
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
    feedback_type = "NEUTRAL"
    if q_value > 0 and actual_return > 0:
        feedback = 1.0
        feedback_type = "REWARD (Correct Bullish)"
    elif q_value < 0 and actual_return < 0:
        feedback = 1.0
        feedback_type = "REWARD (Correct Bearish)"
    elif q_value > 0 and actual_return < 0:
        feedback = -1.0
        feedback_type = "PENALTY (False Bullish)"
    elif q_value < 0 and actual_return > 0:
        feedback = -1.0
        feedback_type = "PENALTY (False Bearish)"
    elif q_value == 0 and abs(actual_return) > 0.01:
        feedback = -0.5
        feedback_type = "PENALTY (Missed Opportunity)"
    
    emoji = "REWARD" if feedback > 0 else "PENALTY" if feedback < 0 else "NEUTRAL"
    print(f"\n    [{emoji}] Reflection Result for {factor_name}: {feedback_type}")
    print(f"      - Prediction (Q): {q_value:+.4f}, Actual Return: {actual_return:+.4f}")
    print(f"      - Feedback Score: {feedback:+.1f}")
    print(f"      - Summary: {reflection_data.get('summary_reason')}")

    # 4. Handle Misleading Cases
    cited_ids = reflection_data.get("cited_doc_ids", [])
    if not cited_ids and feedback < 0:
        print(f"      - No documents cited for penalty, checking for misleading info...")
        misleading_prompt = builder.build_misleading_reflection_prompt(context_text)
        response = structured_llm.invoke([
            ("system", "You are a senior financial analyst performing a post-investment review."),
            ("human", misleading_prompt)
        ])
        reflection_data = response.model_dump()
        cited_ids = reflection_data.get("cited_doc_ids", [])
        if cited_ids:
            print(f"      - Found {len(cited_ids)} misleading documents.")

    # 5. Prepare Memory Updates (collected by the graph)
    memory_updates = []
    if cited_ids and feedback != 0:
        for doc_id in cited_ids:
            doc = doc_map.get(doc_id)
            if doc:
                memory_updates.append({
                    "doc_id": doc_id,
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page_content": doc.page_content,
                    "source_id": doc.metadata.get("source_id"),
                    "tier": doc.metadata.get("tier"),
                    "feedback": feedback
                })

    return {
        "reflections": {factor_name: reflection_data},
        "memory_updates": memory_updates
    }
