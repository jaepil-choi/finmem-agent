import logging
from src.config import settings
from src.graph.state import GraphState
from src.agents.committee import FactorCommittee
from src.agents.prompt_builder import PromptBuilder
from src.agents.risk_manager import select_risk_profile

logger = logging.getLogger(__name__)

def analyst_node(state: GraphState) -> GraphState:
    """
    Node that coordinates factor committees using YAML-based expertise and risk profiles.
    """
    target_factor = state.get("target_factor", "value").lower()
    logger.info(f"Executing Analyst Node for factor: {target_factor}")
    
    # 1. Access Configurations via global settings
    expertise_config = settings.factor_expertise
    risk_config = settings.risk_profiles

    # 2. Extract specific expertise and risk profile
    expertise = expertise_config.get(target_factor, {})
    
    # Dynamic risk profile selection based on performance and mode
    is_training = state.get("is_training", False)
    cumulative_return = state.get("cumulative_return", 0.0)
    
    risk_key = select_risk_profile(is_training, cumulative_return)
    risk_profile = risk_config.get(risk_key, risk_config.get("balanced", {}))
    risk_profile['name'] = risk_profile.get('name', risk_key.replace('_', ' ').capitalize())

    if not expertise:
        logger.warning(f"No expertise found for factor: {target_factor}. Using fallback.")
        expertise = {"name": target_factor.capitalize(), "definition": "General factor analysis."}

    # 3. Initialize PromptBuilder with structured data
    builder = PromptBuilder()
    builder.set_target_date(state["target_date"])\
           .set_factor_expertise(expertise)\
           .set_risk_profile(risk_profile)\
           .set_user_query(state["question"])
    
    # 4. Initialize and Execute Committee (5 agents as requested)
    committee = FactorCommittee(
        factor_theme=expertise.get('name', target_factor.capitalize()),
        num_agents=5
    )
    
    # Prepare context string from retrieved documents
    context_text = "\n\n".join([
        f"Document (Source: {doc.metadata.get('filename', 'Unknown')}, Date: {doc.metadata.get('date', 'Unknown')}):\n{doc.page_content}"
        for doc in state["context"]
    ])
    
    # Aggregate views
    view_result = committee.aggregate_views(
        question=state["question"],
        context=context_text,
        prompt_builder=builder,
        history=state["messages"]
    )
    
    # 5. Update state
    committee_views = state.get("committee_views", {})
    committee_views[expertise.get('name', target_factor.capitalize())] = view_result
    
    return {"committee_views": committee_views}
