import logging
from src.config import settings
from src.graph.state import GraphState
from src.agents.committee import FactorCommittee
from src.agents.prompt_builder import PromptBuilder
from src.agents.risk_manager import select_risk_profile
from src.agents.factory import CommitteeFactory

logger = logging.getLogger(__name__)

def analyst_node(state: GraphState) -> GraphState:
    """
    Node that coordinates factor committees using YAML-based expertise and risk profiles.
    Modified to support multi-factor analysis if target_factor is 'all'.
    """
    target_factor_input = state.get("target_factor", "value").lower()
    is_training = state.get("is_training", False)
    cumulative_return = state.get("cumulative_return", 0.0)
    
    # 1. Determine which factors to analyze
    if target_factor_input == "all":
        factor_keys = list(settings.factor_expertise.keys())
    else:
        # Support comma-separated list or single factor
        factor_keys = [f.strip() for f in target_factor_input.split(",")]

    logger.info(f"Executing Analyst Node for factors: {factor_keys}")
    
    # 2. Risk Profile Selection (Common for all committees in this turn)
    risk_config = settings.risk_profiles
    risk_key = select_risk_profile(is_training, cumulative_return)
    risk_profile = risk_config.get(risk_key, risk_config.get("balanced", {}))
    risk_profile['name'] = risk_profile.get('name', risk_key.replace('_', ' ').capitalize())

    # 3. Prepare common context text
    context_text = ""
    for i, doc in enumerate(state["context"]):
        doc_id = f"doc_{i}"
        doc.metadata["temp_id"] = doc_id
        context_text += f"[{doc_id}] (Source: {doc.metadata.get('filename', 'Unknown')}, Date: {doc.metadata.get('date', 'Unknown')}):\n{doc.page_content}\n\n"

    # 4. Execute Committees (Sequential for now, can be parallelized later)
    committee_views = state.get("committee_views", {})
    
    for factor_key in factor_keys:
        expertise = settings.factor_expertise.get(factor_key)
        if not expertise:
            logger.warning(f"No expertise found for factor: {factor_key}. Skipping.")
            continue

        theme_name = expertise.get('name', factor_key.capitalize())
        logger.info(f"Running committee for: {theme_name}")

        # Initialize PromptBuilder for this factor
        builder = PromptBuilder()
        builder.set_target_date(state["target_date"])\
               .set_factor_expertise(expertise)\
               .set_risk_profile(risk_profile)\
               .set_user_query(state["question"])
        
        # Create Committee via Factory
        committee = CommitteeFactory.create_committee(factor_key, num_agents=5)
        
        # Aggregate views
        view_result = committee.aggregate_views(
            question=state["question"],
            context=context_text,
            prompt_builder=builder,
            history=state["messages"]
        )
        
        committee_views[theme_name] = view_result
    
    return {"committee_views": committee_views}
