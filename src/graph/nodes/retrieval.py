import logging
from src.graph.state import GraphState
from src.rag.strategies.naive import NaiveRAG

logger = logging.getLogger(__name__)

def retrieval_node(state: GraphState) -> GraphState:
    """
    Node that retrieves documents using the NaiveRAG strategy.
    Strictly respects the target_date from the state.
    """
    logger.info(f"Retrieving documents for query: {state['question']} at target_date: {state['target_date']}")
    
    # Initialize the strategy
    # In a more complex setup, this could be injected or selected based on state
    strategy = NaiveRAG()
    
    # Perform retrieval with HARD date filter
    documents = strategy.retrieve(
        query=state["question"],
        target_date=state["target_date"],
        k=5 # Default k
    )
    
    logger.info(f"Retrieved {len(documents)} documents.")
    
    # Return updated state
    return {"context": documents}
