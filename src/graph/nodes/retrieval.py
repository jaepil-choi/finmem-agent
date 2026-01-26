import logging
from src.graph.state import GraphState
from src.rag.strategies.finmem import FinMemRAG

logger = logging.getLogger(__name__)

def retrieval_node(state: GraphState) -> GraphState:
    """
    Node that retrieves documents using the FinMemRAG strategy.
    Combines Similarity, Recency, and Importance for re-ranking.
    """
    logger.info(f"Retrieving documents for query: {state['question']} at target_date: {state['target_date']}")
    
    # Initialize the strategy
    strategy = FinMemRAG()
    
    # Perform retrieval with FinMem composite scoring
    documents = strategy.retrieve(
        query=state["question"],
        target_date=state["target_date"],
        k=5 
    )
    
    logger.info(f"Retrieved {len(documents)} documents.")
    
    # Return updated state
    return {"context": documents}
