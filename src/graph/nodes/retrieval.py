import logging
from src.graph.state import GraphState
from src.rag.strategies.finmem import FinMemRAG

logger = logging.getLogger(__name__)

def retrieval_node(state: GraphState) -> GraphState:
    """
    Node that retrieves documents using the FinMemRAG strategy and fetches today's reports.
    Combines Similarity, Recency, and Importance for re-ranking.
    """
    logger.info(f"Retrieving documents for query: {state['question']} at target_date: {state['target_date']}")
    
    # 1. Fetch Today's Reports from MongoDB
    from src.db.repository import ReportRepository
    repo = ReportRepository()
    today_reports = repo.get_reports_by_date(state["target_date"], collections=["daily"])
    
    daily_summary = "No daily news available for this date."
    if today_reports:
        # Combine headlines or summaries if multiple
        daily_summary = "\n\n".join([f"--- {r['filename']} ---\n{r['text'][:1000]}..." for r in today_reports])
    
    # 2. Perform retrieval with FinMem composite scoring
    strategy = FinMemRAG()
    documents = strategy.retrieve(
        query=state["question"],
        target_date=state["target_date"],
        k=5 
    )
    
    logger.info(f"Retrieved {len(documents)} documents. Daily news: {'Found' if today_reports else 'None'}")
    
    # Return updated state
    return {
        "context": documents,
        "daily_summary": daily_summary
    }
