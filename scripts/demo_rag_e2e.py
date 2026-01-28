import logging
import uuid
from datetime import datetime
from src.graph.builder import create_rag_graph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_committee_demo(target_date_str: str, factor_theme: str):
    """
    Demonstrates the new LangGraph-based RAG pipeline with Agentic Committee views.
    """
    target_date = datetime.fromisoformat(target_date_str)
    query = f"Based on recent macro trends, what is your outlook for the {factor_theme} factor?"
    
    print(f"\n{'='*20} LANGGRAPH COMMITTEE DEMO {'='*20}")
    print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Factor Theme: {factor_theme}")
    print(f"User Query: {query}")
    print(f"{'='*54}\n")

    # 1. Initialize the Graph
    print("[Step 1] Initializing LangGraph RAG Pipeline...")
    app = create_rag_graph()
    
    # 2. Setup Configuration (thread_id for persistent state)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 3. Prepare Initial State
    initial_state = {
        "question": query,
        "target_date": target_date,
        "target_factor": factor_theme.lower(), # Normalize to internal key (lowercase)
        "messages": [],
        "cumulative_return": 0.05,     # Example: Start with positive return (Risk-Seeking)
        "is_training": False           # Demo defaults to Test mode logic
    }

    # 4. Execute the Graph
    print("[Step 2] Executing Graph (Retrieve -> Analyze -> Generate)...")
    try:
        final_state = app.invoke(initial_state, config=config)
        
        # 5. Display Committee Views (The Core Intelligence)
        print(f"\n{'='*20} COMMITTEE ANALYSIS (Q & OMEGA) {'='*20}")
        views = final_state.get("committee_views", {})
        
        factor_key = factor_theme.lower()
        if factor_key in views:
            view = views[factor_key]
            print(f"Factor Key: {view['factor_theme']}")
            print(f"View Magnitude (Q): {view['q_value']:+.2f} (Range: -1.0 to +1.0)")
            print(f"Uncertainty (Ω): {view['omega_value']:.4f} (Higher = More disagreement)")
            print(f"Avg Confidence: {view['avg_confidence']:.2%}")
            print(f"Individual Votes: {view['individual_votes']}")
            print(f"\n--- Consolidated Reasoning ---\n{view['consolidated_reasoning']}")
        else:
            print(f"Note: Specific view for '{factor_theme}' ({factor_key}) not found in views: {list(views.keys())}")

        # 6. Display Final LLM Answer
        print(f"\n{'='*20} FINAL AGENT RESPONSE {'='*20}")
        print(final_state["answer"])
        
        # 7. Evidence Tracking
        print(f"\n{'='*20} SOURCE EVIDENCE (DATE-FILTERED) {'='*20}")
        for i, doc in enumerate(final_state["context"]):
            print(f"[{i+1}] Source: {doc.metadata.get('filename', 'Unknown')}")
            print(f"    Date: {doc.metadata.get('date', 'Unknown')}")
            print(f"    Snippet: {doc.page_content[:150]}...")
            
        print(f"{'='*54}\n")

    except Exception as e:
        print(f"Error during Graph execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example: Test with a date and the Value factor
    # Make sure your VECTOR_DB_ROOT has data indexed!
    run_committee_demo("2023-12-31", "Value")
