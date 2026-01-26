import sys
import os
from datetime import datetime
from uuid import uuid4

# Add project root to path
sys.path.append(os.getcwd())

from src.graph.builder import create_rag_graph
from src.graph.state import GraphState

def test_rag_graph():
    print("=== Testing LangGraph RAG Revision ===")
    
    # 1. Initialize Graph
    app = create_rag_graph()
    
    # 2. Setup Configuration (thread_id for memory)
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 3. Test Case 1: Standard Retrieval with Date Filter
    # Set a target date (e.g., end of 2023)
    target_date = datetime(2023, 12, 31)
    question = "What are the key AI trends mentioned in late 2023?"
    
    print(f"\n[Test 1] Question: {question}")
    print(f"Target Date: {target_date.isoformat()}")
    
    initial_state = {
        "question": question,
        "target_date": target_date,
        "messages": []
    }
    
    # Run graph
    result = app.invoke(initial_state, config=config)
    
    print("\n--- Results ---")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Number of docs retrieved: {len(result['context'])}")
    
    # Verify date filter
    for i, doc in enumerate(result['context']):
        doc_date_str = doc.metadata.get("date")
        if doc_date_str:
            doc_date = datetime.fromisoformat(doc_date_str)
            if doc_date > target_date:
                print(f"❌ ERROR: Document {i} has date {doc_date_str} which is AFTER target_date!")
            else:
                print(f"✅ Doc {i} date: {doc_date_str} (Valid)")
        else:
            print(f"⚠️ Doc {i} has no date metadata.")

    # 4. Test Case 2: Multi-turn Conversation
    follow_up = "Can you elaborate on the first point?"
    print(f"\n[Test 2] Follow-up Question: {follow_up}")
    
    follow_up_state = {
        "question": follow_up,
        "target_date": target_date # Keep same date
    }
    
    # Invoke again with same config (thread_id)
    result_2 = app.invoke(follow_up_state, config=config)
    
    print("\n--- Follow-up Results ---")
    print(f"Answer: {result_2['answer'][:200]}...")
    print(f"Committee Views: {result_2.get('committee_views', {}).keys()}")
    if 'Value' in result_2.get('committee_views', {}):
        view = result_2['committee_views']['Value']
        print(f"Value View Magnitude (Q): {view['q_value']}")
        print(f"Value Uncertainty (Omega): {view['omega_value']}")
    
    print(f"History length (messages): {len(result_2['messages'])}")
    
    if len(result_2['messages']) > 2:
        print("✅ Multi-turn history preserved.")
    else:
        print("❌ Multi-turn history NOT preserved.")

    # 5. Test Case 3: Training Mode with Reflection
    print("\n[Test 3] Testing Training Mode with Reflection")
    train_question = "How should we allocate to the Value factor given current inflation?"
    train_state = {
        "question": train_question,
        "target_date": target_date,
        "is_training": True,
        "actual_return": 0.05, # Positive return
        "target_factor": "Value"
    }
    
    # We use a new thread to avoid confusion with previous chat history
    config_train = {"configurable": {"thread_id": str(uuid4())}}
    result_train = app.invoke(train_state, config=config_train)
    
    print("\n--- Training Results ---")
    print(f"Answer: {result_train['answer'][:100]}...")
    if "reflections" in result_train:
        reflection = result_train["reflections"].get("Value", {})
        print(f"✅ Reflection Generated: {reflection.get('summary_reason', 'No reason')[:100]}...")
        print(f"Cited Docs: {reflection.get('cited_doc_ids', [])}")
    else:
        print("❌ Reflection NOT generated.")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    # Ensure environment variables are loaded if needed (though config handles some)
    # This script assumes OPENAI_API_KEY is in .env or environment
    try:
        test_rag_graph()
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
