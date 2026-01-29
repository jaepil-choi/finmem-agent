from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from src.graph.state import GraphState
from src.graph.nodes.retrieval import retrieval_node
from src.graph.nodes.analyst import analyst_node
from src.graph.nodes.generation import generation_node
from src.graph.nodes.reflection import reflection_node
from src.graph.nodes.memory import memory_update_node

def should_reflect(state: GraphState):
    """
    Conditional edge to determine if reflection is needed and fan-out if so.
    """
    if not state.get("is_training", False):
        return END
    
    # Fan-out to reflection_node for each factor committee
    committee_views = state.get("committee_views", {})
    actual_returns = state.get("actual_returns", {})
    
    sends = []
    for factor_name, view in committee_views.items():
        # Map the factor display name back to the internal key if needed
        # For now, we assume committee_views keys match actual_returns keys or display names
        # Note: Backtester will ensure actual_returns contains correct keys.
        actual_ret = actual_returns.get(factor_name)
        if actual_ret is not None:
            sends.append(Send("reflect", {
                "factor_name": factor_name,
                "actual_return": actual_ret,
                "view": view,
                "target_date": state["target_date"],
                "context": state["context"]
            }))
    
    return sends if sends else END

def create_rag_graph():
    """
    Creates and compiles the RAG StateGraph with Agentic Committees and Reflection.
    """
    # Initialize the graph with our state definition
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("analyze", analyst_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_node("update_memory", memory_update_node)
    
    # Define edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "generate")
    
    # Conditional Fan-out after generate
    workflow.add_conditional_edges(
        "generate",
        should_reflect,
        ["reflect", END]
    )
    
    # All reflection nodes merge back to update_memory
    workflow.add_edge("reflect", "update_memory")
    workflow.add_edge("update_memory", END)
    
    # Initialize memory checkpointer for multi-turn conversations
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app
