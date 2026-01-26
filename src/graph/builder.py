from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.graph.state import GraphState
from src.graph.nodes.retrieval import retrieval_node
from src.graph.nodes.analyst import analyst_node
from src.graph.nodes.generation import generation_node

def create_rag_graph():
    """
    Creates and compiles the RAG StateGraph with Agentic Committees.
    """
    # Initialize the graph with our state definition
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("analyze", analyst_node) # New Analyst Committee Node
    workflow.add_node("generate", generation_node)
    
    # Define edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)
    
    # Initialize memory checkpointer for multi-turn conversations
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app
