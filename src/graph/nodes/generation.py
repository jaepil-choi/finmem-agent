import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from src.graph.state import GraphState
from src.config import settings

logger = logging.getLogger(__name__)

def generation_node(state: GraphState) -> GraphState:
    """
    Node that generates an answer based on retrieved context and history.
    """
    logger.info("Generating answer based on context.")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini", # Using a fast model by default
        api_key=settings.OPENAI_API_KEY,
        temperature=0
    )
    
    # Prepare context string
    context_text = "\n\n".join([
        f"Document (Date: {doc.metadata.get('date', 'Unknown')}):\n{doc.page_content}"
        for doc in state["context"]
    ])
    
    # Define prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst assistant. Use the provided context to answer the user's question. "
                   "The context is strictly filtered to be on or before the target date: {target_date}. "
                   "If you don't know the answer based on context, say you don't know.\n\n"
                   "Context:\n{context}"),
        ("placeholder", "{messages}"),
        ("human", "{question}")
    ])
    
    # Format prompt
    chain = prompt | llm
    
    # Invoke LLM
    response = chain.invoke({
        "question": state["question"],
        "context": context_text,
        "target_date": state["target_date"].isoformat(),
        "messages": state["messages"]
    })
    
    # Return updated state
    return {
        "answer": response.content,
        "messages": [HumanMessage(content=state["question"]), response]
    }
