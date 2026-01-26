import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.db.repository import ReportRepository
from src.agents.summarizer import DailySummarizer
from src.agents.prompt_builder import PromptBuilder
from src.rag.strategies.naive import NaiveRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_e2e_rag_demo(target_date_str: str, factor_theme: str):
    target_date = datetime.fromisoformat(target_date_str)
    query = f"오늘 뉴스를 통해 봤을 때 앞으로 {factor_theme} 팩터는 오를거 같아 아니면 내릴 것 같아?"
    
    print(f"\n{'='*20} E2E RAG DEMO {'='*20}")
    print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Factor Theme: {factor_theme}")
    print(f"User Query: {query}")
    print(f"{'='*54}\n")

    # 1. Initialize Components
    repo = ReportRepository()
    summarizer = DailySummarizer()
    builder = PromptBuilder()
    rag = NaiveRAG()
    llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY)

    # 2. Step 1: Retrieve Daily Reports from MongoDB
    print("[Step 1] Fetching raw reports for the day...")
    daily_reports = repo.get_reports_by_date(target_date, collections=["daily"])
    print(f" - Found {len(daily_reports)} reports.")

    # 3. Step 2: Summarize Daily Reports
    print("[Step 2] Generating Daily News Summary...")
    daily_summary = summarizer.summarize(daily_reports)
    print(f"\n--- Daily News Summary ---\n{daily_summary}\n")

    # 4. Step 3: Perform RAG (Date-constrained, top 1)
    print("[Step 3] Retrieving long-term memory (RAG)...")
    # We use the factor theme itself as a query for memory to find relevant context
    rag_results = rag.retrieve(factor_theme, target_date=target_date, k=1)
    
    context_text = "관련된 과거 기록이 없습니다."
    if rag_results:
        doc = rag_results[0]
        context_text = f"Source: {doc.metadata.get('filename')} (Date: {doc.metadata.get('date')})\nContent: {doc.page_content}"
        print(f" - Found relevant memory from {doc.metadata.get('date')}")
    else:
        print(" - No relevant memory found.")

    # 5. Step 4: Build Final Prompts
    print("[Step 4] Assembling final prompt...")
    builder.set_target_date(target_date)\
           .set_daily_summary(daily_summary)\
           .set_retrieved_context(context_text)\
           .set_user_query(query)\
           .set_factor_expertise(f"{factor_theme} 팩터에 대한 전문 지식을 활용하십시오.")\
           .set_risk_profile("Balanced")

    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt()

    # 6. Step 5: Generate Final Answer
    print("[Step 5] Generating final analysis using LLM...")
    messages = [
        ("system", system_prompt),
        ("user", user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        answer = response.content
        print(f"\n--- Final Agent Analysis ---\n{answer}\n")
        
        if rag_results:
            print(f"{'='*20} SOURCE EVIDENCE {'='*20}")
            print(f"The analysis above was supported by the following FAISS memory:")
            print(f" - Filename: {rag_results[0].metadata.get('filename')}")
            print(f" - Date: {rag_results[0].metadata.get('date')}")
            print(f" - Tier: {rag_results[0].metadata.get('tier', 'N/A')}")
            print(f" - Similarity Score: {rag_results[0].metadata.get('score', 'N/A'):.4f}")
            print(f" - Content Chunk: {rag_results[0].page_content[:200]}...")
            print(f"{'='*54}\n")
            
    except Exception as e:
        print(f"Error during final LLM call: {e}")

    repo.close()

if __name__ == "__main__":
    # Example: Use a date that has data
    run_e2e_rag_demo("2025-04-05", "가치")
