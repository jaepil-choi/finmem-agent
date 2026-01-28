import logging
import sys
import os
from datetime import datetime
from typing import List

# Add project root to path
sys.path.append(os.getcwd())

from src.db.jkp_repository import JKPRepository
from src.db.repository import ReportRepository
from src.core.utils.regime import RegimeCalculator
from src.agents.prompt_builder import PromptBuilder
from src.agents.factory import CommitteeFactory
from src.rag.strategies.finmem import FinMemRAG
from src.config import settings

# Configure logging to be less noisy
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def run_detailed_demo(target_date_str: str, factor_keys: List[str]):
    """
    Runs a detailed demo for multiple factors, showing prompts and individual agent votes.
    """
    target_date = datetime.fromisoformat(target_date_str)
    
    print(f"\n{'='*80}")
    print(f"{'MULTI-FACTOR DETAILED DEMO':^80}")
    print(f"{'='*80}")
    print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Factors: {', '.join(factor_keys)}")
    print(f"{'='*80}\n")

    # 1. Initialize Repositories
    jkp_repo = JKPRepository()
    report_repo = ReportRepository()
    regime_calc = RegimeCalculator()
    rag_strategy = FinMemRAG()

    # 2. Fetch Today's Reports (Daily News)
    print(f"[Step 1] Fetching Today's Reports from MongoDB...")
    today_reports = report_repo.get_reports_by_date(target_date, collections=["daily"])
    if not today_reports:
        print(f"⚠️ Warning: No daily reports found for {target_date_str}. Demo will proceed with empty summary.")
        daily_summary = "No daily news available for this date."
    else:
        daily_summary = "\n\n".join([f"--- {r['filename']} ---\n{r['text'][:1000]}..." for r in today_reports])
        print(f"✅ Found {len(today_reports)} daily reports.")

    # 3. Process Each Factor
    for factor_key in factor_keys:
        print(f"\n\n{'#'*80}")
        print(f"{f'ANALYZING FACTOR: {factor_key.upper()}':^80}")
        print(f"{'#'*80}")

        # A. Fetch History & Calculate Regime Metrics
        print(f"\n[Step 2] Calculating Regime Metrics for '{factor_key}'...")
        history = jkp_repo.get_factor_history(factor_key, target_date)
        if history.empty:
            print(f"❌ Error: No history found for factor '{factor_key}'. Skipping.")
            continue
        regime_metrics = regime_calc.calculate(history)
        print(f"✅ Metrics calculated: Hurst={regime_metrics.get('hurst_exponent', 0):.4f}, ER={regime_metrics.get('efficiency_ratio', 0):.4f}")

        # B. Retrieve Long-term Memory (RAG)
        print(f"\n[Step 3] Retrieving Long-term Memory via FinMemRAG...")
        query = f"Predict the directional movement of {factor_key}."
        context_docs = rag_strategy.retrieve(query=query, target_date=target_date, k=3)
        context_text = ""
        for i, doc in enumerate(context_docs):
            context_text += f"[doc_{i}] (Source: {doc.metadata.get('filename', 'Unknown')}):\n{doc.page_content[:500]}...\n\n"
        print(f"✅ Retrieved {len(context_docs)} relevant documents.")

        # C. Build Final Prompt
        print(f"\n[Step 4] Assembling Final Prompt...")
        expertise = settings.factor_expertise.get(factor_key, {"name": factor_key.capitalize()})
        builder = PromptBuilder()
        builder.set_target_date(target_date)\
               .set_factor_expertise(expertise)\
               .set_risk_profile({"name": "Balanced", "instruction": "Maintain a objective and data-driven perspective."})\
               .set_regime_data(regime_metrics)\
               .set_daily_summary(daily_summary)\
               .set_user_query(query)
        
        system_prompt = builder.build_system_prompt()
        user_prompt = builder.build_user_prompt(query, context_text)

        print("\n" + "-"*30 + " [FINAL SYSTEM PROMPT] " + "-"*30)
        print(system_prompt)
        print("-"*30 + " [FINAL USER PROMPT] " + "-"*30)
        print(user_prompt)
        print("-"*83)

        # D. Execute Committee
        print(f"\n[Step 5] Calling Committee (5 Agents) for '{factor_key}'...")
        committee = CommitteeFactory.create_committee(factor_key, num_agents=5)
        
        # We manually call agents to show individual votes as requested
        votes = []
        for i, agent in enumerate(committee.agents):
            print(f"  > Agent {i+1}/5 is reasoning...", end="\r")
            # Replicate agent.generate_view logic to capture raw structured output
            messages = [("system", system_prompt), ("human", user_prompt)]
            response = agent.model.invoke(messages) # This is the structured CommitteeView object
            votes.append(response)
            print(f"  ✅ Agent {i+1}/5 Voted: {response.vote:^2} | Confidence: {response.confidence:.2f}")
            print(f"     Reasoning: {response.reasoning}")

        # E. Show Aggregated Results
        import numpy as np
        vote_values = [v.vote for v in votes]
        q_value = float(np.mean(vote_values))
        omega_value = float(np.var(vote_values))
        
        print(f"\n{'='*20} AGGREGATED COMMITTEE VIEW {'='*20}")
        print(f"Factor: {factor_key.upper()}")
        print(f"Final View (Q): {q_value:+.4f} (Directional Bias)")
        print(f"Uncertainty (Ω): {omega_value:.4f} (Disagreement Level)")
        print(f"Individual Votes: {vote_values}")
        print(f"{'='*67}")

if __name__ == "__main__":
    # date must exist in JKP and MongoDB
    # Let's use 2024-03-04 which is likely to have data in the requested range
    run_detailed_demo("2024-03-04", ["value", "momentum"])
