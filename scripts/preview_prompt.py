import logging
from datetime import datetime
from src.agents.prompt_builder import PromptBuilder

def show_prompt_assembly():
    target_date = datetime(2025, 4, 5)
    factor_theme = "가치(Value)"
    query = f"오늘 뉴스를 통해 봤을 때 앞으로 {factor_theme} 팩터는 오를거 같아 아니면 내릴 것 같아?"
    
    # Mocked data to show assembly
    daily_summary = """
- 미국 고용 지표가 예상보다 강하게 발표됨에 따라 금리 인하 기대감이 후퇴함.
- 국채 금리 상승으로 인해 기술주(Growth)는 압박을 받고 있으며, 가치주(Value) 섹터 내 금융/에너지 업종이 상대적 강세를 보임.
- 인플레이션 우려가 지속되면서 시장의 변동성이 확대됨.
    """
    
    context_text = """
Source: kiwoom_weekly_20250328.pdf (Date: 2025-03-28)
Content: 가치 팩터는 경기 회복 국면에서 초과 수익을 기록하는 경향이 있음. 현재의 고금리 환경은 전통적인 가치주들에게 유리한 매크로 환경을 제공함.
    """

    print(f"\n{'='*20} PROMPT ASSEMBLY PREVIEW {'='*20}")
    print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    
    builder = PromptBuilder()
    builder.set_target_date(target_date)\
           .set_daily_summary(daily_summary)\
           .set_retrieved_context(context_text)\
           .set_user_query(query)\
           .set_factor_expertise(f"{factor_theme} 팩터에 대한 전문 지식을 활용하십시오.")\
           .set_risk_profile("Balanced")

    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt()

    print("\n--- [SYSTEM PROMPT] ---")
    print(system_prompt)
    print("\n--- [USER PROMPT] ---")
    print(user_prompt)
    print(f"{'='*54}\n")

if __name__ == "__main__":
    show_prompt_assembly()
