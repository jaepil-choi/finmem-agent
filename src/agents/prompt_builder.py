from datetime import datetime
from typing import Optional, List, Any

class PromptBuilder:
    def __init__(self):
        self.target_date: Optional[datetime] = None
        self.daily_summary: str = "No daily summary available."
        self.factor_expertise: str = ""
        self.risk_profile: str = "Standard"
        self.retrieved_context: str = ""
        self.user_query: str = ""

    def set_target_date(self, target_date: datetime) -> 'PromptBuilder':
        self.target_date = target_date
        return self

    def set_daily_summary(self, summary: str) -> 'PromptBuilder':
        self.daily_summary = summary
        return self

    def set_factor_expertise(self, expertise: str) -> 'PromptBuilder':
        self.factor_expertise = expertise
        return self

    def set_risk_profile(self, profile: str) -> 'PromptBuilder':
        self.risk_profile = profile
        return self

    def set_retrieved_context(self, context: str) -> 'PromptBuilder':
        self.retrieved_context = context
        return self

    def set_user_query(self, query: str) -> 'PromptBuilder':
        self.user_query = query
        return self

    def build_system_prompt(self) -> str:
        """
        Assembles the system prompt using the components.
        """
        date_str = self.target_date.strftime("%Y-%m-%d") if self.target_date else "Unknown"
        
        prompt = f"""### Role & Expertise
당신은 금융 공학 및 팩터 투자 전문가입니다. {date_str} 시점의 데이터를 바탕으로 투자 견해를 도출해야 합니다.

### Simulated Current Date
오늘 날짜: {date_str}

### Today's Market News Summary
{self.daily_summary}

### Factor Expertise Context
{self.factor_expertise}

### Risk Profile
Your current risk profile is: {self.risk_profile}

### Important Instructions
1. 모든 판단은 반드시 오늘의 날짜({date_str})와 제공된 컨텍스트를 기준으로만 수행하십시오.
2. 미래 데이터(Look-ahead data)를 알고 있는 것처럼 행동하지 마십시오.
3. 제공된 'Today's Market News Summary'를 현재의 시장 환경으로 간주하고, 'Retrieved Context'를 당신의 장기 기억으로 활용하십시오.
"""
        return prompt

    def build_user_prompt(self) -> str:
        """
        Assembles the user prompt with the query and retrieved context.
        """
        prompt = f"""### Retrieved Context (Long-term Memory)
{self.retrieved_context}

### Question
{self.user_query}
"""
        return prompt
