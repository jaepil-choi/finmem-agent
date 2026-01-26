import yaml
from datetime import datetime
from typing import Optional, List, Any, Dict

class PromptBuilder:
    def __init__(self):
        self.target_date: Optional[datetime] = None
        self.daily_summary: str = "No daily summary available."
        self.factor_expertise: Dict[str, Any] = {}
        self.risk_profile: Dict[str, Any] = {}
        self.retrieved_context: str = ""
        self.user_query: str = ""
        self.actual_return: float = 0.0

    def set_target_date(self, target_date: datetime) -> 'PromptBuilder':
        self.target_date = target_date
        return self

    def set_daily_summary(self, summary: str) -> 'PromptBuilder':
        self.daily_summary = summary
        return self

    def set_factor_expertise(self, expertise: Dict[str, Any]) -> 'PromptBuilder':
        self.factor_expertise = expertise
        return self

    def set_risk_profile(self, profile: Dict[str, Any]) -> 'PromptBuilder':
        self.risk_profile = profile
        return self

    def set_retrieved_context(self, context: str) -> 'PromptBuilder':
        self.retrieved_context = context
        return self

    def set_user_query(self, query: str) -> 'PromptBuilder':
        self.user_query = query
        return self

    def set_actual_return(self, actual_return: float) -> 'PromptBuilder':
        self.actual_return = actual_return
        return self

    def build_system_prompt(self) -> str:
        """
        Assembles the system prompt using the components, 
        conjoining core logic and strategy.
        """
        date_str = self.target_date.strftime("%Y-%m-%d") if self.target_date else "Unknown"
        
        # Format Factor Expertise
        expertise_name = self.factor_expertise.get('name', 'General')
        expertise_def = self.factor_expertise.get('definition', '')
        
        core_logic = self.factor_expertise.get('core_logic', {})
        rational = "\n- ".join(core_logic.get('rational', []))
        behavioral = "\n- ".join(core_logic.get('behavioral', []))
        
        strategy = self.factor_expertise.get('strategy', {})
        bullish = "\n- ".join(strategy.get('bullish', []))
        bearish = "\n- ".join(strategy.get('bearish', []))

        # Format Risk Profile
        risk_name = self.risk_profile.get('name', 'Standard')
        risk_instr = self.risk_profile.get('instruction', 'Maintain a balanced perspective.')

        prompt = f"""### Role & Expertise
You are a top-tier Quantitative Financial Analyst specialized in the '{expertise_name}'.
Your mission is to evaluate market conditions and provide a precise investment view (+1, 0, -1) for this factor.

### Domain Knowledge: {expertise_name}
- **Definition**: {expertise_def}
- **Core Logic (Rational)**:
- {rational}
- **Core Logic (Behavioral)**:
- {behavioral}

### Market Strategy Guidelines
- **Bullish Signals (When to Overweight)**:
- {bullish}
- **Bearish Signals (When to Underweight)**:
- {bearish}

### Current Risk Profile: {risk_name}
{risk_instr}

### Temporal Constraint (CRITICAL)
- **Simulated Today**: {date_str}
- All reasoning MUST be based on information available ON OR BEFORE this date. 
- NEVER use future knowledge.

### Output Requirement
You MUST respond in strict JSON format:
{{
  "vote": <+1 | 0 | -1>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<concise explanation referencing specific evidence>"
}}
"""
        return prompt

    def build_user_prompt(self, question: str, context: str) -> str:
        """
        Assembles the user prompt with the query and retrieved context.
        """
        prompt = f"""### Today's Market News Summary
{self.daily_summary}

### Long-term Memory (Retrieved Context)
{context}

### Investor Query
{question}
"""
        return prompt

    def build_reflection_prompt(self, context: str) -> str:
        """
        Assembles the reflection prompt for training mode.
        """
        date_str = self.target_date.strftime("%Y-%m-%d") if self.target_date else "Unknown"
        expertise_name = self.factor_expertise.get('name', 'General')

        prompt = f"""### Reflection Task: {expertise_name}
The current date is {date_str}. 

### Observed Market Fact
The actual return for the '{expertise_name}' factor for the period following {date_str} was: {self.actual_return:+.4f}

### Instructions
Given the provided 'Long-term Memory' (Retrieved Context), explain why the financial market fluctuation behaved like this. 
Identify which pieces of information were most relevant to this outcome.

### Long-term Memory (Retrieved Context)
{context}

### Output Requirement
You MUST respond in strict JSON format:
{{
  "summary_reason": "<concise explanation of why the market moved this way>",
  "cited_doc_ids": [<list of document IDs that were most relevant, e.g., ["doc_1", "doc_2"]>]
}}
"""
        return prompt

    def build_misleading_reflection_prompt(self, context: str) -> str:
        """
        Assembles a reflection prompt specifically focused on identifying 
        misleading or contradictory documents.
        """
        date_str = self.target_date.strftime("%Y-%m-%d") if self.target_date else "Unknown"
        expertise_name = self.factor_expertise.get('name', 'General')

        prompt = f"""### Post-Mortem Reflection Task: {expertise_name}
The current date is {date_str}. 

### Observed Market Fact
The actual return for the '{expertise_name}' factor for the period following {date_str} was: {self.actual_return:+.4f}

### The Problem
The agent committee failed to predict this outcome correctly (or was neutral when the market moved significantly).

### Instructions
Review the provided 'Long-term Memory' (Retrieved Context). Identify any documents that were **misleading**, **contradictory** to the actual outcome, or **incorrectly interpreted**. 
If you can find documents that might have caused the agent to make a wrong or indecisive prediction, list them.

### Long-term Memory (Retrieved Context)
{context}

### Output Requirement
You MUST respond in strict JSON format:
{{
  "summary_reason": "<explanation of how these documents were misleading or contradictory>",
  "cited_doc_ids": [<list of document IDs that were misleading/contradictory, e.g., ["doc_1", "doc_2"]>]
}}
"""
        return prompt
