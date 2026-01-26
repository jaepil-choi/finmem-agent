import unittest
from datetime import datetime
from src.agents.prompt_builder import PromptBuilder

class TestPromptBuilder(unittest.TestCase):
    def test_builder_pattern(self):
        target_date = datetime(2024, 1, 1)
        summary = "Value stocks are cheap."
        query = "Should I buy value?"
        context = "Historical data shows value wins."
        
        builder = PromptBuilder()
        builder.set_target_date(target_date)\
               .set_daily_summary(summary)\
               .set_user_query(query)\
               .set_retrieved_context(context)
        
        sys_prompt = builder.build_system_prompt()
        user_prompt = builder.build_user_prompt()
        
        self.assertIn("2024-01-01", sys_prompt)
        self.assertIn(summary, sys_prompt)
        self.assertIn(query, user_prompt)
        self.assertIn(context, user_prompt)

if __name__ == "__main__":
    unittest.main()
