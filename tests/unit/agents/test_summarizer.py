import pytest
from langchain_core.language_models.fake import FakeListLLM
from src.agents.summarizer import DailySummarizer

class TestDailySummarizer:
    @pytest.fixture
    def summarizer(self):
        # Create summarizer with a fake LLM
        # This replaces the real ChatOpenAI without mocking the chain structure
        responses = ["Mocked summary: Market is volatile today."]
        fake_llm = FakeListLLM(responses=responses)
        
        sum_instance = DailySummarizer()
        # Inject the fake LLM into the chain
        # Chain is: prompt | llm | parser
        sum_instance.llm = fake_llm
        sum_instance.chain = sum_instance.prompt | fake_llm | sum_instance.chain.steps[-1]
        return sum_instance

    def test_summarize_no_reports(self, summarizer):
        summary = summarizer.summarize([])
        assert summary == "오늘 제공된 시장 리포트가 없습니다."

    def test_summarize_with_reports(self, summarizer):
        reports = [
            {"filename": "news1.pdf", "full_text": "Inflation is rising."},
            {"filename": "news2.pdf", "full_text": "Tech stocks are down."}
        ]
        
        summary = summarizer.summarize(reports)
        
        # Verify the fake response was returned
        assert "Market is volatile today" in summary
        # Verify it wasn't the default error/empty message
        assert "오류" not in summary
