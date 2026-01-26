import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import settings

class DailySummarizer:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 유능한 금융 시장 분석가입니다. 
제공된 당일의 시장 리포트들을 바탕으로, 오늘의 핵심 시장 상황(Daily News Summary)을 요약하십시오.
핵심 경제 지표, 주요 이벤트, 섹터별 흐름, 그리고 팩터 투자(가치, 성장, 모멘텀 등)에 영향을 줄 수 있는 뉴스에 집중하십시오.
요약은 불렛 포인트 형식으로 작성하며, 한국어로 답변하십시오."""),
            ("user", "오늘의 리포트 목록:\n\n{reports_text}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def summarize(self, reports: List[Dict[str, Any]]) -> str:
        """
        Summarizes a list of report documents into a daily news summary.
        """
        if not reports:
            return "오늘 제공된 시장 리포트가 없습니다."

        # Prepare reports text for the prompt
        formatted_reports = []
        for i, doc in enumerate(reports):
            content = f"Report {i+1} (Source: {doc.get('filename', 'Unknown')}):\n{doc.get('full_text', '')[:2000]}..." # Truncate if very long
            formatted_reports.append(content)
        
        reports_text = "\n\n".join(formatted_reports)
        
        try:
            summary = self.chain.invoke({"reports_text": reports_text})
            return summary
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return f"요약 중 오류가 발생했습니다: {str(e)}"
