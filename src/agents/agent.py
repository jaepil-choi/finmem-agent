import logging
import json
import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings
from src.agents.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base Agent class that defines the core reasoning capabilities for a financial analyst.
    Now utilizes PromptBuilder for sophisticated system prompt assembly.
    """
    
    def __init__(
        self, 
        agent_id: str,
        model_name: str = "gpt-4o-mini"
    ):
        self.agent_id = agent_id
        self.model = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.7 # Ensemble diversity
        )

    def generate_view(
        self, 
        question: str, 
        context: str, 
        prompt_builder: PromptBuilder,
        history: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Generates a view using the provided PromptBuilder to assemble the context.
        """
        logger.info(f"Agent {self.agent_id} generating view.")
        
        system_prompt = prompt_builder.build_system_prompt()
        user_prompt = prompt_builder.build_user_prompt(question, context)
        
        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]
        
        # In a real scenario, we might add history between system and human
        # but for now we keep it simple as per requirements.
        
        response = self.model.invoke(messages)
        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parses the LLM output into a structured dictionary."""
        try:
            # Look for JSON block
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                # Replace single quotes with double quotes for valid JSON if necessary
                json_str = match.group()
                return json.loads(json_str)
            return {"vote": 0, "confidence": 0, "reasoning": "Failed to parse JSON from response."}
        except Exception as e:
            logger.error(f"Error parsing agent response: {e}")
            return {"vote": 0, "confidence": 0, "reasoning": f"Parsing Error: {str(e)}"}
