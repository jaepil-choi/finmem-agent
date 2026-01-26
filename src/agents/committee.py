import logging
import numpy as np
from typing import List, Dict, Any
from src.agents.agent import BaseAgent
from src.agents.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

class FactorCommittee:
    """
    Represents a committee of identical experts for a specific factor theme.
    """
    
    def __init__(
        self,
        factor_theme: str,
        num_agents: int = 5,
    ):
        self.factor_theme = factor_theme
        self.num_agents = num_agents
        # Create N identical experts
        self.agents = [
            BaseAgent(agent_id=f"{factor_theme}_expert_{i}")
            for i in range(num_agents)
        ]

    def aggregate_views(
        self, 
        question: str, 
        context: str, 
        prompt_builder: PromptBuilder,
        history: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Collects votes and quantifies uncertainty.
        """
        votes = []
        confidences = []
        reasonings = []
        
        logger.info(f"Committee for {self.factor_theme} is voting with {self.num_agents} members.")
        
        for agent in self.agents:
            view = agent.generate_view(question, context, prompt_builder, history)
            votes.append(view.get("vote", 0))
            confidences.append(view.get("confidence", 0))
            reasonings.append(f"[{agent.agent_id}]: {view.get('reasoning', 'No reasoning provided.')}")
            
        q_value = float(np.mean(votes))
        omega_value = float(np.var(votes)) if len(votes) > 1 else 0.0
        avg_confidence = float(np.mean(confidences))
        
        return {
            "factor_theme": self.factor_theme,
            "q_value": q_value,
            "omega_value": omega_value,
            "avg_confidence": avg_confidence,
            "individual_votes": votes,
            "consolidated_reasoning": "\n".join(reasonings)
        }
