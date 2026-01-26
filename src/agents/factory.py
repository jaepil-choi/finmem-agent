import logging
from typing import Dict, Any, List
from src.config import settings
from src.agents.committee import FactorCommittee

logger = logging.getLogger(__name__)

class CommitteeFactory:
    """
    Factory for creating FactorCommittees based on YAML configuration.
    """
    
    @staticmethod
    def create_committee(factor_key: str, num_agents: int = 5) -> FactorCommittee:
        """
        Creates a single committee for a specific factor key.
        """
        expertise = settings.factor_expertise.get(factor_key)
        if not expertise:
            logger.warning(f"No expertise found for factor key: {factor_key}")
            return FactorCommittee(factor_theme=factor_key.capitalize(), num_agents=num_agents)
        
        factor_theme = expertise.get('name', factor_key.capitalize())
        return FactorCommittee(factor_theme=factor_theme, num_agents=num_agents)

    @staticmethod
    def create_all_committees(num_agents: int = 5) -> Dict[str, FactorCommittee]:
        """
        Creates committees for all factor themes defined in the configuration.
        """
        committees = {}
        for factor_key in settings.factor_expertise.keys():
            committees[factor_key] = CommitteeFactory.create_committee(factor_key, num_agents)
            
        logger.info(f"Created {len(committees)} factor committees.")
        return committees
