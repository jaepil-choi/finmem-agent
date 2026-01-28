import numpy as np
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class BlackLittermanOptimizer:
    """
    Implements the Black-Litterman model to optimize factor weights 
    by combining market priors with agent-generated views and confidence.
    """
    
    def __init__(self, tau: float = 0.05, max_view_return: float = 0.05):
        """
        Initializes the optimizer.
        :param tau: Scaling factor for the uncertainty of the prior distribution.
        :param max_view_return: The maximum expected return assigned to a full +1/-1 vote.
        """
        self.tau = tau
        self.max_view_return = max_view_return

    def optimize(
        self,
        factors: List[str],
        prior_weights: np.ndarray,
        views_q: np.ndarray,
        uncertainty_omega: np.ndarray,
        risk_aversion: float = 2.5
    ) -> np.ndarray:
        """
        Combines priors and views using the Black-Litterman formula.
        
        Formula:
        E[R] = [ (tau * Sigma)^-1 + P^T * Omega^-1 * P ]^-1 * [ (tau * Sigma)^-1 * Pi + P^T * Omega^-1 * Q ]
        
        Simplified for Factor Views (where P is Identity since agents vote directly on factors):
        E[R] = [ (tau * Sigma)^-1 + Omega^-1 ]^-1 * [ (tau * Sigma)^-1 * Pi + Omega^-1 * Q ]
        
        For this simplified implementation, we assume Sigma is estimated from Pi 
        or provided as an identity matrix if not available.
        """
        num_factors = len(factors)
        
        # P matrix (Identity matrix if views are directly on factors)
        P = np.eye(num_factors)
        
        # Pi: Equilibrium expected returns (implied from prior weights)
        # We assume Pi = delta * Sigma * w_prior
        # For simplicity in this agentic framework, we use Pi = prior_weights
        Pi = prior_weights
        
        # Q: View vector
        Q = views_q
        
        # Omega: Diagonal matrix of view uncertainty
        # If omega is very small, confidence is high. 
        # We ensure a minimum value to avoid singularity.
        Omega = np.diag(np.maximum(uncertainty_omega, 1e-6))
        
        # Sigma: Covariance matrix (Prior uncertainty)
        # For this stage, we assume an identity matrix or a diagonal matrix based on variance
        Sigma = np.eye(num_factors)
        
        # Black-Litterman Posterior Return Calculation
        # term1 = (tau * Sigma)^-1 + P^T * Omega^-1 * P
        inv_tau_sigma = np.linalg.inv(self.tau * Sigma)
        inv_omega = np.linalg.inv(Omega)
        
        term1 = inv_tau_sigma + P.T @ inv_omega @ P
        term2 = inv_tau_sigma @ Pi + P.T @ inv_omega @ Q
        
        # posterior_returns = inv(term1) * term2
        posterior_returns = np.linalg.solve(term1, term2)
        
        # Final weights: w = (1/delta) * inv(Sigma) * posterior_returns
        # For simplicity, we normalize the posterior returns to get weights
        new_weights = posterior_returns / np.sum(np.abs(posterior_returns))
        
        return new_weights

    def get_optimized_weights(
        self, 
        committee_results: Dict[str, Dict[str, Any]], 
        target_factors: List[str]
    ) -> Dict[str, float]:
        """
        Helper to prepare inputs from committee results and return optimized weights.
        """
        num_factors = len(target_factors)
        prior_weights = np.ones(num_factors) / num_factors # Equal weight prior
        
        views_q = np.zeros(num_factors)
        uncertainty_omega = np.ones(num_factors) # Default high uncertainty
        
        for i, factor in enumerate(target_factors):
            # Find the result for this factor theme
            # Note: factor_expertise uses keys like 'value', while committee names might be 'Value Factor'
            # We look for partial matches
            res = None
            for theme_name, data in committee_results.items():
                if factor.lower() in theme_name.lower():
                    res = data
                    break
            
            if res:
                # Scale raw votes (-1 to 1) to expected returns
                views_q[i] = res.get('q_value', 0.0) * self.max_view_return
                
                # Scale variance by the square of max_view_return (0.0025)
                # This ensures the uncertainty is in the same units as the squared return
                raw_omega = res.get('omega_value', 1.0)
                uncertainty_omega[i] = raw_omega * (self.max_view_return ** 2)
        
        optimized_w = self.optimize(target_factors, prior_weights, views_q, uncertainty_omega)
        
        return {factor: float(w) for factor, w in zip(target_factors, optimized_w)}
