def select_risk_profile(is_training: bool, cumulative_return: float) -> str:
    """
    Selects the risk profile key based on the current mode and cumulative return.
    Following the FinMem logic for test mode, and defaulting to balanced for training.
    """
    if is_training:
        return "balanced"
    
    # Path-dependent logic from FinMem
    if cumulative_return >= 0:
        return "risk_seeking"
    else:
        return "risk_averse"
