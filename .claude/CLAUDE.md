# Ensemble-Driven Black-Litterman Allocator

## Project Overview

A **Hybrid Neuro-Symbolic Trading Engine** combining traditional financial engineering (Factor Models, Black-Litterman optimization) with Agentic AI (LLM Panel with Layered Memory). Inspired by the FinMem paper but implementing a novel portfolio allocation architecture.

**Core Philosophy:**
- **Symbolic (Math):** Factor Models establish a stable baseline (Prior)
- **Neuro (AI):** Panel of LLM Agents with Layered Memory generates forward-looking market views
- **Synthesis:** Black-Litterman Model fuses both, allowing AI to "tilt" the portfolio only when consensus is high

## Architecture Components

### 1. Data Foundation (Point-in-Time ETL)

**Input Sources:**
- Unstructured: Brokerage Research Reports (Daily, Weekly, Monthly PDFs)
- Structured: Market Prices (KOSPI200, S&P500), Factor Returns (Value, Momentum, Quality)

**Pipeline:**
- Temporal Tagging: Publication Date stamping to prevent future leak in backtests
- Semantic Chunking: Split by topic coherence, not arbitrary character counts
- Layer Assignment: Daily Reports → Shallow Layer (High Decay), Monthly/Strategy Reports → Deep Layer (Low Decay)
- Dual Storage: SQL Lake (audit trails) + Vector Store (RAG with Layer + Date metadata)

### 2. Quantitative Anchor (The "Prior")

- Factor decomposition: Regression of Asset Returns against Market Factors (Value, Momentum, Quality, etc.)
- Calculates Implied Equilibrium Returns (π)
- Acts as system's "Gravity" - defaults to risk-based allocation when AI Agents are unsure or disagree

### 3. AI Core: LangGraph Agent Panel

**Step A: Weighted Retrieval (3-Factor Score)**
1. Semantic Match: Relevance to current market regime
2. Recency Decay: News freshness (heavily weighted for Daily data)
3. Importance Boost: Fundamental strategy reports get permanent boost (Deep Layer)

**Step B: Panel Vote**
- N Independent Agents (e.g., 10 distinct instances)
- 5-point voting scale: Strong Buy (+2) | Buy (+1) | Hold (0) | Sell (-1) | Strong Sell (-2)
- Diversity via slight Temperature variations to mimic diverse investment committee

**Step C: Statistical Aggregation**
- View Magnitude (Q): Derived from Mean of votes (Bullish consensus → Positive View)
- Uncertainty (Ω): Derived from Variance of votes
  - High Variance (Disagreement) → System ignores AI, sticks to Quant Anchor
  - Low Variance (Consensus) → System allows AI to aggressively tilt portfolio

### 4. Black-Litterman Optimization Engine

**Inputs:**
1. Quant Baseline (π)
2. AI Consensus View (Q)
3. AI Uncertainty Matrix (Ω)

**Process:** Black-Litterman formula calculates Posterior Distribution
**Output:** Final Portfolio Weights (e.g., "Overweight KOSPI 200 by 5%, Underweight Cash")

## Key Design Decisions

### Why Black-Litterman?
1. **Stability:** Cannot "hallucinate" extreme allocations - Quant Anchor constrains it
2. **Explainability:** Every trade traceable to retrieved documents and panel votes
3. **Ablation-Ready:** Can disable Panel (set Q=0) or Layered Memory (set retrieval weights=0) to prove component value

### Why Layered Memory (FinMem-inspired)?
- Mimics human trader cognition
- Prevents stale information from dominating decisions
- Allows different decay rates for different information types
- Memory scoring: Compound score = recency_score + (importance_score / 100)

### Why Agent Panel vs Single Agent?
- Reduces variance in LLM outputs
- Enables uncertainty quantification via vote dispersion
- More robust to individual model failures

## Coding Conventions

- Use type hints for all function signatures
- Use Pydantic for data validation and settings
- Google-style docstrings for public APIs
- Keep functions focused and under 50 lines when possible

### Memory System
- All memory records must include: content, embedding, timestamp, layer_tag, importance_score, recency_score
- Layer-specific decay factors must be configurable

### Agent Panel
- Each agent must be stateless between voting rounds
- Vote aggregation must handle edge cases (all abstain, perfect split)

### Black-Litterman
- All matrices must be validated for positive semi-definiteness
- Uncertainty (Ω) must be scaled appropriately to confidence levels

## Reference

- FinMem Paper: arXiv:2311.13743
- Reference implementation: `FinMem-LLM-StockTrading-main/`
