# Factor-FinMem: Autonomous Factor Allocation System

## Project Overview

**Factor-FinMem** is an LLM-driven autonomous learning system for factor-based asset allocation. It interprets unstructured brokerage reports to predict the returns of **13 JKP factors**. The system aims to beat **Equal Weight (1/N)** or **Risk-Parity** benchmarks by fusing LLM-derived beliefs with a **Black-Litterman** optimization framework.

### Core Philosophy

- **Neuro-Symbolic Synthesis:** Merges LLM-based qualitative committee voting (Neuro) with Black-Litterman portfolio optimization (Symbolic).
- **Context Engineering:** Prioritizes temporal integrity and information relevance. This includes preventing **forward-looking bias** (e.g., using D+1 reports to avoid future peeking) and using **Memory-based RAG** to surface historically successful investment logic.
- **Autonomous Evolution:** Implements a dynamic RAG system where memory nodes are promoted, demoted, or generated via reflection based on prediction performance.

## Cognitive Architecture

### 1. Factor Committee System (Context-Engineered)

- **13 Factor Specialist Panels:** One panel for each JKP factor (Value, Momentum, Quality, etc.).
- **Agent Committee (n=10):** Each factor is governed by 10 independent agents.
- **Selective Context:** Agents receive only context relevant to their persona and the specific point-in-time (T), strictly excluding future data (T+1).
- **Belief Synthesis:**
  - **View Magnitude ($Q$):** Mean of committee votes (Buy: +1, Hold: 0, Sell: -1).
  - **Uncertainty ($\Omega$):** Variance of committee votes.
- **Optimization:** Fuses $Q$ and $\Omega$ with the **Market Prior** (EW or RP) to determine final factor weights.

### 2. Dynamic Memory System (FinMem-inspired)

- **Layered Structure:** `WORKING` (Daily), `SHALLOW` (Weekly), `DEEP` (Strategy/Reflections).
- **Decay & Purge:** Memories decay over time; deeper layers decay slower. Nodes with `Importance < 5` or `Recency < 0.05` are purged.
- **Promotion (Upgrade):** Memories that frequently contribute to correct predictions (`Reliability` boost) or have high `Access Count` are moved from `SHALLOW` to `DEEP`. Promotion resets `Recency` to 1.0.
- **Demotion (Expulsion):** Memories contributing to errors receive an `Importance` penalty, leading to faster purging.
- **Reflection Injection:** Weekly "Extended Reflections" (error analysis) are injected directly into the `DEEP` layer.

## Development Principles

### Technical Preferences

- **Deep Learning:** Use **PyTorch** for any neural components; avoid TensorFlow [[memory:8479653]].
- **Encoding:** Ensure all CSV exports use **UTF-8 (no BOM)** [[memory:7770710]].
- **Data Integrity:** Raise explicit errors if real data loading fails; never fall back to synthetic data silently [[memory:7402606]].
- **Imports:** Use **absolute package imports** (e.g., `from factor_finmem.module import ...`); do not modify `sys.path` [[memory:6157731]].

### Communication & Style

- **Tone:** Academic, dry, fact-based, and analytical [[memory:8392127]].
- **Structure:** Prioritize critical evaluation and evidence-driven reasoning before reaching conclusions.
- **Terminology:** Adhere to industry-standard quantitative finance (Black-Litterman, JKP Factors, Risk-Parity).

## Key Workflows

- **Daily Inference:** Load Daily Reports -> 13 Factor Committees Vote -> Calculate $Q, \Omega$ -> Black-Litterman Rebalance.
- **Weekly Learning:** Evaluate Accuracy -> Reliability Tuning -> Memory Promotion/Demotion -> Generate Extended Reflections (Direct to DEEP).
