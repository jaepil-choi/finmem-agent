# Factor-FinMem Architecture Design

## 1. Overview
**Factor-FinMem** is an autonomous factor allocation system that leverages LLM-based macro analysis and self-evolving memory. This document defines the system structure, data flow, and agentic workflows using a modular and configuration-driven approach.

## 2. 3-Tier Data Architecture
To ensure data integrity, search efficiency, and analytical performance, the system utilizes three distinct storage layers.

| Tier | Name | Technology | Purpose |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Raw Data Lake** | MongoDB | Permanent storage of raw PDF texts and metadata. |
| **Tier 2** | **Memory Warehouse** | FAISS | Hierarchical vector storage (Shallow, Intermediate, Deep) for RAG. |
| **Tier 3** | **Structured Mart** | Parquet Files | Time-series factor returns (JKP) and agent performance logs. |

---

## 3. Component Design

### 3.1. Modular RAG (Strategy Pattern)
The retrieval logic is decoupled from the agents. A global RAG strategy is injected into all agents during execution to ensure consistency and facilitate comparison between different RAG methods.
- **Naive RAG**: Standard similarity-based retrieval.
- **Agentic RAG (CRAG/Self-RAG)**: LLM-based relevance grading and query rewriting.
- **FinMem RAG**: Score-based retrieval using Recency and Importance.

### 3.2. Factor Theme Committee (Ensemble for Uncertainty)
The system manages **13 Factor Theme Committees** (e.g., Value, Quality, Momentum). Each committee consists of multiple **Identical Sub-Agents** to quantify the model's confidence.
- **Identical Experts**: Sub-agents within a committee receive the **exact same prompt and context**.
- **Uncertainty Quantification**: 
    - **View Magnitude ($Q_i$)**: Mean of the individual votes (+1, 0, -1).
    - **Uncertainty ($\Omega_i$)**: Variance of the individual votes. A high variance indicates the LLM is unsure or the context is ambiguous.

### 3.3. Prompt Assembly Engine (Builder Pattern)
System prompts are dynamically assembled from modular components to ensure consistency across the committee.
- **Risk Profile (Dynamic)**: Selected based on portfolio performance (e.g., Risk-Averse vs. Risk-Seeking).
- **Factor Expertise (Domain)**: Specialized domain knowledge for each of the 13 factor themes.

### 3.4. Core Engine
- **Black-Litterman Optimizer**: Combines committee views ($Q, \Omega$) with market priors.
- **Backtest Engine**: Measures performance and generates ground truth for reflection.

---

## 4. Configuration Layer (Modular YAML)
Settings are managed through modular YAML files to ensure scalability.
- `configs/agents.yaml`: Defines the committees and the number of sub-agents per committee.
- `configs/prompts/risk_profiles.yaml`: Defines dynamic risk behavior logic.
- `configs/prompts/factor_expertise.yaml`: Contains detailed domain knowledge for each factor theme.

---

## 5. Agentic Workflow (LangGraph)

```mermaid
flowchart TD
    Start([Start]) --> Init["Init State (Target Date, Macro Question, RAG Strategy)"]
    Init --> Broadcast["Translate & Broadcast to 13 Factor Committees"]

    subgraph FactorCommittees [13 Parallel Factor Committees]
        direction TB
        
        subgraph ValueCommittee [Value Factor Committee]
            direction TB
            subgraph ValueAgents [N Identical Sub-Agents]
                V1[Agent 1]
                V2[Agent 2]
                VN[Agent N...]
            end
            
            Retriever["Retriever (Global Strategy)"]
            PromptBuilder["Prompt Assembly (Risk + Expertise)"]
            
            V1 & V2 & VN --> PromptBuilder
            PromptBuilder --> Retriever
            Retriever --> Vote["Individual Votes (+1, 0, -1)"]
            Vote --> Agg["Calculate Q_value & Omega_value"]
        end

        %% ... other 12 committees ...
    end

    Broadcast --> ValueCommittee
    
    Agg --> GlobalAgg["Combine into Global Q Vector & Omega Matrix"]
    GlobalAgg --> Optimize["Black-Litterman Optimization"]
    Optimize --> Output[Final Portfolio Weights]
    
    Output --> IsTrain{Is Train Mode?}
    IsTrain -->|"Yes"| Reflect["Parallel Reflection & Reliability Update"]
    Reflect --> End([End])
    IsTrain -->|"No"| End
```

---

## 6. Design Patterns
- **Strategy Pattern**: Modular RAG strategies injectable into agents.
- **Factory Pattern**: Dynamic creation of committees and agents from YAML configurations.
- **Builder Pattern**: Dynamic assembly of system prompts from modular fragments.
- **Repository Pattern**: Abstracted data access for MongoDB, Vector DB, and Parquet.

---

## 7. Directory Structure
```text
src/
├── core/                # Pure logic (Optimization, Evaluation, Domain Models)
├── agents/              # Agent layer
│   ├── factory.py       # Factory for creating Committees/Agents
│   ├── prompt_builder.py# Builder for modular prompt assembly
│   ├── committee.py     # Committee aggregation logic (Mean/Var)
│   └── agent.py         # Base Agent class
├── rag/                 # Strategy pattern for retrieval
├── db/                  # Data access layer (Repository pattern)
└── graph/               # LangGraph orchestration
```
