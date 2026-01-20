# **Product Requirement Document (PRD): Factor-FinMem**

## **1. 프로젝트 개요 (Project Overview)**

### **1.1. 프로젝트 명**

**Factor-FinMem**: LLM 기반 매크로 분석 및 자율 학습형 팩터 자산 배분 시스템

### **1.2. 목적 (Objective)**

* **비정형 데이터(Text)의 퀀트화:** 키움증권의 시황/전망 리포트(Macro)를 FinMem 에이전트가 해석하여, JKP(Jensen-Kelly-Pedersen) 13개 팩터의 초과 수익률 방향성을 예측한다.
* **자율 학습형 RAG:** 예측 결과에 대한 피드백(Feedback)을 통해 메모리의 신뢰도(Reliability)를 조정하고, 오답 노트(Extended Reflection)를 생성하여 에이전트의 판단력을 지속적으로 향상시킨다.
* **최적 자산 배분:** Black-Litterman 모델을 통해 에이전트 위원회의 주관적 견해(View)와 시장의 균형(Market Prior - Equal Weight 또는 Risk Parity)을 결합하여 벤치마크 대비 초과 수익을 목표로 하는 최적의 팩터 포트폴리오 비중을 산출한다.

---

## **2. 데이터베이스 아키텍처 (3-Tier Database Architecture)**

데이터의 보존, 검색, 분석 용도에 따라 세 가지 저장소를 분리하여 운영합니다.

### **Tier 1: Raw Data Lake (NoSQL - MongoDB)**

* **목적:** 원본 데이터의 영구 보존 및 무결성 보장. 추후 로직 변경 시 재가공(Re-processing)의 원천이 됨.
* **Schema: `raw_reports`**
* `_id`: ObjectId (Unique)
* `filename`: 파일명 (예: `20220110_kiwoom_daily.pdf`)
* `report_type`: `DAILY` | `WEEKLY` | `MONTHLY`
* `publish_date`: 발행일 (ISO 8601)
* `full_text`: 파싱된 전체 텍스트 (String)
* `metadata`: 페이지 수, 저자, 원본 파일 해시값 등
* `created_at`: 수집 시각

### **Tier 2: Memory Warehouse (Vector DB - Chroma/FAISS)**

* **목적:** 에이전트의 인지(Cognition) 및 검색(Retrieval)을 위한 의미론적 저장소. FinMem의 계층적/동적 메모리 구조를 구현.
* **Schema: `memory_nodes`**
* `id`: UUID
* `source_ref_id`: MongoDB `_id` 참조 (Traceability)
* `summary_content`: LLM이 요약한 핵심 팩터/매크로 정보 (Embedding 대상)
* `embedding`: Vector Array
* `layer_type`: `WORKING` | `SHALLOW` | `DEEP`
* `importance`: 정보 중요도 (0 ~ 100)
* `reliability`: 예측 성공 기여도 (Default 1.0, Feedback 루프에 의해 변동)
* `access_count`: 검색 및 사용 빈도
* `timestamp`: 해당 정보의 기준 시점

### **Tier 3: Structured Mart (RDBMS - PostgreSQL/SQLite)**

* **목적:** 시계열 데이터 관리, 모델 파라미터 저장, 성과 분석을 위한 정형 데이터 저장소.
* **Tables:**
* `factor_returns`: 일자별 13개 팩터 수익률 (Source: JKP Data).
* `agent_profiles`: 13개 에이전트별 페르소나, 핵심 지표, 리스크 성향 설정.
* `agent_predictions`: 에이전트의 일자별 예측(View), 확신도(Confidence), 근거(Rationale).
* `portfolio_history`: Black-Litterman 산출 비중 및 백테스팅 성과.

---

## **3. 데이터 흐름 (Data Flow Pipeline)**

### **Step 1: Ingestion & Storage (PDF -> MongoDB)**

1. **Loader:** 지정된 폴더에서 새로운 PDF 감지.
2. **Parser:** PDF를 텍스트로 변환 (OCR 또는 Text Extraction).
3. **Archiving:** 메타데이터와 함께 **MongoDB**에 적재. (가공 없이 원본 그대로 저장)

### **Step 2: Memory Processing (MongoDB -> Vector DB)**

1. **Fetcher:** MongoDB에서 처리되지 않은 리포트 로드.
2. **Summarizer:** LLM을 이용해 팩터 투자 관점(금리, 인플레, 수급 등)에서 요약.
3. **Layering:** 리포트 타입에 따라 메모리 계층 할당.

* Daily -> Working
* Weekly -> Shallow
* Monthly -> Deep

1. **Embedding:** 요약된 텍스트를 벡터화하여 **Vector DB**에 저장.

### **Step 3: Intelligence & Decision (Vector DB -> SQL DB)**

1. **Trigger:** 13개 팩터 에이전트 위원회(Committee) 활성화.
2. **Context Engineering (Retrieval):** 각 에이전트의 프로파일 및 시점(T)에 맞춰 관련 기억을 선별적으로 검색 (Top-K).
   * **Temporal Constraint:** 추론 시점 기준 미래 데이터 배제 (D+1 리포트 원칙 등).
   * **Relevance Scoring:** *Score = Recency + Relevancy + Importance + Reliability*
3. **Committee Voting:** (팩터당 10개 에이전트)
   * 각 에이전트는 Buy(+1), Hold(0), Sell(-1) 투표 수행.
   * **View Magnitude (Q):** 투표 결과의 평균(Mean).
   * **Uncertainty (Ω):** 투표 결과의 분산(Variance).
4. **Logging:** 예측 결과(Q, Ω, 논리)를 **SQL DB**에 저장.

---

## **4. 에이전트 워크플로우 (Logical Flow)**

### **A. Daily Inference Routine (매일 아침 실행)**

포트폴리오 리밸런싱을 위한 뷰(View) 생성 과정입니다.

1. **Update Working Memory:** 오늘자 Daily 마감 시황을 MongoDB -> Vector DB(Working)로 적재.
2. **Broadcast Context:** 모든 에이전트에게 "오늘의 핵심 키워드(예: 금리 급등)" 전파.
3. **Agent Parallel Processing:** (13개 에이전트 위원회 동시 실행)

* **Retrieve:** 과거 유사 국면(Deep)이나 최근 트렌드(Shallow) 검색.
* **Committee Vote:** 10개 에이전트의 독립적 판단을 통해 $Q$와 $\Omega$ 산출.
* **Output View:** View Magnitude ($Q$), Uncertainty ($\Omega$) 산출.

1. **Optimization (Black-Litterman):**

* Market Prior($\pi$) 계산 (Equal Weight / Risk Parity 2가지 버전).
* Agent Views ($Q, \Omega$) 결합 -> 최종 Posterior 산출.
* 최적 비중 계산 및 SQL 저장.

### **B. Weekly Training Routine (주간 피드백 및 메모리 전이)**

RAG 시스템의 성능을 자가 발전시키는 핵심 과정입니다.

1. **Load Ground Truth:** SQL DB에서 지난주 `agent_predictions`와 실제 `factor_returns` 로드.
2. **Evaluate:** 예측 정확도(Hit Ratio) 평가.
3. **Dynamic Memory Transfer (Promotion/Demotion):**
   * **Promotion (승격):** 예측 성공에 기여한 `SHALLOW` 메모리의 `reliability` 상향, `access_count` 증가. 기준 충족 시 `DEEP` 레이어로 이동 및 `recency` 초기화.
   * **Demotion (강등/소멸):** 예측 실패에 기여한 메모리의 `importance` 삭감. 중요도가 임계치 미만일 경우 영구 삭제(Purge).
4. **Reflection Injection:**
   * **Retrospective Analysis:** LLM이 "왜 틀렸는가?" 회고 생성.
   * **Generate Deep Memory:** 회고 내용을 **Extended Reflection**으로 변환하여 곧바로 `DEEP` 레이어에 저장. (지식의 이동 및 강화)

---

## **5. 핵심 기능 명세 (Functional Specifications)**

### **5.1. Context Engineering (핵심 기술)**

에이전트가 의사결정 시점에 "알아야 할 정보"만 선별적으로 제공하여 성능을 최적화하고 편향을 제거하는 핵심 엔지니어링 기법입니다.

* **Forward-Looking Bias 제거:** 백테스팅 및 실시간 추론 시 미래 정보를 엿보는(Peeking) 것을 방지합니다. 예를 들어, `T` 시점의 예측을 위해 `T`일자 장 마감 후 발행된 시황 리포트(`T+1` 발행물 등)를 엄격히 구분하여 제공합니다.
* **Selective Context Provision:** 각 팩터 에이전트의 페르소나와 현재 상황에 직결된 정보만을 필터링하여 컨텍스트 윈도우를 구성합니다.
* **Memory-based RAG:** 과거 투자 성과에 기여도가 높았던(Reliability가 높은) 정보를 우선적으로 추출하여, 에이전트가 검증된 논리에 집중하도록 유도합니다.

### **5.2. Factor Agent Profiles**

13개 테마별로 특화된 `System Prompt`를 DB(SQL)에서 관리하여 유연성 확보.

* **구성요소:**
* **Identity:** "당신은 Value 팩터 전문가입니다."
* **Philosophy:** "저평가된 주식이 장기적으로 시장을 이깁니다."
* **Focus Data:** "PER, PBR, 금리 스프레드에 민감하게 반응하십시오."
* **Behavior:** "성장주 버블 붕괴 신호(Debt Issuance 등)를 찾으십시오."

### **5.2. Black-Litterman Engine**

* **Market Prior Calculation:**
  * Option A: Equal Weight ($w_{eq} = 1/N$) 기반 내재 수익률.
  * Option B: Risk Parity 기반 내재 수익률.
* **View Integration:**
  * **View Vector ($Q$):** 에이전트 위원회 투표의 산술 평균.
  * **Uncertainty Matrix ($\Omega$):** 에이전트 위원회 투표의 분산(Variance)을 대각 성분으로 매핑.
  * 확신도가 높으면(분산 낮음) Agent View 적극 반영, 확신도가 낮으면(분산 높음) Market Prior 추종.

---

## **6. 개발 로드맵 (Phases)**

* **Phase 1: Data Pipeline 구축 (Current)**
* MongoDB, Vector DB, SQL DB 스키마 설계 및 구축.
* PDF 파서 및 MongoDB 적재 로직 구현.

* **Phase 2: Memory & Agent 구현**
* LLM 요약 및 임베딩 파이프라인(MongoDB -> Vector DB) 구현.
* 13개 팩터 에이전트 프롬프트 엔지니어링 및 위원회 추론 모듈 개발.

* **Phase 3: Quant Engine & Feedback Loop**
* Black-Litterman 최적화 모듈 구현 (PyPortfolioOpt 활용).
* Weekly Training (Reliability 조정 및 메모리 전이) 로직 구현.

* **Phase 4: Integration & Testing**
* 전체 파이프라인 통합 테스트.
* 과거 데이터 기반 시뮬레이션(Backtesting).
