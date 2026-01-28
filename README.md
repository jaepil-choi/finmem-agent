# Factor-FinMem: 자율 학습형 팩터 자산 배분 시스템

**Factor-FinMem**은 비정형 금융 리포트를 LLM 에이전트 위원회를 통해 정량적 팩터 견해($Q$)와 불확실성($\Omega$)으로 변환하는 AI 기반 퀀트 투자 시스템입니다. **FinMem** 아키텍처에 기반한 자율 진화형 메모리 시스템을 활용하며, 에이전트의 견해를 **Black-Litterman** 모델을 통해 시장 균형과 결합하여 최적의 포트폴리오를 구성합니다.

---

## 🌟 프로젝트 목표
LLM 앙상블을 통해 매크로 분석부터 팩터 배분까지의 과정을 자동화하고, 통계적 최적화와 스스로 개선되는 장기 기억(Long-term Memory)을 통해 견고한 의사결정 체계를 구축합니다.

---

## 🏗️ 시스템 아키텍처

### 3계층 데이터 레이어
| 계층 | 명칭 | 기술 | 용도 |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Raw Data Lake** | MongoDB | 원본 PDF 텍스트 및 메타데이터 영구 보존. |
| **Tier 2** | **Memory Warehouse** | FAISS | 계층적(Shallow, Intermediate) 벡터 저장소 및 RAG. |
| **Tier 3** | **Structured Mart** | Parquet | 시계열 팩터 수익률(JKP) 및 성과 기록 관리. |

### 오케스트레이션 (Orchestration)
- **LangGraph**: 에이전트의 추론 상태 머신 제어 (검색 → 분석 → 생성 → 회고).
- **Strategy Pattern**: 교체 가능한 RAG 엔진 (Naive vs. FinMem).

---

## 🧠 핵심 메커니즘

### 1. 에이전트 위원회 (불확실성 정량화)
단일 LLM 대신 **13개의 팩터 테마별 위원회**(예: 가치, 퀄리티, 모멘텀)를 운영합니다.
- **견해 강도 ($Q$)**: 개별 에이전트 투표(+1, 0, -1)의 평균값에 상수(예: 0.05)를 곱하여 기대 수익률 스케일로 변환.
- **불확실성 ($\Omega$)**: 투표 결과의 분산에 상수의 제곱(예: 0.0025)을 곱하여 최적화 시 가중치 조절에 사용.

### 2. FinMem 검색 및 스코어링
정규화된 복합 점수[0, 3]를 기반으로 문서를 재정렬합니다:
$$Score = Similarity + Recency + \frac{Importance}{100}$$
- **Recency**: 문서의 시점에 따른 지수 감쇠(Exponential Decay).
- **Importance**: 회고 레이어를 통해 동적으로 업데이트되는 중요도.

### 3. 자율 진화 (회고 레이어)
학습 모드에서 실제 시장 결과와 예측을 비교하는 사후 검토를 수행합니다:
- **접근 카운터 ($AC$)**: 시장 피드백($sign(Q \times Return)$)에 의한 누적 합계.
- **중요도 갱신**: $I_{new} = I_{old} + (AC_{new} \times 5)$ (0~100 사이 캡핑).
- 이 과정을 통해 정확한 예측에 기여한 지식이 검색 시 우선순위를 갖게 됩니다.

---

## 🚀 시작하기

### 사전 요구사항
- **uv** (패키지 매니저)
- MongoDB 인스턴스
- OpenAI / Anthropic API 키

### 설치 및 실행
```powershell
# 의존성 설치
uv sync

# 자율 진화 학습 루프 실행
uv run python -m scripts.train_reflection_loop --mode train --start 20240225 --end 20240403
```
- **Mode**: `train` (Self-Evolution enabled) or `test` (Evaluation only).
- **Dates**: Default training (2024.02.25~2024.04.03), Default test (2024.04.04~2024.05.15).

---

## 🧠 핵심 메커니즘
