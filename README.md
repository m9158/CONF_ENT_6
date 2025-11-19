# 연구 계획: 홀드백 최적화 및 정책 균형 시뮬레이션

이 문서는 Sharma et al. 및 'OTT Suitability Index' 연구를 바탕으로 한국 영화 시장에 적합한 동적 홀드백 최적화 및 산업 생태계 균형 정책을 도출하기 위한 연구 계획입니다.

## 🔑 주요 기호 및 용어 정의 (Key Notations & Definitions)

본 연구에서 사용되는 핵심 변수와 기호들의 정의입니다.

- **$\alpha$ (Alpha, 성공 잠재력):** 개봉 전 예측된 영화의 흥행 잠재력입니다. 본 연구에서는 '개봉 전 예측 가능한 총매출액(극장+OTT) 등급'으로 정의하여 0과 1 사이의 값으로 정규화합니다. (1에 가까울수록 잠재력 높음)
- **$\hat{\alpha}$ (Alpha-hat, 예측된 성공 잠재력):** 모델 1을 통해 실제로 예측된 $\alpha$ 값입니다. (Part 3의 시뮬레이션에서 영화 그룹핑 용도로 사용)
- **$t$ (Time, 홀드백 기간):** 극장 개봉일로부터 OTT 공개일까지의 기간(일수)입니다.
- **$T$ (Total Lifecycle, 전체 생애 주기):** 영화가 시장에서 유의미한 수익을 창출하는 전체 기간입니다. (예: 365일)
- **$R_b(t)$ (Revenue before OTT, OTT 출시 전 극장 수익):** OTT에 공개되지 않았을 때 특정 시점 $t$에 발생하는 일일 극장 예상 수익입니다.
- **$R_a(t)$ (Revenue after OTT, OTT 출시 후 극장 수익):** OTT 공개 이후 특정 시점 $t$에 발생하는 일일 극장 예상 수익입니다. 통상적으로 $R_a(t) < R_b(t)$ (Cannibalization 발생)입니다.
- **$\tau$ (Tau, 디지털 권리료):** 영화 배급사(MD)가 영화의 OTT 방영권을 넘기는 대가로 OTT 플랫폼으로부터 받는 금액(판권료)입니다.
- **TFS (Theatrical-First Score / 극장 적합도):** 영화의 '이벤트성'을 나타내는 지표. (External/Internal Spectacle 기반)
- **ONS (OTT-Native Score / OTT 적합도):** 영화의 'OTT 고유 몰입도'를 나타내는 지표. (Narrative Complexity/Thematic Intimacy 기반)
- **$W_{\text{Industry}}$ (Welfare of Industry, 산업 전체 효용):** 단순 합산이 아닌, 생태계 지속가능성을 위한 제약 조건이 반영된 산업 전체의 조정된 효용입니다.

## ⚙️ 예측 엔진 구축 (선행 단계)

연구의 핵심이 되는 예측 엔진은 영화의 성공 잠재력($\alpha$), 적합도 지표(TFS, ONS), 동적 수익 곡선($R_b, R_a$), 그리고 디지털 권리료($\tau$)를 추정합니다.

### 1. 모델 1: 성공 잠재력 ($\alpha$) 예측 (정적 모델)

- **목표:** 개봉 전 정보만으로 영화의 흥행 등급(Success Factor, $\alpha$)을 예측합니다.
- **입력 데이터 (독립변수):**
    - **인적 요소:** 감독, 주연 배우, 배급사의 과거 흥행 실적 (평균 관객 수, 수상 내역 등).
    - **콘텐츠 요소:** 장르, 원작 IP 유무, 시리즈 여부, 관람 등급, 제작 예산, 러닝타임.
    - **시장 요소:** 개봉 시기(성수기/비수기), 경쟁작 수.
    - **하입(Hype) 지표:** 개봉 전 2주간 소셜 미디어 언급량, 사전 예매율.
- **출력 데이터 (종속변수):** 최종 총매출액 등급 (0~1 사이의 $\alpha$ 값으로 정규화).
- **산출물:** DB 내 모든 영화의 예측된 성공 잠재력($\hat{\alpha}$) 값.
- **추천 방법론:** XGBoost 또는 Random Forest.
- **주요 한계 및 용도:** 영화 흥행은 개봉 후 변수가 커 $R^2$ 값이 낮을 수 있습니다. (예: 0.5 미만) 따라서 이 모델의 $\hat{\alpha}$ 값은 모델 2와 분리하며, 주로 Part 3 시뮬레이션에서 'High/Low $\alpha$' 그룹을 나누는 용도로만 한정합니다. "이 모델은 개봉 전 정보만으로 63%의 설명력을 가지며, 나머지 37%는 개봉 후 관객 반응(WOM) 등에 의해 결정됨을 시사한다"는 점을 연구의 한계점이나 함의로 명시

### 2. 모델 2: 동적 수익 곡선 ($R_b$, $R_a$) 예측 (Hybrid Pipeline)

- **목표:** 개봉 후 일일 데이터를 기반으로 'OTT 미출시 시 예상 수익 곡선($R_b$)'과 'OTT 출시 시 수익 곡선($R_a$)'을 예측합니다. 이 모델은 모델 1의 예측 오류($R^2<0.5$)와 데이터 누락 문제(KOBIS), 그리고 한국 시장의 독점적 홀드백 구조(Missing Data)를 모두 고려하여 '데이터 기반 학습($R_b$)'과 '이론 기반 시뮬레이션($R_a$)'을 결합합니다.

#### 2.1. Feature Engineering (입력 변수 생성)

LSTM 모델 2-A($R_b$ 예측)에 입력할 시퀀스 데이터를 생성합니다.

- **시간 변수:** day_number (개봉 D+n일), is_weekend (주말/공휴일 여부)
- **KOBIS 성과 변수 (Top 10 기준):**
    - `screen_cnt` (스크린 수)
    - `aud_per_show` (회당 관객 수) [좌석 점유율 대리 변수]
        - **계산식:** `daily_audi_cnt / show_cnt`
- **경쟁 강도 변수 (HHI Screen Share, $CI'$):** [$\hat{\alpha}$ 대리 변수]
    - KOBIS Top 10의 `screen_cnt` 데이터를 활용한 HHI 지표. 모델 1의 $\hat{\alpha}$를 사용하지 않아 무결성을 확보합니다.
    - **계산식 (날짜 d, 타겟 영화 i 기준):**
        - $Screen_{Total10}(d) = \sum_{k \in K_{10}} screen\_cnt_k$
        - $Share_j(d) = screen\_cnt_j / Screen_{Total10}(d)$ (경쟁작 $j$)
        - $CI'(d) = \sum_{j \in K_{10}, j \neq i} (Share_j(d))^2$
- **WOM (입소문) 변수:**
    - `social_buzz` (네이버 검색 인덱스 - 관심도)
    - `daily_rating` (일별 평점 - 평가)

#### 2.2. 모델 2-A: $R_b$ 예측 LSTM 모델 (자연 수익 곡선)

- **목표:** OTT의 영향이 배제된 '순수 극장 수익 곡선'의 동적 패턴을 학습합니다.
- **데이터셋:** 'All Theatrical Data'
    - 한국 시장은 OTT 출시 전 극장 상영이 종료되므로, KOBIS에 존재하는 모든 데이터는 $R_b$ 데이터입니다. 따라서 모든 영화의 전체 상영 기간 데이터를 학습에 사용합니다.
- **데이터 변환 (시퀀스 생성):**
    - Time Step (시퀀스 길이) $N=7$일로 가정.
    - **X_train ($R_b$):** (샘플 수, $N=7$, 피처 개수) 형태의 3D 배열로 변환.
    - **Y_train ($R_b$):** (샘플 수, 1) - 8일째의 `daily_sales_amt`
- **모델:** LSTM (Stacked LSTM 권장). "과거 7일간의 스크린 수, 경쟁 강도($CI'$), 평점 등의 패턴이 8일째의 매출에 미치는 시차 효과(Lagged Effect)를 학습"합니다.
- **모델 아키텍처 (Stacked LSTM):**
    - Input Layer (shape=(7, num_features))
    - LSTM(64, return_sequences=True) (시퀀스 정보를 다음 레이어로 전달)
    - LSTM(32) (마지막 시점의 정보만 출력)
    - Dense(1) (최종 매출액 예측)
- **학습:** `model_Rb.fit(X_train_Rb, Y_train_Rb)`
- **산출물:** `model_Rb` (OTT 개입이 없을 시, 일일 매출액을 예측하는 LSTM 모델)

#### 2.3. 모델 2-B: $R_a$ 생성 시뮬레이션 (Rule-Based Cannibalization)

- **문제점:** 한국 시장의 독점적 홀드백 관행으로 인해, `ott_release_dt` 이후에는 극장 데이터가 존재하지 않아(Missing) 잠식률 학습이 불가능합니다.
- **해결책 (Theoretical Simulation):** 데이터 학습 대신 **'이론 기반 시뮬레이션 로직'**으로 $R_a$를 생성합니다.
- **시뮬레이션 로직 ($R_a$ 생성):**
    - Part 3 시뮬레이션 단계에서 특정 홀드백($t_{sim}$) 시나리오가 주어지면, 다음 수식을 통해 $R_a$를 생성합니다.
    - **잠식 계수($C_i$) 산출:** 영화 $i$의 장르적 특성(TFS, ONS)과 소비자 성향($\gamma$)을 반영합니다.
        $$C_i(\text{TFS}, \text{ONS}, \gamma) = \text{Base Rate} \times (1 + \text{ONS}_i) \times (1 - \text{TFS}_i) \times \gamma$$
    - **Base Rate:** 기본 잠식률 (예: 0.3)
        - **정의:** "영화의 장르적 특성(TFS, ONS)이나 소비자 성향($\gamma$)과 무관하게, OTT에 출시된다는 사실만으로 발생하는 기본 극장 수요 감소율."
        - **역할:** 시뮬레이션 수식에서 기준점 역할을 합니다.
        - **범위:** 0과 1 사이의 값 (보통 0.1 ~ 0.5 사이).
    - **BaseRate 설정 방법론 (3가지 접근법):**
        - **A. 문헌 기반 설정 (Literature-Based Approach) - 가장 추천**
            - 선행 연구나 업계 보고서에서 제시된 수치를 인용하여 설정합니다.
            - **해외 사례 참고:** "미국 시장 연구(XXX et al., 20XX)에 따르면, OTT 동시 개봉 시 극장 관객이 평균 20% 감소했다." $\rightarrow$ BaseRate = 0.2
            - **설문조사 데이터 활용:** "소비자 조사 결과, 극장 상영작이 OTT에 바로 나오면 극장에 안 가겠다는 응답이 30%였다." $\rightarrow$ BaseRate = 0.3
        - **B. 시나리오 기반 민감도 분석 (Sensitivity Analysis Approach) - 차선**
            - **비관적 시나리오 (Conservative):** BaseRate = 0.1
            - **중립적 시나리오 (Neutral):** BaseRate = 0.3
            - **충격적 시나리오 (Aggressive):** BaseRate = 0.5
    - **ONS 가중:** ONS가 높을수록(드라마 등) 잠식 심화.
    - **TFS 방어:** TFS가 높을수록(아이맥스 등) 잠식 방어.
    - **$R_a$ 적용:** $t \ge t_{sim}$일 때, $R_a(t) = R_b(t) \times (1 - C_i)$

- **[시뮬레이션 로직]**
    - **Input:** 특정 영화 $i$, 가상의 홀드백 기간 $t_{sim}$ (예: 30일)
    - **Step 1 ($R_b$ 예측):** 모델 2를 돌려 개봉 1일~60일까지의 $R_b$ 곡선을 예측합니다.
    - **Step 2 (잠식률 $C$ 계산):** 영화 $i$의 특성(TFS, ONS)을 기반으로 잠식 계수를 산출합니다.
        - **수식 예:** $C_i = 0.5 + (0.2 \times \text{ONS}_i) - (0.2 \times \text{TFS}_i)$ (범위는 0~1 사이로 조정)
    - **Step 3 ($R_a$ 적용):** $t_{sim}$ (30일) 이후의 날짜부터는 수익을 강제로 깎습니다.
        - $D < 30$: $Revenue = R_b(D)$
        - $D \ge 30$: $Revenue = R_b(D) \times (1 - C_i)$

#### 2.4. 최종 모델 2 파이프라인 (Part 3 시뮬레이션에서 사용)

- **$R_b(t)$ 예측:** `Predicted_Rb(t) = model_Rb.predict(...)` (LSTM 호출)
- **$R_a(t)$ 예측 ($t \ge t_{sim}$ 시점부터):**
    - $C_i$ 계산 (Rule-Based)
    - `Predicted_Ra(t) = Predicted_Rb(t) * (1 - C_i)`

### 3. 모델 3: 디지털 권리료 ($\tau$) 추정 (Heuristic Model)

- **목표:** 영화의 체급(Scale), 홀드백 기간($t$), 적합도 지표($TFS, ONS$)에 따른 디지털 권리료 $\tau$를 추정합니다.
- **문제점 해결:** 정확한 판권료 데이터 확보의 어려움을 고려하여, 업계 통용 기준인 **'제작비 비례 방식'**을 채택하여 현실성을 확보합니다.
- **필요 데이터:**
    - `total_production_cost` (총제작비, P&A 포함)
    - `base_rate` (제작비 대비 판권료 비율, 예: 10%)
- **추천 방법론 (수식 모델링):**
    $$\tau(t, \text{TFS}, \text{ONS}) \approx \underbrace{(\text{Total Cost} \times R)}_{\text{Base Price Est.}} \times (1 + \text{ONS}) \times \frac{1}{(1 + d(\text{TFS}) \cdot t)}$$
- **Base Price Estimation:**
    - 복잡한 역산 모델 대신, **총제작비의 $R\%$(예: 10~15%)**를 기본 판권료로 가정합니다.
    - $R$ 값은 시뮬레이션 파라미터로 설정하여 민감도 분석(Sensitivity Analysis)을 수행할 수 있습니다.
- **ONS (OTT 적합도):** OTT 플랫폼이 선호하는 장르(드라마, 범죄 등)일수록 기본 권리료에 프리미엄(가중치)이 붙음.
- **d(TFS) (감가상각률):** $TFS$가 높은 '이벤트성' 영화일수록 극장 개봉 시점의 화제성이 중요하므로, $t$가 길어질수록 가치 하락폭($d$)이 큼.

---
## [Part 1] 미시적 접근: 개별 주체 수익 최적화 (Updated)

- **질문:** "개별 배급사(MD) 입장에서, 이 영화의 수익($\Pi_M$)을 극대화하는 최적의 홀드백($t^*$)은?"
- **입력:**
    - 예측된 자연 수익 곡선 $R_b(t)$ (Model 2-A: LSTM)
    - 시뮬레이션된 잠식 계수 $C_i$ (Rule-Based: TFS/ONS 기반)
    - 추정된 디지털 권리료 곡선 $\tau(t, \text{TFS}, \text{ONS})$ (Model 3: Heuristic)
- **방법론 (시뮬레이션 최적화):**
    - **$R_b(t)$ 생성:** Model 2-A를 실행하여 $t=1 \dots 180$일 전체 구간의 예상 극장 매출 곡선을 생성합니다.
    - **$C_i$ 계산:** 영화의 장르적 특성(TFS, ONS)에 따른 고유 잠식률을 산출합니다.
    - **수익 시뮬레이션:** 가상의 홀드백 $t_{sim}$을 1일부터 180일까지 1일 단위로 변경하며 총수익을 계산합니다.
        - **극장 수익:** $\sum_{z=1}^{t_{sim}} R_b(z) + \sum_{z=t_{sim}+1}^{T} [R_b(z) \times (1 - C_i)]$
        - **OTT 수익:** $\tau(t_{sim}, \text{TFS}, \text{ONS})$
        - **총수익($\Pi_M$):** 극장 수익 + OTT 수익
    - **최적점 탐색:** $\Pi_M(t)$가 최대가 되는 $t^*$를 도출합니다.
- **산출물:**
    - 개별 영화별 최적 홀드백 기간($t^*$) 및 예상 최대 수익.
    - 영화 유형(High-TFS vs High-ONS)에 따른 $t^*$ 분포 비교 리포트.

---
## [Part 2] 거시적 전환: 미시적 접근의 한계와 충돌 (Revised)

스토리텔링: "개별 최적화(v1) 결과가 산업 생태계의 다른 참여자(특히 '독립 제작사'와 '국내 OTT')에게는 '생존 위협'이 될 수 있음을 TFS/ONS 매트릭스로 입체적으로 증명한다."

### 1. 분석 목표

v1(배급사 수익 최적화)에서 도출된 최적 홀드백($t^*$)이 영화의 성격(TFS, ONS)에 따라 어떻게 극단적으로 나뉘는지 확인하고, 이로 인해 발생하는 '구조적 갈등'을 시각화합니다.

### 2. 갈등 시나리오 (Conflict Simulation based on TFS/ONS)

- **Case A: 'Theatrical Event' 영화 (High TFS / Low ONS)**
    - **예시:** 블록버스터 액션, 아이맥스용 SF 영화 (예: 탑건, 아바타)
    - **v1 결과 (배급사 최적):**
        - 극장 수익($R_b$) 의존도가 절대적이므로, 배급사는 $R_a$(잠식)를 막기 위해 *매우 긴 홀드백($t^* \ge 90 \sim 120$일)**을 선택합니다.
    - **충돌 (국내 OTT의 위기):**
        - 국내 OTT($U_{\text{Local\_OTT}}$)는 신작 수급 경쟁력이 핵심입니다.
        - 100일 넘게 기다려야 하는 영화는 이미 화제성이 식어 매력도($ONS$)가 떨어지며, 넷플릭스 등 글로벌 OTT 대비 라이브러리 경쟁력을 상실하게 됩니다.

- **Case B: 'OTT Native' 영화 (Low TFS / High ONS)**
    - **예시:** 중저예산 로맨스, 드라마, 독립 예술 영화
    - **v1 결과 (배급사 최적):**
        - 극장 관객 동원력($R_b$)은 낮지만 OTT 선호도($ONS$)는 높습니다.
        - 배급사는 극장 개봉 효과(마케팅)만 누리고, 빠르게 OTT로 넘겨 높은 권리료($\tau$)를 챙기는 *매우 짧은 홀드백($t^* \le 30$일)**을 선택합니다.
    - **충돌 (규제의 역설 - 독립 제작사의 위기):**
        - 만약 극장 보호를 위해 정부가 **'일괄 홀드백(예: 90일)'**을 법제화한다면?
        - 독립 제작사는 '골든타임(30일)'을 놓치고 강제로 90일을 기다려야 합니다.
        - 90일 후, 극장에서도 잊혀지고 OTT에서도 식어버린 이 영화의 권리료($\tau$)는 폭락합니다. 이는 제작사의 현금 흐름($U_{\text{Indie}}$)을 끊어 생존을 위협합니다.

### 3. 스토리텔링의 전환 (The Pivot)

"v1의 결과는 '시장 자율(Case B 방치)'은 국내 OTT를 말려 죽이고, '일괄 규제(Case A 강제)'는 독립 영화를 굶겨 죽인다는 딜레마를 보여줍니다."

"따라서 우리의 목표는 단순한 $t^*$ 찾기가 아니라, 영화의 **TFS/ONS 특성에 따라 유연하게 적용되는 '동적 균형 정책(Dynamic Policy)'**을 찾아 $W_{\text{Industry}}$를 극대화하는 것으로 확장되어야 합니다."

---
## [Part 3] 거시적 분석: 산업 생태계 효용 시뮬레이션

**핵심 질문:** "특정 홀드백 정책($t$)은 산업 전체 효용($W_{\text{Industry}}$)과 4대 이해관계자($U_{\text{MD}}, U_{\text{Indie}}, U_{\text{Local\_OTT}}, U_{\text{Consumer}}$)에게 각각 어떤 영향을 미치는가?"

본 분석은 단순한 이해관계자 간의 제로섬 게임이 아닌, **'지속가능한 생태계 균형'**을 찾기 위한 제약 조건 하의 최적화(Constrained Optimization) 모델을 기반으로 합니다.

### 1. 이해관계자별 효용 함수 (Utility Functions)

각 주체의 효용 함수는 예측 엔진(모델 1, 2, 3)과 이론적 가정(Rule-Based Simulation)을 결합하여 정의됩니다.

- **(1) 배급사 및 극장 ($U_{\text{MD/Theater}}$)**
    - **목표:** 금전적 총수익($\Pi_M$)의 극대화.
    - **수식:**
        $$U_{\text{MD}}(t) = \underbrace{\int_{0}^{t} R_b(z) dz}_{\text{독점 극장 수익}} + \underbrace{\int_{t}^{T} [R_b(z) \times (1 - C_i)] dz}_{\text{경쟁 극장 수익}} + \underbrace{\tau(t, \text{TFS}, \text{ONS})}_{\text{OTT 판권료}}$$
    - **핵심 변수:**
        - $R_b(t)$: 자연 감소하는 극장 매출 곡선 (Model 2-A: LSTM 예측값). $C_i$: TFS/ONS 기반 잠식 계수 (Rule-Based). $\tau$: 홀드백 $t$에 따라 감가상각되는 권리료 (Model 3: Heuristic).

- **(2) 독립 제작사 ($U_{\text{Indie}}$)**
    - **목표:** 생존을 위한 '현금 흐름(Cash Flow)' 확보. (극장 흥행보다 빠른 자금 회수가 중요)
    - **수식:**
        $$U_{\text{Indie}}(t) = \sum_{k \in \text{Low-TFS}} \left( \int_{0}^{t} \frac{R_b(z)}{(1+r)^z} dz + \frac{\tau_k(t)}{(1+r)^t} \right)$$
    - **핵심 변수:**
        - $r$ (Discount Rate): 높은 할인율(예: 연 20%)을 적용하여, 긴 홀드백($t$)으로 인한 자금 경색 위험을 페널티로 반영합니다.

- **(3) 국내 OTT ($U_{\text{Local\_OTT}}$)**
    - **목표:** 글로벌 OTT 대비 '상대적 매력도(Relative Attractiveness)' 유지.
    - **수식:**
        $$U_{\text{Local\_OTT}}(t) = \sum_{k \in \text{High-ONS}} \left[ (\text{ONS}_k \times f_{\text{freshness}}(t)) - \text{Cost}(\tau_k) \right]$$
    - **핵심 변수:**
        - $f_{\text{freshness}}(t)$: $t$가 길어질수록 급격히 감소하는 '신선도 함수'.
        - **의미:** ONS가 높은 영화(드라마, 로맨스 등)를 넷플릭스보다 늦게 받거나, 너무 늦게 받으면($t$ 증가) 효용이 급락합니다.

- **(4) 소비자 ($U_{\text{Consumer}}$)**
    - **목표:** 관람 효용(Value) 대비 비용(Cost)과 불편함(Inconvenience)의 최소화.
    - **수식:**
        $$U_{\text{Consumer}}(t) = \text{Value}(t | \gamma) - \text{Price}(t) - \text{Inconv}(t) - P_{\text{Piracy}}(t)$$
    - **핵심 변수:**
        - $P_{\text{Piracy}}(t)$: 홀드백 $t$가 길어질수록, 그리고 소비자 선호도($\gamma$)가 높을수록 증가하는 '불법 복제 위험 비용' (사회적 손실).

### 2. 산업 전체 효용 ($W_{\text{Industry}}$) 모델링

단순 합산($\sum U_i$)은 거대 기업(배급사)의 이익에 편향될 수 있으므로, '생태계 보호'를 위한 제약 조건을 설정합니다.

- **최적화 문제 정의**
    - **목적 함수 (Objective Function):**
        $$\text{Maximize } W = U_{\text{MD/Theater}} + U_{\text{Consumer}}$$
        (산업 전체의 부가가치와 소비자 후생의 총합을 극대화)
    - **제약 조건 (Constraints - Survival & Competitiveness):**
        - 독립 영화 생존 조건: $U_{\text{Indie}} \ge \text{Min_Survival_Threshold}$ (독립 제작사의 현금 흐름이 말라붙지 않아야 함)
        - 국내 OTT 경쟁력 조건: $U_{\text{Local\_OTT}} \ge \text{Competition_Threshold}$ (국내 OTT의 매력도가 넷플릭스와 경쟁 가능한 수준이어야 함)

### 3. 시뮬레이션 시나리오 및 결과

다양한 정책 시나리오를 대입하여 $W_{\text{Industry}}$와 제약 조건 충족 여부를 시뮬레이션합니다.

- **Scenario 1: 일괄 규제 (Regulation)**
    - **정책:** 모든 영화에 $t=90$일 의무 적용.
    - **결과:** $U_{\text{MD}}$: 증가 (High-TFS 영화의 극장 수익 방어).
    - **문제점:** $U_{\text{Indie}}$가 Min_Threshold 미만으로 추락. (독립 영화들이 자금난으로 도산 위기)
    - **판정:** 실패 (Infeasible). 생태계 다양성 파괴.

- **Scenario 2: 완전 자율 (Laissez-faire)**
    - **정책:** 각 배급사가 이익 극대화($t^*$)를 자유롭게 선택.
    - **결과:** $U_{\text{MD}}$: 최대화.
    - **문제점:** $U_{\text{Local\_OTT}}$가 Competition_Threshold 미만으로 하락. (콘텐츠 수급 지연으로 이용자 이탈)
    - **판정:** 실패 (Infeasible). 플랫폼 경쟁력 상실.

- **Scenario 3: 동적 차등 적용 (Dynamic Policy) - [제안]**
    - **정책:** 영화의 TFS/ONS 특성에 따라 홀드백을 차등 적용.
        - High TFS (이벤트형): 90일 권고 (극장 보호).
        - High ONS (OTT형): 30일 허용 (자율/빠른 유통).
    - **결과:** 
        - $U_{\text{MD}}$: 최적값보다는 다소 낮으나 안정적 수익 확보.
        - $U_{\text{Indie}}$: 빠른 자금 회수로 생존 조건 충족.
        - $U_{\text{Local\_OTT}}$: OTT 적합 콘텐츠의 빠른 수급으로 경쟁력 유지.
    - **판정:** 성공 (Optimal Feasible Solution). $W_{\text{Industry}}$가 제약 조건을 모두 만족하면서 가장 높은 수준에 도달함.

### 4. 결론 및 제언: 동적 홀드백 가이드라인

- **(1) 최종 제안: 동적 홀드백 매트릭스 (Dynamic Holdback Matrix)**

| 구분 | Low TFS (비이벤트형) | High TFS (이벤트형) |
| :--- | :--- | :--- |
| **High ONS (OTT 친화형)** | **① Fast-Track (자율/단기)**<br>- 예: 로맨스, 독립영화<br>- 제안: 홀드백 최소화 (15~30일)<br>- 이유: 빠른 OTT행이 생태계 최적 | **② Hybrid Strategy (중기)**<br>- 예: SF 드라마, 판타지<br>- 제안: 유연한 홀드백 (45~60일)<br>- 이유: 극장과 OTT 수요 균형 필요 |
| **Low ONS (OTT 비친화형)** | **③ Niche Market (기타)**<br>- 예: 다큐, 실험 영화<br>- 제안: 자율 (시장 원리)<br>- 이유: 규제 실효성 낮음 | **④ Theater-Exclusive (장기)**<br>- 예: 액션 블록버스터<br>- 제안: 홀드백 보호 (90일 이상)<br>- 이유: 극장 수익 보호가 최우선 |

- **(2) 정책적 함의 (Policy Implications)**
    - **"One-Size-Fits-All 규제의 위험성":** 획일적 규제는 약자(독립 제작사)를 보호하는 것이 아니라 오히려 죽일 수 있음을 데이터로 입증했습니다.
    - **"장르별 핀셋 지원의 필요성":** 영화의 특성(TFS/ONS)을 고려한 유연한 정책이 산업 전체의 효용을 극대화합니다.
