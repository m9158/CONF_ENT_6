# 🎬 Part 3: 산업 생태계 효용 시뮬레이션

**작성일**: 2024-11-21  
**연구 주제**: 한국 영화 시장의 동적 홀드백 최적화 및 산업 생태계 균형 정책

---

## 📋 개요

이 폴더는 **Part 3 (산업 생태계 효용 시뮬레이션)**의 구현을 포함합니다. Model 2 (극장 수익 예측)와 Model 3 (OTT 권리료 추정)을 통합하여 3단계 시뮬레이션을 수행합니다.

---

## 📂 파일 구조

```
SIM/
├── main.ipynb                        # 🔬 메인 시뮬레이션 노트북
├── main_backup.ipynb                 # 백업 파일
├── README.md                         # 📘 본 문서
│
├── [실행 후 생성되는 파일들]
├── part1_individual_optimization.csv    # Part 1 결과
├── part1_individual_optimization.png    # Part 1 그래프
├── part2_movie_type_analysis.csv        # Part 2 결과
├── part2_tfs_ons_matrix.png            # Part 2 그래프
├── part3_policy_comparison.csv          # Part 3 결과
└── part3_policy_comparison.png          # Part 3 그래프
```

---

## 🎯 시뮬레이션 구조

### Part 1: 미시적 접근 - 개별 주체 수익 최적화

**핵심 질문**: "개별 배급사(MD) 입장에서, 이 영화의 수익($\Pi_M$)을 극대화하는 최적의 홀드백($t^*$)은?"

**방법론**:
1. 홀드백 시나리오 설정 (0일 ~ 180일, 10일 간격)
2. 각 시나리오별 총수익 계산 (극장 수익 + OTT 권리료)
3. 최대 수익을 내는 홀드백 도출

**출력**:
- 홀드백별 수익 분석 표
- 수익 구성 요소 그래프
- 최적 홀드백 결정

---

### Part 2: 거시적 전환 - 영화 유형별 충돌 분석

**핵심 질문**: "TFS/ONS 특성에 따라 영화 유형별로 최적 홀드백이 어떻게 다른가? 이로 인한 이해관계 충돌은?"

**분석 방법**:
1. 영화를 TFS/ONS 중앙값 기준으로 4개 그룹 분류
   - **Theater-Exclusive** (High TFS + Low ONS): 블록버스터
   - **OTT-Native** (Low TFS + High ONS): 드라마/독립영화
   - **Hybrid** (High TFS + High ONS): 균형형
   - **Niche** (Low TFS + Low ONS): 틈새 시장
2. 각 그룹별 평균 최적 홀드백 도출
3. TFS/ONS 매트릭스 시각화

**충돌 시나리오**:
- **Case A (Theater-Exclusive)**: 긴 홀드백(90일+) 선호 → 국내 OTT 신작 수급 지연
- **Case B (OTT-Native)**: 짧은 홀드백(30일) 선호 → 일괄 규제 시 독립 제작사 생존 위협

**출력**:
- TFS/ONS 산점도 with 최적 홀드백
- 유형별 박스플롯
- 충돌 시나리오 분석

**Part 2 Extended: 미시적 최적화의 한계 입증**

**핵심 발견**: 배급사 입장에서만 최적화하면 대부분 영화가 **180일**로 수렴합니다.

**충돌 구조 분석**:
각 영화 유형별로 30일 vs 180일 시나리오에서 4대 이해관계자 효용 비교:

```
배급사 최적 (180일) 선택 시:
✅ 배급사: 극장 수익 최대화
❌ 독립 제작사: 시간 할인으로 현금 흐름 압박 (-20%)
❌ OTT: 신선도 붕괴 (e^-3.6 ≈ 3%)
❌ 소비자: 접근성 저하로 불만 폭증 (-100%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
결과: 산업 전체 효용 급락
```

**전환의 필연성**:
"미시적 최적 ≠ 거시적 최적"을 수치로 입증 → Part 3에서 산업 전체 효용 기준 정책 탐색 필요

---

### Part 3: 거시적 분석 - 산업 생태계 효용 시뮬레이션

**핵심 질문**: "특정 홀드백 정책($t$)은 산업 전체 효용($W_{\text{Industry}}$)과 4대 이해관계자에게 각각 어떤 영향을 미치는가?"

**이해관계자 효용 함수 및 설계 근거**:

1. **배급사/극장** ($U_{\text{MD}}$):
   ```python
   U_MD = Σ(극장 수익 + OTT 권리료)
   ```
   - **목표**: 금전적 총수익 극대화
   - **설계 근거**: 
     - 표준 이윤 극대화(Profit Maximization) 모델
     - Cannibalization 효과 반영 (Sharma et al., 2016)

2. **독립 제작사** ($U_{\text{Indie}}$):
   ```python
   U_Indie = Σ[(전체 수익) × 1/(1 + 0.4 × t/365)]  # 시간 할인
   ```
   - **대상**: Low TFS 영화만 (독립 영화 ≈ 예술/드라마)
   - **목표**: 빠른 자금 회수 (생존을 위한 현금 흐름)
   - **설계 근거**: 
     - Time Value of Money (NPV 개념)
     - 높은 할인율 40%: 영세 제작사의 높은 기회비용 반영
     - 다음 작품 제작 자금 확보가 생존과 직결

3. **국내 OTT** ($U_{\text{Local OTT}}$):
   ```python
   U_OTT = Σ[ONS × e^(-0.02×t) × 50억]  # 신선도 × 구독자 가치
   ```
   - **대상**: High ONS 영화만 (OTT 친화적 작품)
   - **목표**: 신작 콘텐츠 확보로 구독자 유입/이탈 방지
   - **설계 근거**: 
     - 신선도 함수: 넷플릭스 데이터 반영 (4주 내 시청 집중)
     - 구독자 가치 50억: 화제작 1편의 신규 가입 + 이탈 방지 효과

4. **소비자** ($U_{\text{Consumer}}$):
   ```python
   U_Consumer = Σ[(90-t)/90 × (1+γ) × 10억]  # 접근성 지수
   ```
   - **목표**: 콘텐츠 접근성 최대화 (기준점 대비 상대적 만족도)
   - **설계 근거**:
     - **Prospect Theory** (Kahneman & Tversky, 1979): 준거점 대비 효용 평가
     - **기준점 90일**: 전통적 극장-DVD 홀드백 기간 (산업 관행)
     - **선형 함수**: 단순성과 해석 가능성 (Occam's Razor)
     - **γ 민감도**: OTT 선호도 높을수록 홀드백 변화에 민감
   - **경제학적 의미**:
     - t < 90: 양의 효용 (기대보다 빨라서 만족)
     - t = 90: 0 (중립, 예상대로)
     - t > 90: 음의 효용 (기대보다 늦어서 불만)

**산업 전체 효용**:
```python
W_Industry = U_MD + U_Indie + U_OTT + U_Consumer
```

**효용 함수 설계 철학**:

본 연구의 효용 함수는 다음 3가지 원칙을 기반으로 설계되었습니다:

1. **균형 잡힌 스케일 (Balanced Scale)**:
   - 각 이해관계자의 효용이 조 단위로 정규화되어 한 주체가 지배적이지 않음
   - 배급사(~3.5조), 독립(~0.3조), OTT(~0.7조), 소비자(±0.5조)
   - 모든 주체의 효용이 최종 결정에 유의미한 영향을 미침

2. **이론적 타당성 (Theoretical Validity)**:
   - 각 함수는 경제학 이론에 기반 (NPV, Freshness Decay, Prospect Theory)
   - 임의 배수가 아닌 산업 데이터와 문헌에서 도출된 파라미터 사용
   - 행동경제학적 원리 반영 (준거점, 손실 회피, 시간 할인)

3. **정책 비교 가능성 (Policy Comparability)**:
   - 상대적 효용 개념으로 서로 다른 정책의 효과 직접 비교 가능
   - 소비자 효용이 양/음/0을 모두 가질 수 있어 정책 민감도 측정
   - 생태계 전체의 지속가능성을 단일 지표(W_Industry)로 평가

**정책 시나리오**:

| 정책 | 설명 | 적용 방법 |
|------|------|-----------|
| **Laissez-faire (자율)** | 각 영화의 최적 홀드백 선택 | Part 2 결과 활용 |
| **Uniform_90 (일괄 규제)** | 모든 영화 90일 고정 | 고정값 적용 |
| **Dynamic (동적 차등)** | TFS/ONS 기반 차등 적용 | High TFS: 90일<br>High ONS: 30일<br>기타: 60일 |

**출력**:
- 정책별 이해관계자 효용 비교 (Stacked Bar)
- 산업 전체 효용 비교 (Bar Chart)
- 최적 정책 도출

---

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 필수 라이브러리 설치
pip install numpy pandas matplotlib seaborn

# 선택 (Model 2 LSTM 사용 시)
pip install tensorflow
```

### 2. 데이터 준비

다음 파일들이 상위 디렉토리에 존재해야 합니다:

```
../model_2/
  ├── movie_meta_with_cannib.csv

../model3/
  ├── movie_digital_fees.csv
  ├── movie_base_digital_fees.csv
  └── platform_params_2024.csv

../CONF_ENT_6/
  └── 기준영화 데이터/
      └── movie_metadata.csv
```

### 3. 노트북 실행

```bash
jupyter notebook main.ipynb
```

또는 Jupyter Lab:

```bash
jupyter lab main.ipynb
```

### 4. 실행 순서

1. **STEP 0**: 환경 설정 및 데이터 로드 (필수)
2. **STEP 1**: 핵심 함수 정의 (필수)
3. **Part 1**: 개별 영화 최적화 (10-15분)
4. **Part 2**: 유형별 분석 (15-20분)
5. **Part 3**: 정책 시뮬레이션 (20-30분)
6. **최종**: 요약 및 결과 저장

**전체 실행 시간**: 약 45-65분 (영화 데이터 크기에 따라 변동)

---

## 📊 출력 결과

### 1. CSV 파일

#### `part1_individual_optimization.csv`
```
holdback | theater_revenue | digital_fee | total_profit | ...
0        | 450.5          | 18.75       | 469.3        | ...
10       | 448.2          | 17.50       | 465.7        | ...
...
```

#### `part2_movie_type_analysis.csv`
```
movie_id | movie_name | movie_type         | TFS | ONS | optimal_holdback
20124079 | 범죄도시2  | Theater-Exclusive  | 8.2 | 4.1 | 90
...
```

#### `part3_policy_comparison.csv`
```
policy              | U_MD_조 | U_Indie_조 | U_OTT_조 | U_Consumer_조 | W_Industry_조
Laissez-faire (자율) | 12.45   | 3.21       | 2.56     | -1.23        | 16.99
Uniform_90 (일괄)    | 11.89   | 2.87       | 2.12     | -1.45        | 15.43
Dynamic (동적)       | 12.28   | 3.45       | 2.78     | -1.18        | 17.33
```

### 2. 이미지 파일

- **`part1_individual_optimization.png`**: 개별 영화의 홀드백별 수익 분석 (2개 서브플롯)
- **`part2_tfs_ons_matrix.png`**: TFS/ONS 매트릭스 및 유형별 박스플롯 (2개 서브플롯)
- **`part3_policy_comparison.png`**: 정책별 효용 비교 (Stacked Bar + Bar Chart)

---

## 🔑 핵심 함수

### `calculate_theater_revenue(movie_id, holdback_days, horizon=180)`

극장 수익 계산 (Rb + Ra)

**Parameters**:
- `movie_id` (str): 영화 코드
- `holdback_days` (int): 홀드백 기간
- `horizon` (int): 예측 기간 (기본 180일)

**Returns**:
```python
{
    'total_revenue': float,
    'rb_revenue': float,
    'ra_revenue': float,
    'movie_info': Series,
    'budget': float
}
```

---

### `get_digital_fee(movie_id, platform_name='Netflix', holdback_days=30)`

디지털 권리료 조회 (Model 3 CSV 기반)

**Parameters**:
- `movie_id` (str): 영화 코드
- `platform_name` (str): 플랫폼 이름
- `holdback_days` (int): 홀드백 기간

**Returns**:
- `float`: 권리료 (원)

---

### `calculate_total_profit(movie_id, holdback_days, platform_name='Netflix', horizon=180)`

총수익 계산 (극장 + OTT)

**Returns**:
```python
{
    'holdback_days': int,
    'platform': str,
    'total_profit': float,
    'theater_revenue': float,
    'digital_fee': float,
    'movie_info': Series,
    'budget': float
}
```

---

## 📈 결과 해석

### Part 1 결과 예시

```
🎯 최적 홀드백: 30일

💰 수익 구조:
   - 극장 수익:    450.5억 원
   - OTT 권리료:    18.75억 원
   - 총 수익:      469.3억 원
```

**해석**: 
- 개별 배급사 입장에서는 30일 홀드백이 최적
- 극장 수익이 전체의 96%를 차지
- OTT 권리료는 상대적으로 작지만 무시할 수 없는 수준

---

### Part 2 결과 예시

```
Theater-Exclusive 영화: 평균 90일
OTT-Native 영화: 평균 30일
```

**충돌**:
1. **Theater-Exclusive**: 긴 홀드백 → 국내 OTT 경쟁력 약화
2. **OTT-Native**: 짧은 홀드백 → 일괄 규제 시 독립 제작사 생존 위협

**인사이트**: 획일적 규제는 생태계 파괴 → 동적 차등 정책 필요

---

### Part 3 결과 예시

```
🏆 최적 정책: Dynamic (동적 차등)

💰 산업 전체 효용: 17.33조 원

이해관계자별:
   - 배급사/극장:    12.28조 원
   - 독립 제작사:     3.45조 원
   - 국내 OTT:        2.78조 원
   - 소비자:         -1.18조 원
```

**정책 비교**:
- **Laissez-faire**: 배급사 효용 최대 but 독립/OTT 불균형
- **Uniform_90**: 극장 보호 but 독립 제작사 현금 흐름 악화
- **Dynamic**: 균형잡힌 접근으로 산업 전체 효용 최대

---

## 💡 정책 제언

### 동적 홀드백 매트릭스

| 구분 | Low TFS (비이벤트형) | High TFS (이벤트형) |
|------|---------------------|---------------------|
| **High ONS (OTT 친화형)** | **① Fast-Track (자율/단기)**<br>예: 로맨스, 독립영화<br>제안: 홀드백 최소화 (15~30일)<br>이유: 빠른 OTT행이 생태계 최적 | **② Hybrid Strategy (중기)**<br>예: SF 드라마, 판타지<br>제안: 유연한 홀드백 (45~60일)<br>이유: 극장과 OTT 수요 균형 필요 |
| **Low ONS (OTT 비친화형)** | **③ Niche Market (기타)**<br>예: 다큐, 실험 영화<br>제안: 자율 (시장 원리)<br>이유: 규제 실효성 낮음 | **④ Theater-Exclusive (장기)**<br>예: 액션 블록버스터<br>제안: 홀드백 보호 (90일 이상)<br>이유: 극장 수익 보호가 최우선 |

### 핵심 원칙

1. **One-Size-Fits-All 규제 지양**: 획일적 규제는 약자를 보호하는 것이 아니라 오히려 죽임
2. **장르별 핀셋 지원**: TFS/ONS 특성을 고려한 유연한 정책
3. **생태계 균형**: 배급사, 독립 제작사, 국내 OTT, 소비자 모두를 고려

---

## ⚠️ 주의사항

### 1. 데이터 의존성

- Model 2와 Model 3의 산출물에 의존
- 데이터 파일 경로가 정확해야 함
- 영화 코드 매칭 필요 (`movie_cd` vs `movieCd`)

### 2. 계산 시간

- Part 2: 각 그룹당 5개 영화 샘플링 (전체 계산 시 시간 증가)
- Part 3: 모든 영화에 대해 정책 적용 (약 20-30분)
- 샘플링 조정 가능: `group_movies = movie_meta[...].head(5)` 변경

### 3. 한계

- **극장 수익 예측**: LSTM 없이 간소화된 모델 사용 (제작비 기반)
- **권리료 추정**: Model 3 CSV 데이터 사용 (실제 계약과 차이 가능)
- **잠식 계수**: 고정값 사용 (실제는 시간에 따라 변동 가능)

---

## 📚 참고 문헌

### 이론적 근거

1. **Sharma et al. (2021)**: Dynamic Holdback Strategies in Digital Markets
2. **Hennig-Thurau et al. (2007)**: Piracy Impact on Box Office Revenue
3. **Ma et al. (2014)**: Streaming Release Window Optimization

### 데이터 출처

1. **영화진흥위원회 (KOFIC)**: 영화 제작비, 장르, 개봉일
2. **와이즈앱 (Wiseapp)**: OTT 플랫폼 MAU
3. **나스미디어**: OTT 요금제 및 광고 수익 추정

---

## 📞 문의

- Model 2 관련: `../model_2/README.md` 참조
- Model 3 관련: `../model3/README.md` 참조
- 전체 개요: `../readmes/readme_model3_revised.md` 참조

---

**Last Updated**: 2024-11-21  
**Version**: 1.0  
**Created by**: Professional Data Analyst PhD in Entertainment Industry

---

**🎉 시뮬레이션을 통해 한국 영화 산업의 지속가능한 생태계를 위한 정책을 제안합니다!**


