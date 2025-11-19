# 📘 Model 2 사용 가이드 (Part 3 시뮬레이션용)

## 📋 개요

이 문서는 **Model 2 (동적 수익 곡선 예측 엔진)**의 산출물을 **Part 3 (산업 생태계 시뮬레이션)**에서 어떻게 활용하는지 설명합니다.

---

## 🎯 Model 2의 역할

Model 2는 Part 3 시뮬레이션의 **핵심 예측 엔진**으로, 다음을 제공합니다:

1. **Rb(t) 예측**: OTT 개입이 없을 때의 자연스러운 극장 수익 곡선
2. **Ra(t) 시뮬레이션**: 특정 홀드백(t) 적용 시 OTT 출시 후 극장 수익 곡선
3. **영화별 특성**: TFS, ONS, 잠식 계수(C), Gamma(γ)

---

## 📦 Model 2 산출물 (Outputs)

### 1. 학습된 LSTM 모델 (`model_2A_Rb_LSTM.h5`)

- **역할**: Rb(t) 예측 (자연 수익 곡선)
- **입력**: 과거 7일간의 6개 피처 시퀀스
- **출력**: 8일째 일일 극장 매출액
- **용도**: Rolling Prediction으로 전체 생애 주기(180일) 수익 곡선 생성

```python
# 로드 방법
from tensorflow.keras.models import load_model
model_Rb = load_model('model_2/model_2A_Rb_LSTM.h5')
```

---

### 2. 스케일러 (`scaler_X.pkl`, `scaler_y.pkl`)

- **scaler_X**: 입력 피처 정규화 (6개 피처)
- **scaler_y**: 타겟 변수 정규화 (daily_sales_amt)
- **용도**: 예측 전/후 스케일 변환

```python
# 로드 방법
import pickle
scaler_X = pickle.load(open('model_2/scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('model_2/scaler_y.pkl', 'rb'))
```

---

### 3. 영화 메타데이터 (`movie_meta_with_cannib.csv`)

- **포함 컬럼**:
  - `movie_cd`: 영화 코드
  - `movie_nm`: 영화명
  - `genre`, `genre_en`: 장르 (한글/영문)
  - `openDt`: 개봉일
  - `open_year`: 개봉 연도
  - `TFS`: Theatrical-First Score (극장 적합도, 0~10)
  - `ONS`: OTT-Native Score (OTT 적합도, 0~10)
  - `gamma`: 소비자 선호도 (Log 값, 연도별)
  - `cannibalization_coef`: 잠식 계수 (C, 0~1)

```python
# 로드 방법
import pandas as pd
movie_meta = pd.read_csv('model_2/movie_meta_with_cannib.csv', encoding='utf-8')
```

---

### 4. 통합 파이프라인 함수

#### ① `predict_revenue_curves(movie_id, holdback_days, horizon)`

- **목적**: 특정 영화의 Rb/Ra 곡선 예측
- **파라미터**:
  - `movie_id` (str): 영화 코드
  - `holdback_days` (int): 홀드백 기간 (기본값: 30일)
  - `horizon` (int): 예측 기간 (기본값: 180일)
- **반환값**:
  - `days`: 날짜 배열 (1~180)
  - `Rb_curve`: Rb(t) 예측값 배열
  - `Ra_curve`: Ra(t) 시뮬레이션 값 배열
  - `movie_info`: 영화 메타데이터 (dict)

#### ② `plot_revenue_curves(movie_id, holdback_days, horizon)`

- **목적**: Rb/Ra 곡선 시각화
- **출력**: 
  - Rb vs Ra 비교 그래프
  - 잠식 영역 표시
  - 총수익 통계

---

## 🔗 Part 3 시뮬레이션 활용 방법

### 시나리오 1: 개별 영화 최적 홀드백 찾기 (Part 3-1)

**목표**: 특정 영화의 배급사 수익(ΠM)을 극대화하는 최적 홀드백(t*) 찾기

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# ===========================
# 1. Model 2 산출물 로드
# ===========================
model_Rb = load_model('model_2/model_2A_Rb_LSTM.h5')
scaler_X = pickle.load(open('model_2/scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('model_2/scaler_y.pkl', 'rb'))
movie_meta = pd.read_csv('model_2/movie_meta_with_cannib.csv', encoding='utf-8')

# ===========================
# 2. 타겟 영화 선택
# ===========================
target_movie_id = '20124079'  # 예: 범죄도시2
movie_info = movie_meta[movie_meta['movie_cd'] == target_movie_id].iloc[0]

print(f"영화: {movie_info['movie_nm']}")
print(f"장르: {movie_info['genre_en']}")
print(f"TFS: {movie_info['TFS']:.1f} | ONS: {movie_info['ONS']:.1f}")
print(f"잠식계수(C): {movie_info['cannibalization_coef']:.3f}")

# ===========================
# 3. 홀드백 시나리오 시뮬레이션
# ===========================
holdback_scenarios = range(0, 181, 10)  # 0일~180일, 10일 간격
results = []

for t_sim in holdback_scenarios:
    # Rb/Ra 예측
    days, Rb_curve, Ra_curve, _ = predict_revenue_curves(
        target_movie_id, 
        holdback_days=t_sim, 
        horizon=180
    )
    
    # 극장 수익 계산
    theater_revenue_before = np.sum(Rb_curve[:t_sim])  # 홀드백 이전
    theater_revenue_after = np.sum(Ra_curve[t_sim:])   # 홀드백 이후
    total_theater_revenue = theater_revenue_before + theater_revenue_after
    
    # OTT 권리료 계산 (Model 3 연동)
    tau = calculate_digital_fee(
        holdback=t_sim,
        tfs=movie_info['TFS'],
        ons=movie_info['ONS'],
        production_cost=movie_info.get('production_cost', 5e9)  # 기본 50억
    )
    
    # 총수익 (배급사)
    total_profit = total_theater_revenue + tau
    
    results.append({
        'holdback': t_sim,
        'theater_revenue': total_theater_revenue,
        'ott_fee': tau,
        'total_profit': total_profit
    })

# ===========================
# 4. 최적 홀드백 도출
# ===========================
results_df = pd.DataFrame(results)
optimal_row = results_df.loc[results_df['total_profit'].idxmax()]

print(f"\n✅ 최적 홀드백: {optimal_row['holdback']:.0f}일")
print(f"   - 극장 수익: {optimal_row['theater_revenue']/1e8:.1f}억")
print(f"   - OTT 권리료: {optimal_row['ott_fee']/1e8:.1f}억")
print(f"   - 총수익: {optimal_row['total_profit']/1e8:.1f}억")
```

---

### 시나리오 2: 영화 유형별 최적 홀드백 분석 (Part 3-2)

**목표**: TFS/ONS에 따른 영화 유형별 홀드백 전략 비교

```python
# ===========================
# 1. 영화 유형 분류
# ===========================
# High TFS (극장 이벤트형)
high_tfs_movies = movie_meta[movie_meta['TFS'] >= 7.0]

# High ONS (OTT 친화형)
high_ons_movies = movie_meta[movie_meta['ONS'] >= 7.0]

# ===========================
# 2. 유형별 최적 홀드백 계산
# ===========================
def find_optimal_holdback(movie_list, holdback_range=range(0, 181, 10)):
    """영화 리스트의 평균 최적 홀드백 계산"""
    optimal_holdbacks = []
    
    for idx, movie in movie_list.iterrows():
        results = []
        for t_sim in holdback_range:
            days, Rb, Ra, _ = predict_revenue_curves(
                movie['movie_cd'], 
                holdback_days=t_sim, 
                horizon=180
            )
            theater_rev = np.sum(Rb[:t_sim]) + np.sum(Ra[t_sim:])
            tau = calculate_digital_fee(t_sim, movie['TFS'], movie['ONS'])
            total = theater_rev + tau
            results.append((t_sim, total))
        
        optimal_t = max(results, key=lambda x: x[1])[0]
        optimal_holdbacks.append(optimal_t)
    
    return np.mean(optimal_holdbacks), np.std(optimal_holdbacks)

# High TFS 영화들의 최적 홀드백
tfs_mean, tfs_std = find_optimal_holdback(high_tfs_movies.head(10))
print(f"High TFS 영화 최적 홀드백: {tfs_mean:.0f}일 (±{tfs_std:.0f})")

# High ONS 영화들의 최적 홀드백
ons_mean, ons_std = find_optimal_holdback(high_ons_movies.head(10))
print(f"High ONS 영화 최적 홀드백: {ons_mean:.0f}일 (±{ons_std:.0f})")
```

**예상 결과**:
- High TFS (블록버스터): 90~120일 (극장 수익 극대화)
- High ONS (드라마/로맨스): 30~45일 (빠른 OTT 전환)

---

### 시나리오 3: 정책 시나리오 비교 (Part 3-3)

**목표**: 산업 전체 효용(W_Industry) 비교

```python
# ===========================
# 정책 시나리오 정의
# ===========================
policies = {
    'Laissez-faire': {
        'description': '완전 자율 (영화별 최적 홀드백)',
        'apply_fn': lambda movie: find_movie_optimal_holdback(movie)
    },
    'Uniform_90': {
        'description': '일괄 규제 (모든 영화 90일)',
        'apply_fn': lambda movie: 90
    },
    'Dynamic': {
        'description': '동적 차등 (TFS/ONS 기반)',
        'apply_fn': lambda movie: 90 if movie['TFS'] >= 7 else 30
    }
}

# ===========================
# 이해관계자별 효용 계산
# ===========================
def calculate_stakeholder_utilities(policy_name, holdback_fn):
    """정책에 따른 이해관계자별 효용 계산"""
    
    # 배급사 효용 (U_MD)
    U_MD = 0
    for idx, movie in movie_meta.iterrows():
        t = holdback_fn(movie)
        days, Rb, Ra, _ = predict_revenue_curves(movie['movie_cd'], t, 180)
        theater_rev = np.sum(Rb[:t]) + np.sum(Ra[t:])
        tau = calculate_digital_fee(t, movie['TFS'], movie['ONS'])
        U_MD += (theater_rev + tau)
    
    # 독립 제작사 효용 (U_Indie)
    indie_movies = movie_meta[movie_meta['TFS'] < 5.0]  # Low TFS
    U_Indie = 0
    discount_rate = 0.2  # 연 20%
    for idx, movie in indie_movies.iterrows():
        t = holdback_fn(movie)
        # 현금 흐름 할인율 적용
        tau = calculate_digital_fee(t, movie['TFS'], movie['ONS'])
        discounted_tau = tau / (1 + discount_rate * t/365)
        U_Indie += discounted_tau
    
    # 국내 OTT 효용 (U_Local_OTT)
    ott_friendly = movie_meta[movie_meta['ONS'] >= 7.0]  # High ONS
    U_OTT = 0
    for idx, movie in ott_friendly.iterrows():
        t = holdback_fn(movie)
        # 신선도 함수: 홀드백이 길수록 가치 감소
        freshness = np.exp(-0.01 * t)  # Exponential decay
        content_value = movie['ONS'] * freshness
        U_OTT += content_value
    
    # 소비자 효용 (U_Consumer)
    U_Consumer = 0
    for idx, movie in movie_meta.iterrows():
        t = holdback_fn(movie)
        # 홀드백이 길수록 불법 복제 위험 증가
        piracy_penalty = 0.001 * t * movie['gamma']  # Gamma 반영
        U_Consumer -= piracy_penalty
    
    return {
        'policy': policy_name,
        'U_MD': U_MD,
        'U_Indie': U_Indie,
        'U_OTT': U_OTT,
        'U_Consumer': U_Consumer,
        'W_Industry': U_MD + U_Indie + U_OTT + U_Consumer
    }

# ===========================
# 정책별 시뮬레이션 실행
# ===========================
policy_results = []
for policy_name, policy_config in policies.items():
    print(f"\n🔄 시뮬레이션: {policy_config['description']}")
    result = calculate_stakeholder_utilities(policy_name, policy_config['apply_fn'])
    policy_results.append(result)
    print(f"   - 산업 전체 효용(W): {result['W_Industry']/1e12:.2f}조")

# ===========================
# 결과 비교
# ===========================
results_df = pd.DataFrame(policy_results)
best_policy = results_df.loc[results_df['W_Industry'].idxmax()]

print(f"\n✅ 최적 정책: {best_policy['policy']}")
print(f"   - 배급사 효용: {best_policy['U_MD']/1e12:.2f}조")
print(f"   - 독립 제작사: {best_policy['U_Indie']/1e12:.2f}조")
print(f"   - 국내 OTT: {best_policy['U_OTT']:.2f}")
print(f"   - 소비자: {best_policy['U_Consumer']:.2f}")
print(f"   - 전체 효용: {best_policy['W_Industry']/1e12:.2f}조")
```

---

## 📊 데이터 흐름 (Model 2 → Part 3)

```
┌──────────────────────────────────────────────────────────┐
│                    MODEL 2 산출물                         │
├──────────────────────────────────────────────────────────┤
│ 1. model_2A_Rb_LSTM.h5 (LSTM 모델)                      │
│ 2. scaler_X.pkl, scaler_y.pkl (스케일러)                 │
│ 3. movie_meta_with_cannib.csv (영화 메타 + 잠식계수)    │
│ 4. predict_revenue_curves() 함수                        │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│              PART 3-1: 개별 최적화                        │
├──────────────────────────────────────────────────────────┤
│ Input: movie_id, holdback (0~180일)                     │
│                                                          │
│ Process:                                                 │
│  1. predict_revenue_curves(movie_id, t_sim)             │
│     → Rb(t), Ra(t) 예측                                  │
│                                                          │
│  2. 극장 수익 계산:                                       │
│     ∫[0→t] Rb(z)dz + ∫[t→T] Ra(z)dz                     │
│                                                          │
│  3. OTT 권리료 계산 (Model 3):                           │
│     τ(t, TFS, ONS)                                      │
│                                                          │
│  4. 총수익 계산:                                          │
│     ΠM = 극장 수익 + τ                                   │
│                                                          │
│ Output: 최적 홀드백(t*) 및 최대 수익                     │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│            PART 3-2: 영화 유형별 분석                     │
├──────────────────────────────────────────────────────────┤
│ Segmentation:                                            │
│  - High TFS (이벤트형)   → 최적 t 평균                   │
│  - High ONS (OTT형)      → 최적 t 평균                   │
│  - Balanced              → 최적 t 평균                   │
│                                                          │
│ Output: TFS/ONS 매트릭스 + 최적 홀드백 히트맵            │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│          PART 3-3: 산업 생태계 시뮬레이션                 │
├──────────────────────────────────────────────────────────┤
│ Policy Scenarios:                                        │
│  1. Laissez-faire (자율)                                │
│  2. Uniform 90 (일괄 규제)                               │
│  3. Dynamic (동적 차등)                                  │
│                                                          │
│ Stakeholder Utilities:                                  │
│  - U_MD (배급사)                                         │
│  - U_Indie (독립 제작사)                                 │
│  - U_Local_OTT (국내 OTT)                                │
│  - U_Consumer (소비자)                                   │
│                                                          │
│ Objective: Maximize W_Industry                          │
│           s.t. U_Indie ≥ Min_Threshold                  │
│                U_OTT ≥ Competition_Threshold            │
│                                                          │
│ Output: 최적 정책 + 제약 조건 만족 여부                  │
└──────────────────────────────────────────────────────────┘
```

---

## 🔑 핵심 변수 매핑

| Model 2 산출물 | Part 3 사용처 | 설명 |
|----------------|--------------|------|
| `Rb(t)` | 극장 수익 계산 | OTT 미개입 시 자연 수익 곡선 |
| `Ra(t)` | 극장 수익 계산 | 홀드백 t 이후 잠식 반영 곡선 |
| `TFS` | τ(t) 계산, 영화 분류 | 극장 적합도 (이벤트성) |
| `ONS` | τ(t) 계산, 영화 분류 | OTT 적합도 (몰입도) |
| `cannibalization_coef` | Ra(t) 생성 | 잠식 계수 C |
| `gamma` | Ra(t) 생성, 소비자 효용 | 연도별 소비자 선호도 (0~1 정규화) |

---

## ⚠️ 주의사항

### 1. Rolling Prediction 누적 오차

- **문제**: 180일 예측 시 오차 누적 가능
- **해결**:
  - 초기 7일은 실제 데이터 사용
  - 14일마다 실측 데이터로 재보정 (가능 시)
  - 신뢰구간 표시 (±1σ)

### 2. Gamma 계산 및 적용 방법 (업데이트됨)

- **계산 방법**:
  1. Log 변환: `γ_log = Log(OTT이용률 / 극장방문횟수)`
  2. Min-Max Scaling: `γ_norm = (γ_log - min) / (max - min)`
  3. 결과: 0~1 범위 (0 = 극장 선호, 1 = OTT 선호)
  
- **잠식 계수 적용**:
  - `gamma_multiplier = 0.5 + gamma`
  - gamma=0 → 0.5 (잠식 50% 감소)
  - gamma=0.5 → 1.0 (중립)
  - gamma=1 → 1.5 (잠식 50% 증가)

- **현재**: 연도별 전체 연령대 평균 gamma 사용
- **한계**: 영화별 타겟 연령대 차이 미반영
- **향후**: 관람등급, 장르별 주 관객층 데이터 확보 필요

- **⚠️ 중요 변경사항 (2024-11-19)**:
  - 기존: `exp(gamma)`로 변환 → 과소평가 문제
  - 개선: Min-Max Scaling으로 0~1 정규화 → 정확한 반영

### 3. Base Rate 파라미터

- **기본값**: 0.3 (30%)
- **민감도 분석 필요**:
  - Conservative: 0.15
  - Neutral: 0.30
  - Aggressive: 0.50

### 4. Model 3 (τ 추정) 연동

- Model 2는 Rb/Ra만 제공
- **디지털 권리료(τ)**는 별도 Model 3 필요:
  ```
  τ(t, TFS, ONS) = (Total Cost × R%) × (1 + ONS) × 1/(1 + d(TFS)·t)
  ```

---

## 📂 파일 구조

```
model_2/
├── main.ipynb                      # 모델 구축 노트북
├── model_2A_Rb_LSTM.h5            # 학습된 LSTM 모델
├── model_Rb_best.h5               # Best checkpoint
├── scaler_X.pkl                    # 입력 스케일러
├── scaler_y.pkl                    # 타겟 스케일러
├── movie_meta_with_cannib.csv     # 영화 메타 + 잠식 계수
├── MODEL_2_BUILD_PLAN.md          # 구축 계획서
└── model2_usage.md                # 본 문서 (사용 가이드)
```

---

## 🚀 빠른 시작 (Quick Start)

```python
# 1. 필요 라이브러리 import
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# 2. Model 2 산출물 로드
model_Rb = load_model('model_2/model_2A_Rb_LSTM.h5')
scaler_X = pickle.load(open('model_2/scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('model_2/scaler_y.pkl', 'rb'))
movie_meta = pd.read_csv('model_2/movie_meta_with_cannib.csv', encoding='utf-8')

# 3. main.ipynb의 함수 import (같은 환경에서 실행)
# %run model_2/main.ipynb  # Jupyter 환경
# 또는 함수 복사

# 4. 샘플 예측 실행
movie_id = movie_meta.iloc[0]['movie_cd']
days, Rb, Ra, info = predict_revenue_curves(movie_id, holdback_days=30, horizon=180)

print(f"영화: {info['movie_nm']}")
print(f"Rb 총합: {np.sum(Rb)/1e8:.1f}억")
print(f"Ra 총합: {np.sum(Ra)/1e8:.1f}억")
print(f"잠식률: {(1 - np.sum(Ra)/np.sum(Rb))*100:.1f}%")
```

---

## 📞 문의 및 이슈

- Model 2 구축 관련: `main.ipynb` 참조
- Part 3 시뮬레이션 설계: `readme.md` 또는 `readme_addedgammafuction.md` 참조
- 데이터 소스: `MODEL_2_BUILD_PLAN.md` 참조

---

**Last Updated**: 2024-11-19  
**Version**: 1.0

