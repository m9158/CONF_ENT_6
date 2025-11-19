# 🎬 Model 2: 동적 수익 곡선 예측 엔진 (Rb & Ra)

**한국 영화 시장의 동적 홀드백 최적화를 위한 Hybrid 예측 모델**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![LSTM](https://img.shields.io/badge/Model-LSTM-green)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [핵심 기능](#-핵심-기능)
3. [모델 구조](#-모델-구조)
4. [모델 평가](#-모델-평가)
5. [파일 구조](#-파일-구조)
6. [사용 방법](#-사용-방법)
7. [Part 3 시뮬레이션 연동](#-part-3-시뮬레이션-연동)
8. [주요 함수](#-주요-함수)
9. [Troubleshooting](#-troubleshooting)

---

## 🎯 프로젝트 개요

### 연구 배경

한국 영화 시장의 **독점적 홀드백(Exclusive Holdback)** 관행으로 인해:
- OTT 출시 전 극장 상영 종료
- **OTT 출시 후 극장 데이터 부재** (Missing Data)
- 전통적 데이터 학습으로 잠식률 측정 불가

### 해결 방안: Hybrid Pipeline

본 모델은 **데이터 기반 학습 + 이론 기반 시뮬레이션**을 결합:

```
┌──────────────────────────────────────────────────┐
│  Model 2-A: Rb(t) 예측 (LSTM)                   │
│  → OTT 미개입 시 자연 수익 곡선                  │
│  → 실제 데이터 학습 (KOBIS)                     │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│  Model 2-B: Ra(t) 시뮬레이션 (Rule-Based)       │
│  → OTT 출시 후 잠식 수익 곡선                    │
│  → 장르 특성(TFS/ONS) + 소비자 선호도(γ) 반영   │
└──────────────────────────────────────────────────┘
```

---

## ✨ 핵심 기능

### 1. Rb(t) 예측: 자연 수익 곡선

**입력 피처** (6개):
```
1. day_number       - 개봉 후 경과일
2. is_weekend       - 주말 여부
3. screen_cnt       - 스크린 수
4. aud_per_show     - 회당 관객 수 (좌석 점유율 대리)
5. competition_index - HHI 경쟁 강도 (CI', α 대리)
6. social_buzz      - 네이버 검색 지수 (WOM)
```

**모델**: Stacked LSTM
```
LSTM(64, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(32, return_sequences=False)
    ↓
Dropout(0.2)
    ↓
Dense(16, relu)
    ↓
Dense(1) → daily_sales_amt
```

**특징**:
- ✅ 과거 7일 데이터 → 8일째 매출 예측
- ✅ Rolling Prediction으로 최대 180일 예측
- ✅ 시계열 패턴 학습 (주말 효과, 경쟁, WOM 등)

---

### 2. Ra(t) 시뮬레이션: 잠식 수익 곡선

**공식**:
```
Ra(t) = Rb(t) × (1 - C)  (t ≥ 홀드백 시점)

잠식 계수(C):
C = BaseRate × (1 + ONS_norm) × (1 - TFS_norm) × (0.5 + γ_norm)
```

**변수 설명**:
| 변수 | 의미 | 범위 |
|------|------|------|
| **BaseRate** | 기본 잠식률 | 0.3 (30%) |
| **TFS** | Theatrical-First Score (극장 적합도) | 0~10 |
| **ONS** | OTT-Native Score (OTT 적합도) | 0~10 |
| **γ (Gamma)** | 소비자 선호도 (연도별) | 0~1 (Min-Max 정규화) |

**특징**:
- ✅ 장르별 차별화 (액션 vs 드라마)
- ✅ 연도별 소비자 트렌드 반영 (2023 vs 2024)
- ✅ 이론적 근거 (Sharma et al. 연구)

---

### 3. 통합 API

```python
# 원스톱 예측 파이프라인
days, Rb, Ra, info = predict_revenue_curves(
    movie_id='20124079',  # 범죄도시2
    holdback_days=30,     # 30일 홀드백
    horizon=180           # 180일 예측
)

# 시각화
plot_revenue_curves(movie_id='20124079', holdback_days=30, horizon=180)
```

---

## 🏗️ 모델 구조

### 7단계 데이터 파이프라인

```
STEP 1: 데이터 로딩 (5개 데이터셋)
    ↓
STEP 2: Feature Engineering (6개 피처)
    ↓
STEP 3: 시계열 변환 (7일→8일 시퀀스)
    ↓
STEP 4: LSTM 모델 훈련 (Rb 예측)
    ↓
STEP 5: Ra 시뮬레이션 (Gamma 반영)
    ↓
STEP 6: 통합 파이프라인 (API)
    ↓
STEP 7: 모델 저장 및 검증
```

**상세 구현**: `ipynb/main.ipynb` 참조

---

## 📊 모델 평가

### LSTM 성능 (Rb 예측)

| 평가 지표 | 값 | 평가 |
|----------|-----|------|
| **MAE** | 121,103,993 KRW | 평균 1.21억 원 오차 |
| **RMSE** | 228,324,045 KRW | 평균 2.28억 원 오차 |
| **R² Score** | **0.8129** | ✅ **81.3% 설명력 (우수)** |

**평가**:
- ✅ 영화 흥행 예측 모델로서 매우 양호한 성능
- ✅ R² > 0.81은 업계 표준 대비 우수
- 📌 개봉 후 WOM 등 예측 불가 변수로 인해 R² 0.5~0.7도 정상 범위

### 실제 vs 예측 분포

```
대각선에 가까운 분포 → 높은 예측 정확도
고매출 영화도 비교적 잘 예측
일부 이상치(Outlier)는 예상 외 흥행작
```

### 잠식 계수(C) 분포

| 통계량 | 값 | 설명 |
|--------|-----|------|
| **평균** | 0.270 | 평균 27% 잠식 |
| **최소** | 0.135 | 극장 이벤트형 (High TFS) |
| **최대** | 0.405 | OTT 친화형 (High ONS) |
| **표준편차** | 0.068 | 영화별 차이 존재 |

**장르별 잠식률**:
```
Action (액션):       20.5% ↓ (낮음, 극장 선호)
Romance (로맨스):    34.2% ↑ (높음, OTT 선호)
Drama (드라마):      29.8% (중간)
Thriller (스릴러):   26.1% (중간)
Horror (호러):       31.5% (약간 높음)
```

---

## 📁 파일 구조

```
git_submission/
│
├── README.md                    # 📘 본 문서
│
├── ipynb/
│   └── main.ipynb              # 🔬 전체 구축 노트북 (7단계)
│
└── model/
    ├── model_Rb_best.h5        # 🏆 Best 체크포인트 (권장)
    ├── model_2A_Rb_LSTM.h5     # 🧠 최종 학습 모델
    ├── scaler_X.pkl            # 📊 입력 피처 스케일러
    ├── scaler_y.pkl            # 📊 타겟 변수 스케일러
    └── movie_meta_with_cannib.csv  # 📄 영화 메타데이터
```

---

## 🗂️ 모델 파일 설명

### 1. model_Rb_best.h5 ⭐ **권장**

**특징**:
- ✅ **Validation loss 최저** 시점의 모델
- ✅ **Epoch 24**: val_loss = **0.03731** (최저)
- ✅ **Overfitting 방지**: EarlyStopping과 함께 사용
- ✅ **일반화 성능 우수**: 검증 데이터에서 입증

**사용 시점**:
- ModelCheckpoint가 자동 저장
- Validation loss가 개선될 때마다 업데이트
- 최종적으로 가장 좋은 성능의 모델 저장

**권장 사유**:
```
✅ 실무 표준: Best checkpoint 사용이 업계 권장 사항
✅ 신뢰성: 검증 데이터에서 최고 성능 입증
✅ 안정성: Overfitting 리스크 최소화
```

### 2. model_2A_Rb_LSTM.h5

**특징**:
- 📌 **최종 epoch** 종료 후 저장된 모델
- 📌 더 학습했지만 validation 성능은 불명확
- 📌 Overfitting 가능성 존재

**사용 시점**:
- 훈련 완료 후 `model_Rb.save()` 실행
- 마지막 상태 그대로 저장

**비교**:
```
model_Rb_best.h5       → Epoch 24 (val_loss: 0.03731) ✅
model_2A_Rb_LSTM.h5    → Epoch 33 (val_loss: 미확인)
```

### 🎯 사용 권장 사항

| 용도 | 권장 모델 | 이유 |
|------|----------|------|
| **Part 3 시뮬레이션** | `model_Rb_best.h5` | 최고 일반화 성능 |
| **재현 실험** | `model_Rb_best.h5` | 논문/연구 표준 |
| **추가 학습 (Fine-tuning)** | `model_2A_Rb_LSTM.h5` | 마지막 상태에서 계속 |
| **비교 분석** | 두 모델 모두 | 성능 차이 확인 |

**기본 선택**: ✅ **model_Rb_best.h5** 사용 권장

---

## 📊 메타데이터 파일 (movie_meta_with_cannib.csv)

**주요 컬럼**:

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| `movieCd` | 영화 코드 | '20124079' |
| `movieNm` | 영화 제목 | '범죄도시2' |
| `openDt` | 개봉일 | '2022-05-18' |
| `genre_en` | 장르 (영문) | 'Action' |
| `TFS` | 극장 적합도 | 8.4 (0~10) |
| `ONS` | OTT 적합도 | 4.2 (0~10) |
| `gamma` | 소비자 선호도 | 0.827 (0~1) |
| `cannibalization_coef` | 잠식 계수 | 0.285 (0~1) |

**행 수**: ~500개 영화  
**용도**: 영화 정보 조회 및 잠식 계수 적용

---

## 🚀 사용 방법

### 1. 환경 설정

```bash
# Python 3.8 이상 필요
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn joblib
```

### 2. 노트북 실행 (전체 구축)

```bash
jupyter notebook ipynb/main.ipynb
```

**실행 순서** (STEP 1~7 순차적으로):
- STEP 1-3: 데이터 준비 및 전처리
- **STEP 4: LSTM 훈련** (⏱️ 약 10분 소요)
- STEP 5-7: 시뮬레이션 및 저장

### 3. 학습된 모델 사용 (Part 3용)

```python
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# ===========================
# 1. 모델 및 데이터 로드
# ===========================
model_Rb = load_model('model/model_Rb_best.h5')  # ⭐ Best 모델 권장
scaler_X = pickle.load(open('model/scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('model/scaler_y.pkl', 'rb'))
movie_meta = pd.read_csv('model/movie_meta_with_cannib.csv', encoding='utf-8')

print("✅ 모델 로드 완료")
print(f"   - LSTM 모델: {model_Rb.count_params():,} 파라미터")
print(f"   - 영화 수: {len(movie_meta)}개")

# ===========================
# 2. 함수 정의 (main.ipynb에서 복사)
# ===========================

def predict_revenue_curves(movie_id, holdback_days=30, horizon=180, verbose=True):
    """
    특정 영화의 Rb, Ra 곡선을 예측
    
    Parameters:
    - movie_id: 영화 ID (str)
    - holdback_days: 홀드백 기간 (int)
    - horizon: 예측 기간 (int, 최대 180일)
    - verbose: 출력 여부 (bool)
    
    Returns:
    - days: 날짜 배열 (1~horizon)
    - Rb_curve: Rb(t) 예측 곡선 (numpy array)
    - Ra_curve: Ra(t) 시뮬레이션 곡선 (numpy array)
    - movie_info: 영화 정보 (dict)
    """
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"영화 ID {movie_id} 수익 곡선 예측 시작")
        print(f"{'='*50}")
    
    # 1. 영화 메타데이터 가져오기
    movie_data = movie_meta[movie_meta['movieCd'] == str(movie_id)]
    
    if len(movie_data) == 0:
        raise ValueError(f"영화 {movie_id}를 찾을 수 없습니다.")
    
    movie_info_dict = movie_data.iloc[0].to_dict()
    cannib_coef = movie_info_dict['cannibalization_coef']
    
    if verbose:
        print(f"[1] 영화: {movie_info_dict.get('movieNm', 'Unknown')}")
        print(f"    장르: {movie_info_dict.get('genre_en', 'Unknown')}")
        print(f"    TFS: {movie_info_dict.get('TFS', 0):.1f}")
        print(f"    ONS: {movie_info_dict.get('ONS', 0):.1f}")
        print(f"    잠식계수(C): {cannib_coef:.3f}")
    
    # 2. 초기 7일 데이터 준비 (실제 구현 시 performance 데이터에서 가져와야 함)
    # 여기서는 간소화된 버전 - main.ipynb의 전체 코드 참조
    
    # ... (Rolling Prediction 로직 생략, main.ipynb 참조)
    
    # 3. Ra 시뮬레이션
    Ra_curve = Rb_curve.copy()
    Ra_curve[holdback_days:] = Rb_curve[holdback_days:] * (1 - cannib_coef)
    
    days = np.arange(1, horizon + 1)
    
    if verbose:
        print(f"\n[결과]")
        print(f"  총 Rb: {np.sum(Rb_curve):,.0f} KRW")
        print(f"  총 Ra: {np.sum(Ra_curve):,.0f} KRW")
        print(f"  잠식률: {(1 - np.sum(Ra_curve)/np.sum(Rb_curve))*100:.1f}%")
    
    return days, Rb_curve, Ra_curve, movie_info_dict

# ===========================
# 3. 사용 예제
# ===========================
movie_id = '20124079'  # 범죄도시2
days, Rb, Ra, info = predict_revenue_curves(movie_id, holdback_days=30, horizon=180)

print(f"\n✅ 예측 완료!")
```

---

## 🔗 Part 3 시뮬레이션 연동

Model 2는 **Part 3 (산업 생태계 시뮬레이션)**의 핵심 예측 엔진입니다.

### 시나리오 1: 개별 영화 최적 홀드백 찾기

**목표**: 배급사 수익(ΠM) 극대화하는 t* 찾기

```python
# 홀드백 시나리오별 수익 계산
movie_id = '20124079'
holdback_scenarios = range(0, 181, 10)  # 0~180일, 10일 간격
results = []

for t_sim in holdback_scenarios:
    # Rb/Ra 예측
    days, Rb, Ra, info = predict_revenue_curves(
        movie_id, 
        holdback_days=t_sim, 
        horizon=180,
        verbose=False
    )
    
    # 극장 수익 계산
    theater_revenue = np.sum(Rb[:t_sim]) + np.sum(Ra[t_sim:])
    
    # OTT 권리료 계산 (Model 3 필요)
    tau = calculate_digital_fee(t_sim, info['TFS'], info['ONS'])
    
    # 총수익
    total_profit = theater_revenue + tau
    
    results.append({
        'holdback': t_sim,
        'theater_revenue': theater_revenue,
        'ott_fee': tau,
        'total_profit': total_profit
    })

# 최적 홀드백 도출
import pandas as pd
results_df = pd.DataFrame(results)
optimal = results_df.loc[results_df['total_profit'].idxmax()]

print(f"✅ 최적 홀드백: {optimal['holdback']:.0f}일")
print(f"   - 극장 수익: {optimal['theater_revenue']/1e8:.1f}억")
print(f"   - OTT 권리료: {optimal['ott_fee']/1e8:.1f}억")
print(f"   - 총수익: {optimal['total_profit']/1e8:.1f}억")
```

**출력 예시**:
```
✅ 최적 홀드백: 60일
   - 극장 수익: 1,100억
   - OTT 권리료: 150억
   - 총수익: 1,250억
```

---

### 시나리오 2: 영화 유형별 전략 비교

**목표**: TFS/ONS에 따른 최적 전략 도출

```python
# High TFS (블록버스터) vs High ONS (드라마)
high_tfs_movies = movie_meta[movie_meta['TFS'] >= 7.0]
high_ons_movies = movie_meta[movie_meta['ONS'] >= 7.0]

def find_avg_optimal_holdback(movie_list, n_samples=10):
    """영화 리스트의 평균 최적 홀드백 계산"""
    optimal_holdbacks = []
    
    for idx, movie in movie_list.head(n_samples).iterrows():
        results = []
        for t in range(0, 181, 15):
            days, Rb, Ra, _ = predict_revenue_curves(
                movie['movieCd'], 
                holdback_days=t, 
                horizon=180,
                verbose=False
            )
            theater_rev = np.sum(Rb[:t]) + np.sum(Ra[t:])
            tau = calculate_digital_fee(t, movie['TFS'], movie['ONS'])
            total = theater_rev + tau
            results.append((t, total))
        
        optimal_t = max(results, key=lambda x: x[1])[0]
        optimal_holdbacks.append(optimal_t)
    
    return np.mean(optimal_holdbacks), np.std(optimal_holdbacks)

# 계산
tfs_mean, tfs_std = find_avg_optimal_holdback(high_tfs_movies)
ons_mean, ons_std = find_avg_optimal_holdback(high_ons_movies)

print(f"High TFS (블록버스터): {tfs_mean:.0f}±{tfs_std:.0f}일")
print(f"High ONS (드라마):     {ons_mean:.0f}±{ons_std:.0f}일")
```

**예상 결과**:
```
High TFS (블록버스터): 90±15일 (극장 보호)
High ONS (드라마):     30±10일 (빠른 OTT 전환)
```

---

### 시나리오 3: 정책 시나리오 평가

**목표**: 산업 전체 효용(W_Industry) 비교

```python
# 정책 정의
policies = {
    'Laissez-faire': lambda movie: find_optimal_holdback(movie),  # 자율
    'Uniform_90': lambda movie: 90,                               # 일괄 90일
    'Dynamic': lambda movie: 90 if movie['TFS'] >= 7 else 30     # 동적 차등
}

# 이해관계자별 효용 계산
def calculate_industry_welfare(policy_fn, movie_sample):
    """정책에 따른 산업 전체 효용 계산"""
    
    total_welfare = 0
    
    for idx, movie in movie_sample.iterrows():
        t = policy_fn(movie)
        days, Rb, Ra, _ = predict_revenue_curves(
            movie['movieCd'], 
            holdback_days=t, 
            horizon=180,
            verbose=False
        )
        
        # 배급사 효용
        theater_rev = np.sum(Rb[:t]) + np.sum(Ra[t:])
        tau = calculate_digital_fee(t, movie['TFS'], movie['ONS'])
        U_MD = theater_rev + tau
        
        # 소비자 효용 (간소화)
        U_Consumer = -0.001 * t * movie['gamma']  # 홀드백 길수록 불편
        
        total_welfare += (U_MD + U_Consumer)
    
    return total_welfare

# 정책 비교
movie_sample = movie_meta.sample(50, random_state=42)

for policy_name, policy_fn in policies.items():
    welfare = calculate_industry_welfare(policy_fn, movie_sample)
    print(f"{policy_name:15s}: {welfare/1e12:.2f}조")
```

**출력 예시**:
```
Laissez-faire  : 1.85조 (최대 수익)
Uniform_90     : 1.62조 (독립 영화 타격)
Dynamic        : 1.78조 (균형적) ✅
```

---

## 🔧 주요 함수

### 1. predict_revenue_curves()

**시그니처**:
```python
def predict_revenue_curves(movie_id, holdback_days=30, horizon=180, verbose=True):
    """
    영화의 Rb/Ra 곡선 예측
    
    Returns:
        days: 날짜 배열 (1~horizon)
        Rb_curve: 자연 수익 곡선
        Ra_curve: 잠식 수익 곡선
        movie_info: 영화 정보 dict
    """
```

**주요 로직**:
1. 영화 메타데이터 조회 (TFS, ONS, 잠식계수)
2. Rolling Prediction으로 Rb 예측
3. 잠식 계수 적용하여 Ra 시뮬레이션
4. 결과 반환

### 2. calculate_digital_fee() (Model 3 필요)

**공식**:
```python
τ(t, TFS, ONS) = (Total Cost × R%) × (1 + ONS) × 1/(1 + d(TFS)·t)

- R%: 제작비 대비 판권료 비율 (10~15%)
- d(TFS): 감가상각률 (TFS 높을수록 시간에 민감)
```

### 3. simulate_Ra_from_Rb()

**시그니처**:
```python
def simulate_Ra_from_Rb(Rb_predictions, cannibalization_coef, holdback_day):
    """
    Rb에서 Ra 생성
    
    Returns:
        Ra: 잠식 수익 곡선
    """
    Ra = Rb_predictions.copy()
    if holdback_day < len(Ra):
        Ra[holdback_day:] = Rb_predictions[holdback_day:] * (1 - cannibalization_coef)
    return Ra
```

---

## 💡 주요 특징

### 1. α 분리 (Model 1과 독립)

```
Model 1의 α(성공 잠재력) 사용 안 함
  ↓
HHI(CI')로 경쟁 강도 측정
  ↓
모델 무결성 확보
```

### 2. Gamma 정규화 (v2.0 업데이트)

```
이전: Log(OTT율/극장횟수) → exp() 변환 → 과소평가 ❌
신규: Log → Min-Max Scaling (0~1) → 정확 반영 ✅
```

### 3. Hybrid Pipeline

```
데이터 학습 (Rb)
    +
이론 시뮬레이션 (Ra)
    =
Missing Data 문제 해결
```

---

## 🔍 Troubleshooting

### Q1: 모델 로드 오류

```python
# 오류: TensorFlow 버전 불일치
# 해결:
pip install tensorflow==2.13.0  # 또는 2.x 버전

# 확인:
import tensorflow as tf
print(tf.__version__)
```

### Q2: 어떤 모델 파일을 사용해야 하나요?

**권장**: `model_Rb_best.h5` ✅
- Validation loss 최저 (0.03731)
- 일반화 성능 우수
- Overfitting 방지

**대안**: `model_2A_Rb_LSTM.h5`
- 최종 epoch 모델
- 추가 학습(Fine-tuning)용

### Q3: 예측 시간이 너무 오래 걸려요

```python
# 해결책 1: horizon 줄이기
days, Rb, Ra, info = predict_revenue_curves(
    movie_id, 
    holdback_days=30, 
    horizon=90  # 180 → 90으로
)

# 해결책 2: GPU 사용
# TensorFlow가 GPU를 자동 감지
# CUDA 설치 확인: nvidia-smi
```

### Q4: Gamma 값이 이상해요

```python
# 확인:
print(movie_meta['gamma'].describe())
# mean: ~0.5, min: 0, max: 1 정상

# 문제 시 gamma 재계산 (main.ipynb Cell 23-24 참조)
```

### Q5: 메모리 부족 오류

```python
# 해결책: 배치 사이즈 줄이기
# main.ipynb에서 시퀀스 생성 시
# 영화 개수 제한 또는 horizon 축소
```

---

## 📚 참고 자료

### 내부 문서
- `ipynb/main.ipynb`: 전체 구현 코드 (7단계)

### 이론적 배경
- **Sharma et al. (2021)**: Dynamic Holdback Strategies
- **OTT Suitability Index**: TFS/ONS 점수 방법론
- **KOBIS**: 영화진흥위원회 박스오피스 데이터

### 데이터 소스
- Daily_Performance.csv: 일별 극장 성과
- kobis.csv: KOBIS Top 10
- Online_Buzz_filtered_v1.csv: 검색 지수
- movie_metadata.csv: 영화 메타데이터
- OTT_suitability_score.csv: 장르별 TFS/ONS
- Consumer_Preference.csv: 소비자 선호도 (Gamma 계산용)

---

## 📝 업데이트 내역

### v2.0 (2024-11-19) - Gamma Normalization
- ✅ Gamma 계산 방식 변경: `exp()` → Min-Max Scaling
- ✅ 잠식 계수 과소평가 문제 해결
- ✅ 두 모델 파일 모두 포함 (Best + Final)

### v1.0 (2024-11-10) - Initial Release
- ✅ LSTM 모델 구축 완료 (R² = 0.76)
- ✅ Ra 시뮬레이션 함수 구현
- ✅ 통합 파이프라인 API 제공

---

## 🌟 핵심 요약

```
┌─────────────────────────────────────────────────────────────┐
│                     Model 2 핵심 요약                        │
├─────────────────────────────────────────────────────────────┤
│ ✅ Rb(t): LSTM으로 자연 수익 곡선 예측 (R² = 0.76)          │
│ ✅ Ra(t): Rule-Based로 잠식 수익 곡선 시뮬레이션            │
│ ✅ Gamma: Min-Max Scaling으로 소비자 선호도 정확 반영       │
│ ✅ 2개 모델: Best (권장) + Final (대안)                     │
│ ✅ API: predict_revenue_curves() 원스톱 파이프라인          │
│ ✅ 용도: Part 3 시뮬레이션의 핵심 예측 엔진                 │
└─────────────────────────────────────────────────────────────┘
```

**Model 2로 할 수 있는 것**:
1. ✅ 개별 영화의 Rb/Ra 곡선 예측
2. ✅ 홀드백 기간별 극장 수익 시뮬레이션
3. ✅ 영화 유형별 최적 전략 도출
4. ✅ 정책 시나리오 평가 (Part 3 연동)

---

## 📧 문의

이 프로젝트는 한국 영화 시장의 홀드백 최적화 연구의 일환입니다.

**Created**: 2024-11-19  
**Version**: 2.0 (Gamma Normalization)

---

**⭐ 학습된 모델을 활용하여 Part 3 시뮬레이션을 진행하세요!**


