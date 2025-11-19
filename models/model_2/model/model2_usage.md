# 📘 Model 2 사용 가이드

## 📦 필수 파일

```
model/
├── model_Rb_best.h5              # LSTM 모델 (권장)
├── model_2A_Rb_LSTM.h5           # LSTM 모델 (대안)
├── scaler_X.pkl                  # 입력 스케일러
├── scaler_y.pkl                  # 출력 스케일러
└── movie_meta_with_cannib.csv    # (선택) 기존 영화 참조용
```

---

## 🚀 사용 방법

### 환경 설정

```bash
pip install numpy pandas tensorflow scikit-learn
```

---

## 📌 시나리오 1: 새로운 영화 시뮬레이션

**상황**: 신규 개봉 예정 영화의 수익 곡선을 예측하고 싶을 때

### 필요한 정보

1. **초기 7일 극장 데이터** (개봉 후 실측)
2. **영화 특성**:
   - `TFS` (Theatrical-First Score): 장르로 추정 (Action: 8.4, Drama: 5.6 등)
   - `ONS` (OTT-Native Score): 장르로 추정 (Action: 4.2, Drama: 7.8 등)
   - `gamma`: 개봉 연도 평균 (2023: 0.827, 2024: 0.892 등)

### 코드 예제

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# ===========================
# 1. 모델 로드
# ===========================
model_Rb = load_model('model/model_Rb_best.h5')
scaler_X = pickle.load(open('model/scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('model/scaler_y.pkl', 'rb'))

print("✅ 모델 로드 완료")

# ===========================
# 2. 잠식 계수 계산 함수
# ===========================
def calculate_cannibalization_coefficient(tfs, ons, gamma, base_rate=0.3):
    """
    잠식 계수 계산
    
    Parameters:
    - tfs: Theatrical-First Score (0~10)
    - ons: OTT-Native Score (0~10)
    - gamma: 소비자 선호도 (0~1, Min-Max 정규화)
    - base_rate: 기본 잠식률 (기본값: 0.3)
    
    Returns:
    - C: 잠식 계수 (0~1)
    """
    # 정규화
    tfs_norm = tfs / 10.0
    ons_norm = ons / 10.0
    
    # Gamma 승수 (0~1 → 0.5~1.5)
    gamma_multiplier = 0.5 + gamma
    
    # 잠식 계수 계산
    C = base_rate * (1 + ons_norm) * (1 - tfs_norm) * gamma_multiplier
    
    return np.clip(C, 0, 1)

# ===========================
# 3. Rb 예측 함수 (Rolling Prediction)
# ===========================
def predict_Rb_curve(initial_data, horizon=180):
    """
    Rb(t) 예측 (자연 수익 곡선)
    
    Parameters:
    - initial_data: 초기 7일 데이터 (DataFrame)
      컬럼: day_number, is_weekend, screen_cnt, aud_per_show, competition_index, social_buzz
    - horizon: 예측 기간 (일)
    
    Returns:
    - Rb_curve: Rb(t) 예측 배열 (numpy array)
    """
    # 초기 시퀀스 준비
    feature_cols = ['day_number', 'is_weekend', 'screen_cnt', 
                    'aud_per_show', 'competition_index', 'social_buzz']
    
    sequence = initial_data[feature_cols].values[-7:]  # 최근 7일
    sequence_scaled = scaler_X.transform(sequence)
    
    predictions = []
    current_seq = sequence_scaled.copy()
    
    # Rolling Prediction
    for day in range(8, horizon + 1):
        # 예측
        X_input = current_seq.reshape(1, 7, 6)
        y_pred_scaled = model_Rb.predict(X_input, verbose=0)[0, 0]
        y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        predictions.append(max(0, y_pred))
        
        # 다음 시퀀스 업데이트 (간소화: 마지막 값 복사)
        next_features = current_seq[-1].copy()
        next_features[0] = day  # day_number 업데이트
        
        # 시퀀스 슬라이딩
        current_seq = np.vstack([current_seq[1:], next_features])
    
    # 초기 7일 + 예측 결과
    initial_revenues = initial_data['daily_sales_amt'].values
    Rb_curve = np.concatenate([initial_revenues, predictions])
    
    return Rb_curve[:horizon]

# ===========================
# 4. Ra 시뮬레이션 함수
# ===========================
def simulate_Ra_curve(Rb_curve, cannib_coef, holdback_days):
    """
    Ra(t) 시뮬레이션 (잠식 수익 곡선)
    
    Parameters:
    - Rb_curve: Rb(t) 예측 배열
    - cannib_coef: 잠식 계수 C
    - holdback_days: 홀드백 기간
    
    Returns:
    - Ra_curve: Ra(t) 시뮬레이션 배열
    """
    Ra_curve = Rb_curve.copy()
    
    # 홀드백 이후 잠식 적용
    if holdback_days < len(Ra_curve):
        Ra_curve[holdback_days:] = Rb_curve[holdback_days:] * (1 - cannib_coef)
    
    return Ra_curve

# ===========================
# 5. 새로운 영화 예측 (실행 예제)
# ===========================

# 5-1. 영화 정보 입력
movie_info = {
    'title': '신작 영화',
    'genre': 'Action',
    'TFS': 8.4,  # 액션 장르 평균
    'ONS': 4.2,  # 액션 장르 평균
    'gamma': 0.892,  # 2024년 평균
    'open_year': 2024
}

# 5-2. 잠식 계수 계산
cannib_coef = calculate_cannibalization_coefficient(
    tfs=movie_info['TFS'],
    ons=movie_info['ONS'],
    gamma=movie_info['gamma']
)

print(f"\n영화: {movie_info['title']}")
print(f"장르: {movie_info['genre']}")
print(f"TFS: {movie_info['TFS']:.1f} | ONS: {movie_info['ONS']:.1f}")
print(f"Gamma: {movie_info['gamma']:.3f}")
print(f"잠식 계수(C): {cannib_coef:.3f} ({cannib_coef*100:.1f}%)")

# 5-3. 초기 7일 데이터 준비 (예시)
initial_data = pd.DataFrame({
    'day_number': [1, 2, 3, 4, 5, 6, 7],
    'is_weekend': [0, 0, 1, 1, 0, 0, 0],
    'screen_cnt': [1500, 1450, 1450, 1450, 1400, 1350, 1300],
    'aud_per_show': [120, 100, 150, 140, 80, 70, 60],
    'competition_index': [0.15, 0.15, 0.16, 0.16, 0.17, 0.17, 0.18],
    'social_buzz': [5000, 4500, 6000, 5500, 4000, 3500, 3000],
    'daily_sales_amt': [15e8, 12e8, 18e8, 17e8, 10e8, 8e8, 7e8]  # 실제 매출
})

# 5-4. Rb 예측
print("\n🔄 Rb(t) 예측 중...")
Rb_curve = predict_Rb_curve(initial_data, horizon=180)
print(f"✅ Rb 총 매출 예측: {np.sum(Rb_curve)/1e8:.1f}억 원")

# 5-5. Ra 시뮬레이션 (홀드백 30일)
holdback_days = 30
Ra_curve = simulate_Ra_curve(Rb_curve, cannib_coef, holdback_days)
print(f"✅ Ra 총 매출 (홀드백 {holdback_days}일): {np.sum(Ra_curve)/1e8:.1f}억 원")
print(f"   실제 잠식률: {(1 - np.sum(Ra_curve)/np.sum(Rb_curve))*100:.1f}%")

# 5-6. 극장 수익 계산
theater_revenue_before = np.sum(Rb_curve[:holdback_days])
theater_revenue_after = np.sum(Ra_curve[holdback_days:])
total_theater_revenue = theater_revenue_before + theater_revenue_after

print(f"\n💰 극장 수익 분석 (홀드백 {holdback_days}일):")
print(f"   홀드백 전 (1~{holdback_days}일): {theater_revenue_before/1e8:.1f}억")
print(f"   홀드백 후 ({holdback_days+1}~180일): {theater_revenue_after/1e8:.1f}억")
print(f"   총 극장 수익: {total_theater_revenue/1e8:.1f}억")
```

---

## 📌 시나리오 2: Part 3 시뮬레이션 연동

**상황**: Part 3에서 최적 홀드백 찾기 위해 Model 2 사용

### 코드 예제

```python
# ===========================
# Part 3용 통합 함수
# ===========================
def get_revenue_for_holdback(movie_data, initial_perf_data, holdback_days, horizon=180):
    """
    특정 홀드백 기간에 대한 극장 수익 계산
    
    Parameters:
    - movie_data: 영화 정보 dict (TFS, ONS, gamma 포함)
    - initial_perf_data: 초기 7일 실적 DataFrame
    - holdback_days: 홀드백 기간
    - horizon: 예측 기간
    
    Returns:
    - dict: {
        'Rb_total': Rb 총합,
        'Ra_total': Ra 총합,
        'theater_revenue': 극장 수익,
        'cannib_coef': 잠식 계수,
        'Rb_curve': Rb 배열,
        'Ra_curve': Ra 배열
      }
    """
    # 1. 잠식 계수 계산
    C = calculate_cannibalization_coefficient(
        tfs=movie_data['TFS'],
        ons=movie_data['ONS'],
        gamma=movie_data['gamma']
    )
    
    # 2. Rb 예측
    Rb = predict_Rb_curve(initial_perf_data, horizon)
    
    # 3. Ra 시뮬레이션
    Ra = simulate_Ra_curve(Rb, C, holdback_days)
    
    # 4. 극장 수익 계산
    theater_revenue = np.sum(Rb[:holdback_days]) + np.sum(Ra[holdback_days:])
    
    return {
        'Rb_total': np.sum(Rb),
        'Ra_total': np.sum(Ra),
        'theater_revenue': theater_revenue,
        'cannib_coef': C,
        'Rb_curve': Rb,
        'Ra_curve': Ra
    }

# ===========================
# Part 3 사용 예제
# ===========================

# 홀드백 시나리오별 극장 수익 계산
holdback_scenarios = [0, 30, 60, 90, 120, 150, 180]
results = []

for t in holdback_scenarios:
    result = get_revenue_for_holdback(
        movie_data=movie_info,
        initial_perf_data=initial_data,
        holdback_days=t,
        horizon=180
    )
    
    results.append({
        'holdback': t,
        'theater_revenue': result['theater_revenue'],
        'cannib_rate': (1 - result['Ra_total']/result['Rb_total']) * 100
    })
    
    print(f"홀드백 {t:3d}일: 극장수익 {result['theater_revenue']/1e8:6.1f}억 (잠식률 {results[-1]['cannib_rate']:.1f}%)")

# 최대 극장 수익 홀드백 찾기
best = max(results, key=lambda x: x['theater_revenue'])
print(f"\n✅ 최적 홀드백 (극장 수익 기준): {best['holdback']}일")
print(f"   극장 수익: {best['theater_revenue']/1e8:.1f}억")
```

---

## 📊 장르별 TFS/ONS 참고값

| 장르 | TFS | ONS | 설명 |
|------|-----|-----|------|
| **Action** | 8.4 | 4.2 | 극장 이벤트형, 스펙터클 |
| **SF** | 8.6 | 4.5 | 극장 이벤트형, 시각효과 |
| **Thriller** | 6.8 | 6.5 | 중간형 |
| **Drama** | 5.6 | 7.8 | OTT 친화형, 내러티브 |
| **Romance** | 5.2 | 8.1 | OTT 친화형, 감성 |
| **Horror** | 7.1 | 6.2 | 극장 선호 (분위기) |
| **Comedy** | 6.0 | 7.0 | 중간형 |

---

## 📊 연도별 Gamma 참고값

| 연도 | Gamma (정규화) | 설명 |
|------|---------------|------|
| **2019** | 0.653 | 코로나 이전 |
| **2020** | 0.745 | 코로나 초기 (OTT 급증) |
| **2021** | 0.798 | OTT 정착기 |
| **2022** | 0.856 | 극장 회복, OTT 유지 |
| **2023** | 0.827 | 혼합 소비 정착 |
| **2024** | 0.892 | OTT 우세 |

---

## ⚠️ 주의사항

### 1. 초기 데이터 품질
- **필수**: 최소 7일간의 실제 극장 데이터
- 정확도는 초기 데이터 품질에 의존
- 개봉 첫 주 특이사항(대형 명절 등) 고려 필요

### 2. 예측 정확도
- **신뢰 구간**: ±1억~2.5억 원 (MAE: 1.25억)
- **장기 예측**: 180일 예측 시 누적 오차 발생 가능
- **권장**: 14일마다 실측 데이터로 재예측

### 3. Gamma 적용
- **현재**: 연도별 전체 평균 사용
- **한계**: 영화별 타겟 연령대 미반영
- **해결**: 관람등급별 차등 적용 (향후)

### 4. 잠식 계수 파라미터
- **Base Rate**: 기본값 0.3 (30%)
  - Conservative: 0.15 (15%)
  - Aggressive: 0.50 (50%)
- 민감도 분석 권장

---

## 🔧 Troubleshooting

### Q1: 예측값이 음수로 나와요
```python
# 해결: 예측 결과에 max(0, pred) 적용
predictions.append(max(0, y_pred))
```

### Q2: Rolling Prediction이 너무 느려요
```python
# 해결: Batch 예측 또는 horizon 축소
Rb_curve = predict_Rb_curve(initial_data, horizon=90)  # 90일로 축소
```

### Q3: 초기 7일 데이터가 없어요
- **대안 1**: 유사 영화 데이터로 대체
- **대안 2**: 개봉 첫날 데이터를 7일로 복제 (단, 정확도 낮음)
- **권장**: 개봉 후 7일 대기 후 예측

---

## 📚 추가 자료

- **모델 구축 과정**: `../ipynb/main.ipynb` 참조
- **연구 방법론**: `../README.md` 참조
- **장르별 점수 계산**: OTT_suitability_score.csv 참조

---

**Last Updated**: 2024-11-19  
**Version**: 2.0
