# 📘 Model 3 사용 가이드

## 📋 개요

이 문서는 **Model 3 (디지털 권리료 τ 추정 엔진)**을 실제로 사용하는 방법을 설명합니다.

---

## 🎯 Model 3의 역할

Model 3는 **OTT 플랫폼 판권료를 추정**하는 Heuristic 모델로, Part 3 시뮬레이션에서 다음을 제공합니다:

1. **τ(t) 추정**: 홀드백 기간에 따른 권리료
2. **플랫폼별 차별화**: MAU + 광고 비중(β) 반영
3. **장르 효과**: ONS에 따른 수명 주기

---

## 📦 필수 파일

```
model3/
├── main.ipynb                      # 함수 정의 (실행 필수)
├── mau_avg_by_year.csv             # 연도별 평균 MAU
├── platform_params_2024.csv        # 플랫폼 파라미터
├── movie_digital_fees.csv          # 영화별 권리료 (전체 시나리오) ⭐
└── movie_base_digital_fees.csv     # 영화별 기본 권리료 (Netflix 30일) ⭐
```

**⭐ 핵심 산출물**: 
- `movie_digital_fees.csv`: 모든 영화에 대해 플랫폼별, 홀드백별 권리료 계산 완료
- `movie_base_digital_fees.csv`: Part 3에서 바로 사용 가능한 기본 권리료 (Netflix, 30일)

---

## 🚀 사용 방법

### 환경 설정

```bash
pip install numpy pandas matplotlib
```

---

## 📌 시나리오 0: 미리 계산된 권리료 사용 (가장 간단)

**상황**: 함수 없이 CSV 파일만으로 권리료 조회

### 코드 예제

```python
import pandas as pd

# ===========================
# 1. 미리 계산된 권리료 로드
# ===========================

# 전체 시나리오 (영화 × 플랫폼 × 홀드백)
all_fees = pd.read_csv('movie_digital_fees.csv', encoding='utf-8-sig')

# 기본 권리료 (Netflix, 30일)
base_fees = pd.read_csv('movie_base_digital_fees.csv', encoding='utf-8-sig')

print(f"✅ 권리료 데이터 로드 완료")
print(f"   - 전체: {len(all_fees):,}개 레코드")
print(f"   - 기본: {len(base_fees):,}개 영화")

# ===========================
# 2. 특정 영화 권리료 조회
# ===========================

# 방법 1: 기본 권리료 (Netflix, 30일)
movie_title = '악마들'
fee_info = base_fees[base_fees['title'] == movie_title].iloc[0]

print(f"\n영화: {fee_info['title']}")
print(f"제작비: {fee_info['budget']/1e8:.1f}억 원")
print(f"권리료 (Netflix, 30일): {fee_info['digital_fee_billion_netflix_30d']:.2f}억 원")
print(f"제작비 대비: {fee_info['fee_ratio_netflix_30d']:.1f}%")

# 방법 2: 다양한 시나리오 조회
movie_scenarios = all_fees[all_fees['title'] == movie_title]

print(f"\n{movie_title}의 권리료 시나리오:")
print(movie_scenarios[['platform', 'holdback_days', 'digital_fee_billion', 'fee_ratio']])

# ===========================
# 3. 플랫폼별 비교 (특정 영화)
# ===========================

movie_30d = all_fees[
    (all_fees['title'] == movie_title) & 
    (all_fees['holdback_days'] == 30)
]

print(f"\n{movie_title} - 플랫폼별 권리료 (홀드백 30일):")
for _, row in movie_30d.iterrows():
    print(f"  {row['platform']:15}: {row['digital_fee_billion']:.2f}억")

# ===========================
# 4. 홀드백별 권리료 곡선 (특정 영화, Netflix)
# ===========================

movie_netflix = all_fees[
    (all_fees['title'] == movie_title) & 
    (all_fees['platform'] == 'Netflix')
].sort_values('holdback_days')

print(f"\n{movie_title} - Netflix 홀드백별 권리료:")
print(f"{'홀드백(일)':>12} | {'권리료(억)':>12}")
print("-" * 30)
for _, row in movie_netflix.iterrows():
    print(f"{row['holdback_days']:12d} | {row['digital_fee_billion']:12.2f}")
```

**출력 예시**:
```
✅ 권리료 데이터 로드 완료
   - 전체: 1,020개 레코드
   - 기본: 85개 영화

영화: 악마들
제작비: 2.0억 원
권리료 (Netflix, 30일): 0.19억 원
제작비 대비: 9.3%

악마들의 권리료 시나리오:
   platform  holdback_days  digital_fee_billion  fee_ratio
0   Netflix              0                 0.25       12.5
1   Netflix             30                 0.19        9.3
2   Netflix             60                 0.15        7.3
...

악마들 - 플랫폼별 권리료 (홀드백 30일):
  Netflix        : 0.19억
  Tving          : 0.13억
  Wavve          : 0.10억
```

---

## 📌 시나리오 1: 단일 권리료 추정 (함수 사용)

**상황**: 특정 영화의 OTT 권리료를 빠르게 추정하고 싶을 때

### 필요한 정보

1. **영화 정보**:
   - 제작비 (Budget)
   - 장르 → ONS 점수

2. **플랫폼 정보**:
   - 플랫폼 이름 (Netflix, Tving 등)
   - 연도 (2023, 2024, 2025)

3. **홀드백 시나리오**:
   - 예: 30일, 60일, 90일

### 코드 예제

```python
import numpy as np
import pandas as pd

# ===========================
# 1. 함수 및 데이터 로드
# ===========================

# main.ipynb 실행 후 함수 가져오기 (또는 아래 함수 직접 정의)

def calculate_digital_fee(
    budget,              # 제작비 (원)
    platform_mau,        # 플랫폼 MAU
    platform_beta,       # 광고 수익 비중 (0~1)
    ons,                 # OTT-Native Score (0~10)
    holdback_days,       # 홀드백 기간 (일)
    mau_avg,             # 연도별 평균 MAU
    r_base=0.10          # 기본 판권료 비율
):
    """디지털 권리료(τ) 추정"""
    
    # 1. 기본 가치
    base_value = budget * r_base
    
    # 2. 시장 지배력
    scale_factor = np.log(platform_mau) / np.log(mau_avg)
    
    # 3. ARPU 효율
    arpu_efficiency = 1.0 - (0.75 * platform_beta)
    
    # 4. 감가상각률
    ons_norm = ons / 10.0
    lambda_decay = 0.05 - (0.04 * platform_beta) - (0.01 * ons_norm)
    
    # 5. 시간 감가
    time_decay = np.exp(-lambda_decay * holdback_days)
    
    # 6. 최종 권리료
    tau = base_value * scale_factor * arpu_efficiency * time_decay
    
    return max(0, tau)

# 파라미터 데이터 로드
platform_params = pd.read_csv('platform_params_2024.csv')
mau_avg_data = pd.read_csv('mau_avg_by_year.csv')

print("✅ 함수 및 데이터 로드 완료")

# ===========================
# 2. 영화 정보 설정
# ===========================

movie_info = {
    'title': '범죄도시3',
    'budget': 15e9,      # 제작비 150억
    'genre': 'Action',
    'ons': 4.2,          # 액션 장르 ONS
    'year': 2024
}

print(f"\n영화: {movie_info['title']}")
print(f"제작비: {movie_info['budget']/1e8:.1f}억 원")
print(f"장르: {movie_info['genre']} (ONS: {movie_info['ons']})")

# ===========================
# 3. 플랫폼 파라미터 조회
# ===========================

platform_name = 'Netflix'
year = 2024

# 플랫폼 정보 가져오기
platform = platform_params[platform_params['platform_name'] == platform_name].iloc[0]
mau_avg = mau_avg_data[mau_avg_data['year'] == year].iloc[0]['mau_avg']

print(f"\n플랫폼: {platform_name}")
print(f"  MAU: {platform['platform_mau']:,}명")
print(f"  Beta: {platform['ads_rev_ratio']:.2f}")
print(f"  평균 MAU: {mau_avg:,.0f}명")

# ===========================
# 4. 홀드백별 권리료 계산
# ===========================

holdback_scenarios = [0, 30, 60, 90, 120, 150, 180]

print(f"\n💰 홀드백별 예상 권리료 ({platform_name}):")
print(f"{'홀드백(일)':>12} | {'권리료(억원)':>12} | {'제작비대비':>12}")
print("-" * 45)

for t in holdback_scenarios:
    tau = calculate_digital_fee(
        budget=movie_info['budget'],
        platform_mau=platform['platform_mau'],
        platform_beta=platform['ads_rev_ratio'],
        ons=movie_info['ons'],
        holdback_days=t,
        mau_avg=mau_avg
    )
    ratio = (tau / movie_info['budget']) * 100
    print(f"{t:12d} | {tau/1e8:12.2f} | {ratio:11.1f}%")

print(f"\n✅ 권리료 추정 완료")
```

**출력 예시**:
```
영화: 범죄도시3
제작비: 150.0억 원
장르: Action (ONS: 4.2)

플랫폼: Netflix
  MAU: 11,800,000명
  Beta: 0.05
  평균 MAU: 6,312,000명

💰 홀드백별 예상 권리료 (Netflix):
  홀드백(일) |  권리료(억원) |  제작비대비
---------------------------------------------
           0 |        18.75 |       12.5%
          30 |        14.70 |        9.8%
          60 |        11.54 |        7.7%
          90 |         9.06 |        6.0%
         120 |         7.11 |        4.7%
         150 |         5.58 |        3.7%
         180 |         4.38 |        2.9%
```

---

## 📌 시나리오 2: 플랫폼별 비교

**상황**: 여러 플랫폼 중 어디에 판매할지 권리료 비교

### 코드 예제

```python
# ===========================
# 플랫폼별 권리료 비교
# ===========================

platforms = ['Netflix', 'Tving', 'Wavve', 'Disney+', 'Coupang Play']
holdback_fixed = 30  # 30일 고정

print(f"\n플랫폼별 권리료 비교 (홀드백 {holdback_fixed}일)")
print(f"영화: {movie_info['title']} (제작비 {movie_info['budget']/1e8:.1f}억)\n")
print(f"{'플랫폼':15} | {'MAU':>12} | {'Beta':>6} | {'권리료(억)':>10}")
print("-" * 60)

results = []

for pf_name in platforms:
    pf = platform_params[platform_params['platform_name'] == pf_name].iloc[0]
    
    tau = calculate_digital_fee(
        budget=movie_info['budget'],
        platform_mau=pf['platform_mau'],
        platform_beta=pf['ads_rev_ratio'],
        ons=movie_info['ons'],
        holdback_days=holdback_fixed,
        mau_avg=mau_avg
    )
    
    results.append({
        'platform': pf_name,
        'mau': pf['platform_mau'],
        'beta': pf['ads_rev_ratio'],
        'tau': tau / 1e8
    })
    
    print(f"{pf_name:15} | {pf['platform_mau']:12,} | {pf['ads_rev_ratio']:6.2f} | {tau/1e8:10.2f}")

# 최고 권리료 플랫폼
best = max(results, key=lambda x: x['tau'])
print(f"\n✅ 최고 권리료: {best['platform']} ({best['tau']:.2f}억)")
```

**출력 예시**:
```
플랫폼별 권리료 비교 (홀드백 30일)
영화: 범죄도시3 (제작비 150.0억)

플랫폼           |          MAU |  Beta | 권리료(억)
------------------------------------------------------------
Netflix         |   11,800,000 |   0.05 |      14.70
Tving           |    7,050,000 |   0.26 |       9.15
Wavve           |    2,600,000 |   0.15 |       6.30
Disney+         |    1,810,000 |   0.04 |       5.70
Coupang Play    |    7,610,000 |   0.10 |      10.80

✅ 최고 권리료: Netflix (14.70억)
```

---

## 📌 시나리오 3: Part 3 통합 사용 (CSV 활용)

**상황**: Model 2(극장 수익) + Model 3(OTT 권리료)를 결합하여 최적 홀드백 찾기

### 방법 A: CSV 파일 직접 사용 (권장)

```python
import pandas as pd
import numpy as np

# ===========================
# 1. 데이터 로드
# ===========================

# Model 3: 권리료 데이터
digital_fees = pd.read_csv('model3/movie_digital_fees.csv', encoding='utf-8-sig')

# (Model 2가 생성한다고 가정)
# Model 2: 극장 수익 데이터 (예시)
# theater_revenues = pd.read_csv('model2/movie_theater_revenues.csv')

print("✅ 데이터 로드 완료")

# ===========================
# 2. 특정 영화 + 플랫폼 최적 홀드백 찾기
# ===========================

movie_title = '악마들'
platform = 'Netflix'

# 해당 영화 + 플랫폼의 권리료 데이터
movie_fees = digital_fees[
    (digital_fees['title'] == movie_title) & 
    (digital_fees['platform'] == platform)
].sort_values('holdback_days')

print(f"\n🎯 최적 홀드백 탐색: {movie_title} ({platform})")

results = []

for _, fee_row in movie_fees.iterrows():
    holdback = fee_row['holdback_days']
    tau = fee_row['digital_fee']
    
    # Model 2에서 극장 수익 가져오기 (실제 구현)
    # theater_rev = get_theater_revenue(movie_title, holdback)
    
    # 임시: 극장 수익 가정
    budget = fee_row['budget']
    cannibalization = 0.3 * (holdback / 180.0)
    theater_rev = budget * 3.0 * (1 - cannibalization)
    
    # 총수익
    total = theater_rev + tau
    
    results.append({
        'holdback': holdback,
        'theater': theater_rev / 1e8,
        'ott_fee': tau / 1e8,
        'total': total / 1e8
    })

results_df = pd.DataFrame(results)

print(f"\n{'홀드백(일)':>12} | {'극장(억)':>10} | {'OTT(억)':>10} | {'총수익(억)':>10}")
print("-" * 50)
for _, r in results_df.iterrows():
    print(f"{r['holdback']:12.0f} | {r['theater']:10.1f} | {r['ott_fee']:10.2f} | {r['total']:10.1f}")

# 최적 홀드백
optimal = results_df.loc[results_df['total'].idxmax()]
print(f"\n✅ 최적 홀드백: {optimal['holdback']:.0f}일")
print(f"   총수익: {optimal['total']:.1f}억")
```

---

### 방법 B: 함수 사용 (유연성 높음)

### 필요한 것

- Model 2의 `predict_revenue_curves()` 함수 (극장 수익 예측)
- Model 3의 `calculate_digital_fee()` 함수 (권리료 추정)

### 코드 예제

```python
# ===========================
# Part 3: 최적 홀드백 도출
# ===========================

# (가정) Model 2 함수가 로드되어 있다고 가정
# from model_2 import predict_revenue_curves

def find_optimal_holdback(
    movie_id,            # 영화 코드
    budget,              # 제작비
    ons,                 # ONS 점수
    platform_name,       # 플랫폼
    year,                # 연도
    holdback_range=range(0, 181, 10)  # 홀드백 시나리오
):
    """
    극장 수익 + OTT 권리료를 합산하여 최적 홀드백 찾기
    
    Returns:
        optimal_holdback: 최적 홀드백 기간
        max_profit: 최대 총수익
        results: 시나리오별 상세 결과
    """
    
    # 플랫폼 파라미터
    pf = platform_params[platform_params['platform_name'] == platform_name].iloc[0]
    mau_avg_val = mau_avg_data[mau_avg_data['year'] == year].iloc[0]['mau_avg']
    
    results = []
    
    for t in holdback_range:
        # Model 2: Rb/Ra 예측 (실제 구현 필요)
        # days, Rb, Ra, info = predict_revenue_curves(movie_id, t, 180)
        # theater_revenue = np.sum(Rb[:t]) + np.sum(Ra[t:])
        
        # 임시: 극장 수익 가정 (제작비의 3배 - 잠식 효과 고려)
        cannibalization = 0.3 * (t / 180.0)  # 시간에 비례한 잠식
        theater_revenue = budget * 3.0 * (1 - cannibalization)
        
        # Model 3: τ 계산
        tau = calculate_digital_fee(
            budget=budget,
            platform_mau=pf['platform_mau'],
            platform_beta=pf['ads_rev_ratio'],
            ons=ons,
            holdback_days=t,
            mau_avg=mau_avg_val
        )
        
        # 총수익
        total_profit = theater_revenue + tau
        
        results.append({
            'holdback': t,
            'theater_revenue': theater_revenue,
            'ott_fee': tau,
            'total_profit': total_profit
        })
    
    # 최적 홀드백
    optimal = max(results, key=lambda x: x['total_profit'])
    
    return optimal['holdback'], optimal['total_profit'], results


# 실행
print(f"\n🎯 최적 홀드백 탐색")
print(f"영화: {movie_info['title']}")
print(f"플랫폼: {platform_name}\n")

optimal_t, max_profit, all_results = find_optimal_holdback(
    movie_id='20124079',
    budget=movie_info['budget'],
    ons=movie_info['ons'],
    platform_name=platform_name,
    year=2024
)

print(f"{'홀드백(일)':>12} | {'극장수익(억)':>12} | {'OTT권리료(억)':>14} | {'총수익(억)':>10}")
print("-" * 60)

for r in all_results:
    print(f"{r['holdback']:12d} | {r['theater_revenue']/1e8:12.1f} | {r['ott_fee']/1e8:14.2f} | {r['total_profit']/1e8:10.1f}")

print(f"\n✅ 최적 홀드백: {optimal_t}일")
print(f"   최대 총수익: {max_profit/1e8:.1f}억")
```

**출력 예시**:
```
🎯 최적 홀드백 탐색
영화: 범죄도시3
플랫폼: Netflix

  홀드백(일) |  극장수익(억) |   OTT권리료(억) | 총수익(억)
------------------------------------------------------------
           0 |        450.0 |          18.75 |      468.8
          10 |        448.5 |          17.50 |      466.0
          20 |        447.0 |          16.30 |      463.3
          30 |        445.5 |          15.20 |      460.7
          40 |        444.0 |          14.18 |      458.2
          50 |        442.5 |          13.23 |      455.7
          60 |        441.0 |          12.34 |      453.3
          70 |        439.5 |          11.51 |      451.0
          80 |        438.0 |          10.74 |      448.7
          90 |        436.5 |          10.02 |      446.5
         ...

✅ 최적 홀드백: 0일
   최대 총수익: 468.8억
```

---

## 📊 장르별 ONS 참고표

| 장르 | ONS | 설명 |
|------|-----|------|
| **Action** | 4.2 | 극장 이벤트형, 빠른 감가 |
| **Sci-Fi** | 7.4 | 중간형 |
| **Thriller** | 9.6 | OTT 친화형, 느린 감가 |
| **Drama** | 9.4 | OTT 친화형, 느린 감가 |
| **Romance** | 7.6 | OTT 친화형 |
| **Horror** | 6.8 | 중간형 |
| **Comedy** | 5.8 | 중간형 |
| **Animation** | 6.8 | 중간형 |
| **Fantasy** | 8.4 | OTT 친화형 |
| **Crime** | 9.6 | OTT 친화형 |

---

## 📊 플랫폼별 파라미터 (2024년)

| 플랫폼 | MAU | Beta | 특징 |
|--------|-----|------|------|
| **Netflix** | 11.8M | 0.05 | 최대 규모, 구독형 |
| **Coupang Play** | 7.6M | 0.10 | 번들형, 구독 중심 |
| **Tving** | 7.1M | 0.26 | 광고형 혼합 |
| **Wavve** | 2.6M | 0.15 | 광고형 혼합 |
| **Disney+** | 1.8M | 0.04 | 구독형 |

---

## ⚠️ 주의사항

### 1. 모델의 한계

- **실제 데이터 부재**: OTT 판권료는 비공개 → 검증 불가
- **Heuristic 기반**: 문헌 및 통념 의존 → 절대값 정확도 제한
- **단순화 가정**: 계약 복잡성 미반영 (독점, 지역, 기간 등)

### 2. 활용 권장 사항

✅ **적합한 용도**:
- 플랫폼 간 **상대적 비교**
- 홀드백 기간에 따른 **경향성 분석**
- Part 3 시뮬레이션의 **파라미터 입력**

❌ **부적합한 용도**:
- 실제 계약 금액의 **정확한 예측**
- 법적·재무적 의사결정의 **유일한 근거**

### 3. R_BASE 민감도

기본값 `R_BASE = 0.10` (10%)은 조정 가능합니다:

```python
# Conservative (보수적): 5%
tau_conservative = calculate_digital_fee(..., r_base=0.05)

# Standard (표준): 10%
tau_standard = calculate_digital_fee(..., r_base=0.10)

# Aggressive (공격적): 15%
tau_aggressive = calculate_digital_fee(..., r_base=0.15)
```

---

## 🔧 Troubleshooting

### Q1: 권리료가 너무 낮게 나와요

**원인**: 
- 홀드백 기간이 길거나
- 플랫폼 MAU가 작거나
- 광고 비중(β)이 높을 때

**해결**:
- 홀드백을 짧게 설정
- MAU가 큰 플랫폼 선택
- R_BASE 상향 조정 (0.10 → 0.15)

### Q2: 플랫폼 데이터가 없어요

**해결**:
- 직접 파라미터 입력:
```python
tau = calculate_digital_fee(
    budget=100e8,
    platform_mau=10000000,  # 1,000만 MAU
    platform_beta=0.1,      # 10% 광고 비중
    ons=6.0,
    holdback_days=30,
    mau_avg=6312000         # 2024년 평균
)
```

### Q3: 장르별 ONS를 모르겠어요

**해결**:
- 위 "장르별 ONS 참고표" 사용
- 비슷한 장르의 평균값 적용
- 기본값: 6.0 (중간)

---

## 📚 추가 자료

- **모델 구축 과정**: `main.ipynb` 참조
- **이론적 배경**: `BUILD_PLAN.md` 참조
- **전체 개요**: `README.md` 참조

---

**Last Updated**: 2024-11-19  
**Version**: 1.0

