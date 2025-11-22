# 📘 Online_Buzz_filtered_v1.csv — Column Documentation

*네이버 데이터랩 기반 영화별 온라인 버즈(검색량 지수) 데이터셋*

---

## 📂 파일 개요

`Online_Buzz_filtered_v1.csv`는 영화별로 **네이버 검색 데이터(데이터랩 API)** 에서 수집한

**일별 검색량 지수(0–100)** 를 정리한 테이블입니다.

- 각 행(row)은 “특정 영화의 특정 날짜 검색량”을 의미합니다.
- `movieCd` 를 기준으로 다른 테이블(메타데이터, 박스오피스 성과 등)과 조인됩니다.

---

## 🧱 Column-by-Column 설명

### ### 1) `buzz_id`

- **설명:** 버즈 데이터의 고유 식별자
- **구성:** `"movieCd_buzz_date"` 형태의 문자열
    
    예) `20190300_2024-12-31`
    
- **용도:** 테이블 내 레코드를 고유하게 구분하기 위한 Primary Key 역할
- **생성 방식:** `f"{movieCd}_{buzz_date}"` 형태로 자동 생성

---

### ### 2) `movieCd`

- **설명:** 영화 고유 코드 (KOBIS movieCd)
- **출처:** KOBIS 영화 목록 또는 movie_metadata.csv
- **특징:**
    - 다른 모든 데이터셋(`Daily_Performance`, `movie_metadata`, Google Trends 등)과의 **조인 키**
    - 동일 movieCd는 동일 영화

---

### ### 3) `buzz_date`

- **설명:** 해당 검색량 지수가 측정된 날짜 (YYYY-MM-DD)
- **출처:** 네이버 데이터랩 요청 기간(`startDate ~ endDate`)
- **특징:**
    - 개봉일 이전 ~ 이후 특정 기간의 버즈 흐름을 나타냄
    - time-series 형태 분석 가능
    - Online_Buzz_final, Daily_Performance 등과 날짜 기준 병합 가능

---

### ### 4) `search_buzz_vol`

- **설명:** **네이버 데이터랩 검색량 지수(0–100)**
- **출처:**
    - Naver DataLab Search API
    - `"ratio"` 값 사용
- **의미:**
    - 해당 기간 내에서 ‘가장 검색량이 높은 날’을 100으로 두고
        
        나머지를 상대값으로 환산한 비율
        
- **주의점:**
    - *영화 내부에서만 상대 스케일임* → 영화 간 검색량 절대 비교 불가
    - 저검색량 영화일수록 값이 과장되는 구조적 문제가 있음
    - 모델에 넣을 때는 정규화·평균·slope 등 파생변수 생성 필요

---

##
