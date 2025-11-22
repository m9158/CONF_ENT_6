📂 파일 개요

Daily_Performance.csv는 영화별로 일 단위 박스오피스 성과 지표를 담고 있는 테이블입니다.
이 데이터는 KOBIS(영화진흥위원회) 박스오피스 일별 데이터를 기반으로 생성되었습니다.

CSV의 각 행(row)은 특정 날짜의 특정 영화 성과 1건을 의미합니다.

🧱 컬럼 설명 (Column-by-Column Guide)

아래는 CSV 내부 컬럼들의 의미·출처·특징을 정리한 공식 가이드입니다.

### 1) performance_id

설명: 자동 증가 형태의 고유 ID (Primary Key)

용도: 테이블 내부에서 행을 고유하게 구분하기 위한 식별자

계산 방식: ETL 과정에서 1부터 순차적으로 생성

2) movie_id

설명: 영화 고유 코드 (movieCd)

출처: KOBIS 영화 고유 코드 사용

특징:

다른 테이블(movie_metadata, Online_Buzz_final, Google_Trends, actors)과의 조인 키

같은 movie_id는 같은 영화의 성과 기록

3) performance_date

설명: 해당 박스오피스 성과가 기록된 날짜 (YYYY-MM-DD)

출처: KOBIS 일별 박스오피스 API/데이터

특징:

개봉일 포함 전체 상영 기간 내 여러 날짜 존재

time-series 분석의 기준 축

4) daily_audi_cnt

설명: 일 관객 수

출처: KOBIS 박스오피스 데이터 (audiCnt)

특징:

해당 날짜에 실제 극장에서 영화를 관람한 총 관객 수

영화 트렌드/성공 지표 분석의 핵심 target 변수

5) daily_sales_amt

설명: 일 매출액(원)

출처: KOBIS 박스오피스 데이터 (salesAmt)

특징:

관객 수 × 평균 티켓 가격 등으로 형성되는 실 매출

극장 수입 및 영화 수익성 분석 시 활용

6) screen_cnt

설명: 해당 날짜의 스크린 수

출처: KOBIS 박스오피스 데이터 (scrnCnt)

특징:

상영관이 아닌 “스크린 개수”

영화 배급사의 스크린 배정 전략을 측정하는 변수

스크린 수 변화는 매출·관객 수 변동과 높은 상관 있음

7) show_cnt

설명: 해당 날짜의 상영 횟수

출처: KOBIS 박스오피스 데이터 (showCnt)

특징:

하루 동안 상영된 회차(타임슬롯)의 총 개수

공급량(공급 파라미터)을 보여주는 변수

상영 횟수는 관객 수 및 매출에 직접적 영향
