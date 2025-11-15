# OTT 장르별 적합성 지수 분석

이 문서는 `OTT_suitability_score.csv` 파일의 각 장르별 점수를 `OTT Suitability Index Per Genre.pdf` 문서에 기반하여 설명합니다.

## 점수 항목 설명

PDF 문서에 따르면 각 점수 항목은 다음과 같은 의미를 가집니다.

*   **ES (External Spectacle) Score (외부적 스펙터클 점수):** 시청각적 규모, 즉 CGI, 거대한 세트, 몰입감 있는 사운드 등 감각적 몰입을 유도하는 요소를 평가합니다. (극장 경험의 핵심 요소)
*   **IS (Internal Spectacle) Score (내부적 스펙터클 점수):** 서사적 또는 연기적 강렬함, 즉 명배우의 연기, 대담한 반전, 유명 감독/제작사의 브랜드 파워 등을 평가합니다. (극장 경험의 또 다른 핵심 요소)
*   **CV (Collective Value) Score (집단적 가치 점수):** 공유된 관객 반응(웃음, 탄성 등)에서 오는 즐거움을 평가합니다.
*   **NC (Narrative Complexity) Score (서사 복잡성 점수):** 여러 개의 하위 플롯, 복잡한 캐릭터 관계망 등 시청자가 자신의 속도에 맞춰 소비할 때 더 효과적인 서사 구조를 가졌는지 평가합니다. (OTT 환경에 적합한 요소)
*   **TI (Thematic Intimacy) Score (주제적 친밀성 점수):** '혼자 보는 시청자'에게 더 깊은 감정적 반응("감동", "몰입")을 이끌어내는 심리적 힘을 평가합니다. (OTT 환경에 적합한 요소)
*   **TFS (Theatrical-First Score - 극장 우선 점수):** 극장 개봉의 적합성을 나타내는 가중 점수입니다. `(ES * 0.5) + (IS * 0.3) + (CV * 0.2)`
*   **ONS (OTT-Native Score - OTT 네이티브 점수):** OTT 플랫폼 적합성을 나타내는 가중 점수입니다. `(NC * 0.6) + (TI * 0.4)`

---

## 장르별 점수 해설

### 액션 (Action)
*   **ES Score:** 10
*   **IS Score:** 6
*   **CV Score:** 8
*   **NC Score:** 5
*   **TI Score:** 3
*   **TFS:** 8.4
*   **ONS:** 4.2
*   **Strategic Verdict:** Theatrical-First (극장 우선)
*   **설명:** 액션 장르는 특수 효과, 몰입감 있는 오디오, 빠른 편집 등 '외부적 스펙터클'의 전형으로, 극장 경험에 최적화되어 있습니다. 박스오피스의 주요 동력이며, 극장 개봉을 통해 형성된 브랜드 가치가 스트리밍으로 이어지는 효과가 큽니다.

### Sci-Fi (공상 과학)
*   **ES Score:** 10
*   **IS Score:** 8
*   **CV Score:** 8
*   **NC Score:** 9
*   **TI Score:** 5
*   **TFS:** 9.0
*   **ONS:** 7.4
*   **Strategic Verdict:** True Hybrid (Format-Dependent) (진정한 하이브리드 - 포맷 의존적)
*   **설명:** Sci-Fi는 높은 '외부적 스펙터클'과 함께, 복잡한 세계관과 서사를 통해 '서사 복잡성' 점수도 높게 나타납니다. 거대한 IP(스타워즈, 듄 등) 기반의 영화는 극장에 적합하지만, 복잡한 세계관을 다루는 오리지널 IP는 OTT 시리즈에 더 적합할 수 있습니다.

### 판타지 (Fantasy)
*   **ES Score:** 9
*   **IS Score:** 8
*   **CV Score:** 7
*   **NC Score:** 10
*   **TI Score:** 6
*   **TFS:** 8.3
*   **ONS:** 8.4
*   **Strategic Verdict:** True Hybrid (Format-Dependent) (진정한 하이브리드 - 포맷 의존적)
*   **설명:** 판타지는 '서사 복잡성'과 '세계관 밀도'에서 가장 높은 점수를 받는 장르입니다. 방대한 판타지 세계관은 2시간짜리 영화보다 긴 호흡의 OTT 시리즈에 더 적합할 수 있습니다. 따라서 작품의 형태에 따라 극장과 OTT 모두에서 성공 가능성이 있습니다.

### 드라마 (Drama)
*   **ES Score:** 2
*   **IS Score:** 7
*   **CV Score:** 3
*   **NC Score:** 9
*   **TI Score:** 10
*   **TFS:** 4.7
*   **ONS:** 9.4
*   **Strategic Verdict:** OTT-Native (OTT 네이티브)
*   **설명:** 드라마는 '외부적 스펙터클'이 낮은 대신, '주제적 친밀성'과 '서사 복잡성'에서 압도적으로 높은 점수를 받습니다. 이는 혼자 몰입하여 복잡한 캐릭터 관계와 감정선을 따라가는 OTT 환경에 가장 최적화된 장르라는 것을 의미합니다. 스트리밍 수요 1위 장르입니다.

### 호러 (Horror)
*   **ES Score:** 5
*   **IS Score:** 9
*   **CV Score:** 10
*   **NC Score:** 6
*   **TI Score:** 8
*   **TFS:** 7.2
*   **ONS:** 6.8
*   **Strategic Verdict:** True Hybrid (진정한 하이브리드)
*   **설명:** 호러는 독특한 경제 모델을 가진 '극장 이벤트' 장르입니다. 집단적 비명과 긴장감을 극대화하는 '집단적 가치'가 매우 높습니다. 동시에, 심리적 공포와 서스펜스는 '혼자 보는 시청자' 환경에서 증폭되므로 OTT에서도 강점을 보입니다. 가장 재무적으로 유연한 장르입니다.

### 스릴러 (Thriller)
*   **ES Score:** 3
*   **IS Score:** 8
*   **CV Score:** 5
*   **NC Score:** 10
*   **TI Score:** 9
*   **TFS:** 4.9
*   **ONS:** 9.6
*   **Strategic Verdict:** OTT-Native (OTT 네이티브)
*   **설명:** 스릴러는 구조적으로 OTT의 '플렉시-서사(flexi-narrative)' 모델에 가장 완벽하게 부합하는 장르입니다. 서스펜스, 반전, 심리적 깊이, 클리프행어 등은 시청자의 몰아보기와 리텐션을 유도하는 핵심적인 장치들입니다.

### 코미디 (Comedy)
*   **ES Score:** 2
*   **IS Score:** 5
*   **CV Score:** 7
*   **NC Score:** 5
*   **TI Score:** 7
*   **TFS:** 4.4
*   **ONS:** 5.8
*   **Strategic Verdict:** OTT-Native (OTT 네이티브)
*   **설명:** 과거 집단적 웃음을 기반으로 극장 시장에서 강세를 보였으나, 현재는 그 시장이 붕괴되었습니다. 스트리밍 수요 역시 감소 추세에 있지만, '주제적 친밀성'에 의존하는 저비용 '컴포트 푸드' 콘텐츠로서 OTT 리텐션에 기여합니다.

### 로맨스 (Romance)
*   **ES Score:** 1
*   **IS Score:** 6
*   **CV Score:** 4
*   **NC Score:** 6
*   **TI Score:** 10
*   **TFS:** 3.8
*   **ONS:** 7.6
*   **Strategic Verdict:** OTT-Native (OTT 네이티브)
*   **설명:** 극장 시장 점유율은 미미하지만, '주제적 친밀성'에서 만점을 기록하며 OTT 네이티브 장르로 자리 잡았습니다. 개인적이고 친밀한 감정 교감이 중요하므로 '혼자 보는 시청자' 환경에서 그 가치가 극대화됩니다.

### 다큐멘터리 (Documentary)
*   **ES Score:** 1
*   **IS Score:** 6
*   **CV Score:** 2
*   **NC Score:** 7
*   **TI Score:** 8
*   **TFS:** 2.7
*   **ONS:** 7.4
*   **Strategic Verdict:** OTT-Dominant (OTT 지배적)
*   **설명:** 극장 개봉은 거의 명망 있는 시상식 출품 자격용으로만 이루어집니다. 스트리밍 플랫폼의 부상과 함께 가장 빠르게 성장한 장르 중 하나로, OTT가 제작 및 배급을 주도하는 'OTT 지배적' 장르입니다.
