# Domain Entity 요구사항 명세

## 1. Domain 엔티티 요구사항
### 1.1 TouristSpot 엔티티

**목적:**  
관광지 정보를 관리하며, 사용자가 여행 일정을 구성할 때 관광지의 기본 정보(위치, 이름, 카테고리 등)를 제공합니다.

**주요 속성:**  
- **id (int):** 관광지 고유 식별자  
- **name (str):** 관광지 이름  
- **latitude (float):** 위도 (기본값: 0.0)  
- **longitude (float):** 경도 (기본값: 0.0)  
- **activity_level (float):** 관광지의 활동성 지표 (기본값: 0.0)  
- **address (Optional[str]):** 주소 정보  
- **business_hours (Optional[str]):** 영업시간 등 추가 설명  
- **categories (List[str]):** 관광지 분류 (예: 박물관, 공원 등)  
- **home_page (Optional[str]), booking_url (Optional[str]), tel (Optional[str]):** 추가 연락처 및 예약 정보  
- **thumbnail_url (Optional[str]), thumbnail_urls (List[str]):** 썸네일 이미지 URL  
- **travel_city_id (Optional[int]):** 소속 도시 식별자  
- **average_visit_duration (float):** 평균 방문 체류 시간 (기본값: 1.0)  
- **opening_hours (Optional[str]):** "HH:MM-HH:MM" 형식의 개장 시간 문자열  
- **opening_time (Optional[time]), closing_time (Optional[time]):** 파싱된 개장/폐장 시간

**주요 기능:**  
- **타입 변환 및 유효성 검사:**  
  - __post_init__ 메서드에서 문자열 입력을 적절한 숫자나 시간 객체로 변환  
  - 잘못된 입력 값에 대해 기본값 또는 None 처리
- **영업시간 검사:**  
  - `is_open_at(check_time: time) -> bool`: 주어진 시간이 개장 시간 범위 내에 있는지 판단
- **동등성 비교:**  
  - __eq__ 메서드를 통해 두 관광지 객체는 id가 동일하면 동일한 객체로 판단
- **문자열 표현:**  
  - __str__ 메서드로 관광지의 id와 name을 포함한 문자열 반환

---

### 1.2 Itinerary 엔티티

**목적:**  
전체 여행 일정 정보를 관리하며, 최적화 로직과 사용자에게 보여질 일정표를 생성하는 역할을 담당합니다.

**주요 속성:**  
- **daily_routes (List[DayRoute]):** 최적화 로직을 위해 사용되는 일자별 경로 정보  
- **days (List[ItineraryDay]):** 사용자에게 보여질 일자별 실제 일정표  
- **overall_distance (float):** 전체 이동 거리 (km)  
- **created_at (datetime), updated_at (datetime):** 생성 및 수정 시각  
- **start_date (Optional[date]), end_date (Optional[date]):** 여행 시작 및 종료 날짜  
- **num_days (int):** 총 여행 일수 (daily_routes와 days 중 큰 값으로 결정)

**주요 기능:**  
- __post_init__에서 입력 데이터가 dict일 경우 객체로 자동 변환  
- **일정 조회:**  
  - `get_all_tourist_spots_from_routes() -> List[int]`: daily_routes에 포함된 모든 관광지 ID를 중복 없이 반환  
  - `get_day_route(day: int) -> Optional[DayRoute]`: 특정 일차의 경로 정보 반환  
- **통계 계산:**  
  - `calculate_stats_from_routes() -> Dict[str, Any]`: daily_routes를 기반으로 총 일수, 총 거리, 평균 관광지 수 등 통계 정보 산출  
  - `calculate_stats_from_days() -> Dict[str, Any]`: 실제 일정표(days)를 기준으로 통계 정보 산출


## 2. 일반 요구사항

- **유효성 검사 및 타입 변환:**  
  모든 도메인 엔티티 및 값 객체는 __post_init__ 메서드를 통해 입력 값의 타입 변환, 유효성 검사, 예외 처리 등을 수행해야 합니다.
  
- **동등성 및 불변성:**  
  도메인 엔티티는 일반적으로 고유 식별자(id)를 기준으로 동등성을 판단하고, 값 객체는 불변성(Immutable)을 유지하여 동일한 값은 동일한 객체로 인식됩니다.

- **문자열 및 출력 표현:**  
  __str__ 및 __repr__ 메서드를 구현하여, 로깅, 디버깅, 사용자 메시지 출력 시 유용하게 활용할 수 있어야 합니다.

- **확장성:**  
  향후 도메인 로직이 확장될 수 있으므로, 엔티티와 값 객체의 설계는 OOP 원칙 및 DDD의 전략적 설계를 반영하여, 각 요소가 독립적이며 확장 가능한 구조로 만들어져야 합니다.

---

## 3. 문서화 및 테스트 전략

- **문서화:**  
  각 엔티티와 값 객체의 역할, 속성, 메서드에 대해 상세하게 주석과 문서화를 진행합니다.  
  
- **TDD 적용:**  
  기능 구현 전, 각 도메인 엔티티 및 값 객체의 요구사항에 따라 테스트 케이스를 작성하고, 이를 기반으로 최소 기능을 구현한 후, 리팩토링하는 TDD 사이클을 적용합니다.

- **자동화된 테스트:**  
  pytest 등을 사용하여 모든 도메인 모델의 핵심 기능, 예외 처리, 통계 계산 등이 올바르게 동작하는지 검증하는 단위 테스트를 작성합니다.
