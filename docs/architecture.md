아래는 제공해주신 폴더 트리와 헥사고날 아키텍처 설계 내용을 바탕으로 작성한 프로젝트 아키텍처 문서 예시입니다. 이 문서는 docs/architecutre.md 파일에 저장할 수 있으며, 프로젝트의 각 계층과 그 역할, 그리고 전체 흐름을 이해하는 데 도움이 됩니다.

---

# 프로젝트 아키텍처 설계

이 문서는 **JSG-ML-API** 프로젝트의 아키텍처를 헥사고날 아키텍처 원칙에 따라 설계한 내용을 설명합니다. 각 계층은 명확한 책임과 역할을 가지며, 도메인 로직을 외부 시스템(데이터베이스, API, UI 등)으로부터 독립시키는 것을 목표로 합니다.

---

## 1. 폴더 구조

```
JSG-ML-API/
├── Dockerfile                        # 컨테이너 배포를 위한 Docker 설정
├── JSG_ML_API.egg-info               # 패키지 배포를 위한 메타데이터
├── Makefile                          # 빌드, 패키징, 배포 자동화 스크립트
├── README.md                         # 프로젝트 개요 및 사용법
├── application/                      # 애플리케이션 계층
│   ├── __init__.py
│   └── usecase/                      # 각 기능별 유스케이스
│       └── __init__.py
├── build/                            # 빌드 산출물 (자동 생성)
│   └── lib/                          # 배포 시 포함될 라이브러리 코드
│       └── …                         # 각 계층별 소스 코드 (자동 복사)
├── dist/                             # 생성된 wheel 파일 등 배포 산출물
├── docs/                             # 문서화 자료
│   └── architecutre.md                # (현재 문서)
├── domain/                           # 도메인 계층: 핵심 비즈니스 로직 및 모델
│   ├── __init__.py
│   ├── entities/                     # 비즈니스 엔티티
│   │   └── __init__.py
│   ├── repositories/                 # 도메인 리포지토리 인터페이스
│   │   └── __init__.py
│   ├── services/                     # 도메인 서비스
│   │   └── __init__.py
│   └── value_objects/                # 값 객체
│       └── __init__.py
├── infrastructure/                   # 인프라스트럭처 계층: 외부 시스템 연동
│   ├── __init__.py
│   └── adapters/                     # 어댑터 구현체
│       ├── __init__.py
│       └── repositories/             # 리포지토리 구현
│           └── __init__.py
├── interface/                        # 인터페이스 계층: 외부 진입점
│   ├── __init__.py
│   └── api/                        # RESTful API 엔드포인트
│       └── __init__.py
├── main.py                           # 애플리케이션 진입점
├── requirements.txt                  # 의존성 목록
├── setup.cfg                         # 여러 도구의 공통 설정
├── setup.py                          # 패키지 배포 및 설치 스크립트
└── tests/                            # 테스트 코드
    ├── __init__.py
    ├── e2e/                        # 엔드투엔드 테스트
    │   └── __init__.py
    ├── integration/               # 통합 테스트
    │   └── __init__.py
    └── unit/                      # 단위 테스트
        └── __init__.py
```

**설명:**

- **domain/**:  
  - **entities/**, **value_objects/**: 비즈니스 도메인의 핵심 모델 정의  
  - **services/**: 도메인 로직 구현  
  - **repositories/**: 도메인 데이터 저장소의 추상 인터페이스 정의

- **application/**:  
  - **usecase/**: 도메인 계층을 활용한 유스케이스 구현  
  - 애플리케이션 서비스나 포트 인터페이스(입력/출력)도 이 계층에서 관리할 수 있음

- **infrastructure/**:  
  - 실제 데이터베이스, 외부 API, 머신러닝 모델과 연동하는 어댑터 및 리포지토리 구현체  
  - 설정 파일이나 외부 라이브러리 연결 코드 포함

- **interface/**:  
  - API 엔드포인트를 통한 사용자 및 외부 시스템과의 통신  
  - CLI 등 다른 인터페이스도 이 계층에 포함 가능

- **tests/**:  
  - 단위, 통합, E2E 테스트 코드 구성

- **빌드/배포 관련 파일:**  
  - **Dockerfile**와 **Makefile**: 자동 빌드 및 배포 파이프라인 구현  
  - **setup.py, setup.cfg, requirements.txt**: 패키징 및 의존성 관리

---

## 2. 헥사고날 아키텍처 구성 요소

### 2.1 도메인 계층

- **목적:**  
  비즈니스의 핵심 모델과 로직을 구현하며, 외부 시스템의 변화에 영향받지 않도록 독립적으로 관리합니다.

- **구성:**  
  - **엔티티 및 값 객체:**
  - **도메인 서비스:**
  - **리포지토리 인터페이스:**

### 2.2 애플리케이션 계층

- **목적:**  
  도메인 계층을 활용하여 구체적인 비즈니스 프로세스(유스케이스)를 구현합니다.

- **구성:**  
  - **유스케이스:**   
  - **애플리케이션 서비스:**  
    - 도메인 로직 호출 및 외부와의 데이터 변환(입력/출력 DTO)
  - **포트 인터페이스:**  
    - 입력 포트와 출력 포트를 통해 외부와 도메인의 결합도를 낮춤

### 2.3 인프라스트럭처 계층

- **목적:**  
  도메인 및 애플리케이션 계층이 실제 외부 시스템(데이터베이스, 외부 API, 머신러닝 모델)과 상호작용할 수 있도록 연결합니다.

- **구성:**  
  - **리포지토리 구현:**
  - **어댑터:**  
    - 데이터베이스 어댑터, 외부 API 연동, 머신러닝 모델 어댑터 등

### 2.4 인터페이스 계층

- **목적:**  
  사용자 및 외부 시스템과의 통신을 담당하여, 애플리케이션에 대한 접근점을 제공합니다.

- **구성:**  
  - **API 엔드포인트:**  
    - `/api/v1/itinerary`  
  - **DTO:**  
    - 데이터 전송 객체를 통해 API 요청/응답 형식을 표준화합니다.

---

## 3. 주요 데이터 흐름 및 프로세스

### 3.1 여행 일정 생성
1. 사용자가 API를 통해 여행 일정 생성 요청  
2. API 컨트롤러가 요청 데이터를 DTO로 변환  
3. 애플리케이션 서비스가 `GenerateItineraryUseCase`를 호출  
4. 유스케이스는 도메인 서비스와 리포지토리 인터페이스를 통해 비즈니스 로직 실행  
5. 결과를 DTO로 변환하여 응답 반환

### 3.2 주변 관광지 추천
1. 사용자가 API를 통해 추천 요청  
2. API 컨트롤러가 요청 데이터를 DTO로 변환  
3. 애플리케이션 서비스가 `RecommendNearbySpotUseCase`를 호출  
4. 유스케이스가 도메인 서비스와 리포지토리, 머신러닝 어댑터를 통해 추천 생성  
5. 결과를 DTO로 변환하여 응답 반환

---

## 4. 빌드 및 배포

- **Dockerfile:**  
  컨테이너 이미지를 생성하여, 일관된 배포 환경을 제공합니다.

- **Makefile:**  
  Wheel 파일 생성, Docker 이미지 빌드, 컨테이너 배포 등의 작업을 자동화합니다.

- **패키징:**  
  `setup.py` 및 `setup.cfg`를 통해 패키지 배포와 의존성 관리를 수행합니다.

---

## 5. 테스트 전략

- **테스트 폴더:**  
  - **unit/**: 각 계층의 단위 테스트  
  - **integration/**: 계층 간 통합 테스트  
  - **e2e/**: 전체 시스템의 엔드투엔드 테스트

- **자동화:**  
  CI/CD 파이프라인과 연동하여, 커밋 및 PR 단계에서 자동 테스트를 수행합니다.