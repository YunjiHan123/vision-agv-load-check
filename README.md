# Vision AGV Load Check

> Vision-based AGV autonomous navigation and load inspection system  
> AGV가 적재 위치와 목적지를 비전 기반으로 인식하고, OCR과 객체 탐지를 통해 적재 상태와 LOT 정보를 확인하는 Python 기반 스마트팩토리 비전 프로젝트입니다.

<br/>

## 1. 프로젝트 소개

Vision AGV Load Check는 AGV의 자율 주행 및 적재 검사를 지원하기 위한 비전 기반 시스템입니다.

카메라 이미지를 분석하여 적재물 상태를 확인하고, YOLO 기반 객체 탐지와 OCR을 활용해 LOT 정보를 인식합니다.  
또한 ROS2 AGV 시스템과 연동할 수 있도록 FastAPI 기반 통신 서버를 구성하여, 외부 AGV 노드가 명령을 조회하거나 분석 요청을 보낼 수 있도록 구현했습니다.

<br/>

## 2. 주요 기능

- YOLO 기반 객체 탐지
- mini-box 학습 및 라벨링 데이터 구성
- PaddleOCR 기반 LOT 코드 인식
- OCR crop 보정 및 회전/deskew 처리
- 재고 및 적재 상태 판단
- 테스트 데이터 기반 end-to-end 평가
- FastAPI 기반 ROS2 AGV 연동 서버
- `/health`, `/command`, `/analyze` API 제공
- 간단한 UI 화면 뼈대 구성

<br/>

## 3. 기술 스택

| 영역 | 기술 |
|---|---|
| Language | Python |
| AI / Vision | YOLO, Object Detection |
| OCR | PaddleOCR, EasyOCR fallback |
| Backend Server | FastAPI |
| Test | pytest |
| Integration | ROS2 AGV 연동 |
| UI | Python 기반 UI 화면 |

<br/>

## 4. 프로젝트 구조

```text
vision-agv-load-check
├── configs              # OCR 및 파이프라인 설정 파일
├── data                 # 샘플 데이터 및 테스트 데이터
├── docs                 # 문서 자료
├── models               # 학습 모델 및 관련 파일
├── scripts              # 실행 보조 스크립트
├── src                  # 핵심 소스 코드
│   ├── pipeline          # 비전 파이프라인
│   ├── ocr               # LOT OCR 인식
│   └── server            # FastAPI 통신 서버
├── tests                # 테스트 코드
├── main.py              # 비전 파이프라인 실행 진입점
├── server_main.py       # ROS2 AGV 연동 서버 실행 진입점
└── ui_main.py           # UI 실행 파일
````

<br/>

## 5. 팀원 역할

| 이름 | 담당 영역                               | 주요 기여                                                                                                                                                                                         |
| ------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 한윤지               | Vision Pipeline / YOLO / Test / UI  | 프로젝트 초기 폴더 구조를 생성하고, 테스트 목업 데이터와 테스트 코드를 구성했습니다. 쓰러짐/정상 분류 로직, YOLO mini-box 학습 전 라벨링, 테스트 데이터 수정, OCR과 재고 파악 로직을 결합한 end-to-end 파이프라인 구현에 기여했습니다. 또한 최종적으로 UI 뼈대 작업과 develop 브랜치 통합을 담당했습니다. |
| 홍원희                | OCR / PaddleOCR / Normalization     | PaddleOCR 모델을 구성하고 OCR 결과의 normalize 보정 작업을 담당했습니다. LOT 코드 인식 정확도를 높이기 위해 OCR 처리 흐름과 보정 로직을 개선했습니다.                                                                                           |
| 조혜원               | FastAPI Server / ROS2 Communication | `src/server`에 서버 프로그램을 구축하여 ROS2 AGV와 연동 가능한 통신 구조를 구현했습니다. `/health`, `/command`, `/analyze`와 같은 API 기반으로 AGV가 명령을 조회하고 분석 결과를 주고받을 수 있는 서버 구조를 담당했습니다.                                      |

<br/>

## 6. 시스템 흐름

```text
[Camera / Image Input]
        |
        v
[Vision Pipeline]
- YOLO 객체 탐지
- mini-box 검출
- 적재 상태 확인
        |
        v
[OCR Module]
- LOT 영역 crop
- deskew / rotation 보정
- PaddleOCR 기반 LOT 코드 인식
        |
        v
[Load Check Result]
- 재고 정보 판단
- 적재 상태 판단
- 결과 반환
        |
        v
[FastAPI Server]
- ROS2 AGV 명령 연동
- 분석 요청 처리
- 결과 응답
```

<br/>

## 7. 프로젝트를 통해 배운 점

이 프로젝트를 통해 스마트팩토리 환경에서 비전 AI가 단순히 객체를 탐지하는 것에 그치지 않고, AGV의 이동과 적재 검사 흐름에 연결되어야 한다는 점을 배웠습니다.

특히 OCR은 이미지 품질, crop 영역, 회전, 기울어짐, 문자 보정에 따라 결과가 크게 달라졌기 때문에, PaddleOCR 적용뿐 아니라 crop padding, deskew, rotation search, normalize 보정이 중요하다는 것을 경험했습니다.

또한 YOLO 탐지 결과와 OCR 결과를 결합해 end-to-end로 재고 판단까지 이어가면서, AI 모델 하나보다 전체 파이프라인의 안정성이 더 중요하다는 것을 배웠습니다.

<br/>

## 8. 한 줄 회고

Vision AGV Load Check는 AGV의 자율 주행 및 적재 검사 과정에 비전 AI와 OCR을 결합하여, 스마트팩토리 물류 흐름을 자동화하기 위한 AI 비전 시스템입니다.
