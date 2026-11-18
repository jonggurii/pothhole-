# YOLOv8과 RT-DETR 기반 포트홀 탐지 모델의 성능 비교 및 외부 도로 환경 일반화 분석

## 1. 개발동기 및 목적

포트홀은 도로 표면의 파손으로 발생하며 차량 손상 및 주행 안전성 저하를 유발할 수 있다. 기존의 인력 중심 도로 점검을 보완하기 위해 컴퓨터 비전 기반의 자동 포트홀 탐지 기술이 필요하다.

그러나 실제 도로 영상에는 조명 변화, 그림자, 유사 노면 패턴 및 작은 포트홀 등 다양한 탐지 장애 요인이 존재한다. 또한 탐지율을 높이기 위해 Confidence Threshold를 완화하면 오탐이 증가하는 문제가 발생할 수 있다.

본 프로젝트는 YOLOv8 기반 포트홀 탐지 모델을 구축하고 CNN 재판별 구조와 Transformer 기반 RT-DETR을 비교하여, 탐지 정확도와 오탐 부담 사이의 관계를 분석하는 것을 목적으로 한다. 나아가 새로운 한국 도로 이미지에서 기존 모델의 일반화 성능과 주요 실패 유형을 검증한다.

## 2. 개발내용

**1단계: 데이터 구축 및 YOLOv8 기준 모델 개발 (Week 1~3)**

공개 도로 손상 이미지와 annotation을 수집하고, XML 좌표를 YOLO 형식으로 변환하였다. 라벨 검수와 클래스 통합을 수행하고, YOLOv8n/s/m 모델을 학습하였다. 날씨 증강, 입력 해상도, TTA 및 Confidence Threshold에 따른 Precision–Recall 변화를 분석하였다.

**2단계: YOLOv8+CNN 하이브리드 탐지 구조 구현 (Week 3~5)**

YOLOv8에서 생성한 후보 박스를 CNN으로 재분류하는 2-Stage Cascade를 구성하였다. GT와 후보 박스의 IoU 기반 라벨링을 적용하고, YOLO 단독 모델과 CNN 결합 모델을 동일한 5-fold 평가 조건에서 비교하였다. CNN 결합은 Recall을 소폭 높였으나, 오탐과 추론 지연시간도 증가하였다.

**3단계: RT-DETR 학습 및 모델 성능 비교 (Week 6~8)**

Transformer 기반 RT-DETR-L의 학습 환경을 구축하고, YOLOv8s와 동일한 598장 데이터 및 5-fold 분할을 사용하였다. 공통 평가 기준으로 mAP, Precision, Recall, F1을 측정하고, FP/image 목표 및 입력 해상도에 따른 탐지 성능을 추가 분석하였다.

**4단계: 외부 한국 도로 데이터 일반화 검증 (Week 9~10)**

기존 모델 가중치와 Confidence Threshold를 변경하지 않고 새로운 한국 도로 이미지 745장, 포트홀 803개를 대상으로 외부 평가를 수행하였다. 이미지·JSON 연결 검증, Bounding Box 무결성 검사, Checkpoint 해시 검증을 적용하여 평가 재현성을 확보하였다. 이후 크기별 Recall, 오탐, 미검출 원인 및 모델 간 탐지 결과의 상호보완성을 분석하였다.

## 3. 주요기술

- **YOLOv8s:** CNN Backbone과 다중 스케일 특징 융합을 활용한 포트홀 객체 탐지 모델
- **YOLOv8+CropCNN:** YOLO 후보 영역의 CNN 분류 확률을 이용한 2단계 점수 재조정 구조
- **RT-DETR-L:** CNN Backbone, Hybrid Encoder, Transformer Decoder를 활용한 객체 탐지 모델
- **5-Fold Cross Validation:** 동일한 데이터 분할 및 독립적인 outer test를 이용한 반복 성능 평가
- **Confidence Threshold 및 FP/image 분석:** Validation에서 설정한 오탐 목표에 따라 Threshold를 선택하고 실제 test 성능 평가
- **COCO Evaluation:** AP50, AP50–95 및 객체 크기별 AP 평가
- **Failure Analysis:** IoU, Confidence, 객체 크기 및 모델별 예측 관계를 이용한 TP·FP·FN 분석
- **Python / PyTorch / Ultralytics / OpenCV:** 학습, 추론, 데이터 처리 및 평가 자동화

## 4. 결과 및 분석

**내부 5-fold 성능 비교**

| 지표 | YOLOv8s | RT-DETR |
|---|---:|---:|
| mAP50–95 | 45.17% | **53.34%** |
| mAP50 | 72.89% | **80.44%** |
| Recall | 64.77% | **74.63%** |
| F1 | 70.75% | **76.51%** |

RT-DETR은 내부 평가에서 YOLOv8s보다 높은 객체 탐지 성능을 보였다. mAP50–95는 약 8.17%p, Recall은 약 9.86%p 높았다.

**외부 데이터 성능 비교**

| 지표 | YOLOv8s | RT-DETR |
|---|---:|---:|
| AP50 | 49.07% | **58.99%** |
| AP50–95 | 24.20% | **32.48%** |
| Recall (FP 목표 0.2) | 37.41% | **52.73%** |
| 실제 FP/image (FP 목표 0.2) | **0.182** | 0.314 |

외부 도로 이미지에서도 RT-DETR의 AP 및 Recall 우위가 관찰되었지만, FP/image가 더 높아 오탐 부담도 함께 증가하였다.

또한 외부 데이터의 상대 크기 하위 25% 포트홀에서 Recall은 YOLOv8s 27.76%, RT-DETR 43.58%로, 작은 포트홀 탐지는 두 모델 모두에게 주요한 한계로 확인되었다.

미검출 분석 결과, FP 목표 0.2에서 Confidence Threshold 미달 유형은 YOLOv8s FN의 77.2%, RT-DETR FN의 87.0%를 차지하였다.

## 5. 활용방안 및 기대효과

**도로 손상 자동 점검 지원**

차량 카메라 및 도로 영상에서 포트홀 후보를 자동 탐지하여 도로 유지보수 업무의 효율성을 높이는 데 활용할 수 있다.

**도로 안전 관리 데이터 확보**

탐지한 포트홀의 이미지와 위치 정보 등을 추가적으로 연계하면 도로 손상 현황 파악 및 유지보수 우선순위 분석에 활용할 수 있다.

**실제 도로 환경에 적합한 모델 선택**

모델별 Recall, Precision, FP/image 및 추론 속도 특성을 비교하여 적용 환경에 맞는 객체 탐지 모델을 선택하기 위한 근거를 제공한다.

**소형 포트홀 및 오탐 개선 연구**

이번 연구에서 확인한 작은 포트홀의 낮은 Recall과 Confidence 관련 미검출 유형은 향후 데이터 보강, 모델 개선 및 Threshold 최적화를 위한 기초 자료로 활용할 수 있다.

## 6. 최종 결론

본 연구에서는 YOLOv8s, YOLOv8+CNN 및 RT-DETR을 단계적으로 개발·비교하고, 5-fold 교차검증과 외부 데이터 평가를 통해 포트홀 탐지 성능을 분석하였다.

CNN 추가는 Recall의 소폭 향상에도 불구하고 종합 성능 개선을 확인하지 못했으며, RT-DETR은 내부 및 외부 데이터에서 YOLOv8s보다 높은 AP와 Recall을 기록하였다.

그러나 RT-DETR은 더 많은 오탐을 발생시켰고, 두 모델 모두 작은 포트홀에서 상대적으로 낮은 탐지 성능을 보였다.

따라서 실제 도로 환경에서의 포트홀 탐지 모델 선택에는 단일 정확도 지표뿐 아니라 오탐 수준, 객체 크기별 탐지 성능 및 추론 비용을 함께 고려해야 한다.


## 7. 저장소 구조 및 파일 안내

```text
pothhole-/
├── README.md
├── .gitignore
├── notebooks/
│   ├── training/          # 초기 학습, 데이터 증강 및 RT-DETR 실험
│   ├── comparison/        # YOLO vs CNN, 5-fold 및 해상도 비교
│   └── analysis/          # 추가 오류/크기 분석
├── scripts/
│   └── evaluation/        # 한국 외부 도로 데이터 모델 비교
├── results/
│   └── external_eval/     # 평가 CSV (원본 이미지 없이 공유)
└── reports/               # 기존 실험 보고서와 결과 압축 파일
```

### 주요 코드 및 노트북

| 구분 | 파일 | 용도 |
|---|---|---|
| 학습/실험 | `notebooks/training/project.ipynb` | 초기 프로젝트 실험 노트북 |
| 증강 | `notebooks/training/road_augmentation.ipynb` | 젖은 아스팔트, 그림자, 블러 및 노이즈 증강 |
| 학습 | `notebooks/training/week6_rtdetr.ipynb` | 6주차 RT-DETR 실험 |
| 비교 | `notebooks/comparison/week5_kfold5.ipynb` | 5-Fold 모델 성능 비교 |
| 비교 | `notebooks/comparison/week7_fp_resolution.ipynb` | 오탐 목표와 해상도 비교 |
| 비교 | `notebooks/comparison/yolo_vs_cnn_yolo.ipynb` | YOLO 단독 및 CNN 결합 비교 |
| 분석 | `notebooks/analysis/pothole_error_size_analysis.ipynb` | 오류 및 객체 크기 분석 |
| 외부 평가 | `scripts/evaluation/korean_external_yolo_rtdetr_comparison_v3.py` | 한국 도로 외부 평가 코드 |

나머지 노트북은 해당 실험 범주에 보존되어 있다. 파일명 이동만으로 노트북 내부의 데이터 경로나 체크포인트 경로가 자동 수정되는 것은 아니므로 실행 전에 경로를 점검해야 한다.

### 외부 평가 CSV 스키마

대용량 도로 이미지 대신 아래와 같은 **수치 결과 파일**을 `results/external_eval/`에 보관한다.

| CSV | 의미 |
|---|---|
| `metrics_by_checkpoint.csv` | Fold·모델·입력 해상도·FP 목표별 TP/FP/FN, Recall, Precision 등 |
| `model_summary.csv` | 모델별 평균·표준편차, AP 및 작은 포트홀 성능 요약 |
| `paired_model_differences.csv` | 동일 Fold 조건에서 RT-DETR과 YOLO 성능 차이 |
| `per_image_metrics.csv` | 이미지 단위 TP·FP·FN 및 설정별 평가 기록 |

CSV 값은 평가 조건과 임계값에 종속된다. 서로 다른 FP 목표나 데이터 분할의 결과를 같은 조건으로 간주하지 않는다.

## 8. 데이터 관리 및 재현 안내

- **Git에 포함:** Python 스크립트, Jupyter 노트북, README, 재현 가능한 평가 CSV
- **Git에서 제외:** 원본 도로 이미지·라벨 대량 묶음, 학습 체크포인트(`.pt`, `.pth`), 대형 중간 산출물, 로컬 환경 파일
- 데이터셋과 모델 가중치의 입수 경로·라이선스·버전·해시를 별도로 기록해야 완전한 재현이 가능하다. 이 저장소만으로 원본 데이터나 학습 가중치가 제공되는 것은 아니다.
- 노트북은 Colab 및 로컬 환경에 따라 경로와 의존성이 다를 수 있다. 코드 실행 전 데이터 경로, 모델 가중치 경로, Python 패키지 버전을 확인한다.
- 민감 정보, 접근 토큰, 개인 경로가 있는 경우 공개 전에 제거한다.

### 실행 개요

1. 데이터 사용 권한을 확인하고 도로 이미지·라벨을 로컬 또는 별도 스토리지에 준비한다.
2. 필요한 Python 환경 및 의존성을 각 노트북/스크립트에 맞게 설치한다.
3. 학습 또는 평가 코드를 실행하되 데이터 및 가중치 경로를 실제 위치로 수정한다.
4. 생성된 CSV를 `results/external_eval/`의 관련 결과와 비교한다.

> 주의: README의 결과표는 기존 실험 기록을 요약한 것으로, 데이터·가중치가 제공되지 않는 상태에서 바로 재현할 수 있음을 의미하지 않는다.
