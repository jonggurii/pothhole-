data file link 
https://universe.roboflow.com/brad-dwyer/pothole-voxrl


🛣️ 머신러닝 및 딥러닝 기반 도로 노면 손상 분류 기법의 비교 연구A Comparative Study of Machine Learning and Deep Learning Approaches for Road Surface Damage Classification   본 연구는 도로 노면 손상(포트홀)의 정밀한 감지를 위해 다양한 YOLOv8 모델 체급(Nano, Small, Medium)과 5-Fold 교차 검증, 영상처리 기반 미탐(False Negative) 원인 분석, 그리고 2-Stage Cascade (YOLO Detector + CNN Classifier) 파이프라인을 구축하여 오탐(False Positive)을 극대화하여 억제하는 연구 프로젝트입니다.   


📌 주요 연구 내용 (Key Contributions)데이터 수급 및 정제 규격화: Roboflow Pothole Dataset (voxrl) 및 Kaggle 수집 데이터셋의 Pascal VOC XML 포맷을 YOLO TXT 포맷으로 변환하고, 단일 클래스(0: pothole) 정제 및 수동 라벨 검수를 통한 노이즈 제거.   데이터 누수 방지 & 5-Fold 교차 검증: Train, Validation, Test 세트 분할(70:15:15 / 80:10:10)과 5-Fold Out-of-Fold(OOF) Cross-Validation 파이프라인을 구축하여 일반화 성능 검증.   도메인 특화 데이터 증강: 그림자(Shadow), 젖은 노면 반사광(Reflection), 부분 가려짐(Occlusion) 환경에 대응하기 위해 HSV 변주(hsv_v=0.4, hsv_s=0.5), Random Erasing(erasing=0.4), Mosaic 후반 해제(close_mosaic=20) 기법 적용.   영상처리 기반 동적 오류 분석 (Failure Analysis): 100장 테스트 세트 내 라벨링되었으나 놓친 77개 미탐(FN) 객체의 패치를 크롭하여 면적 비율, 평균 밝기, 대비 표준편차 수치를 바탕으로 5가지 원인군으로 자동 분류.   2-Stage Cascade 파이프라인 설계: 1-Stage 탐지기의 오탐 한계를 극복하기 위해 Stage 1 (High Recall YOLO Detector)과 IoU $\ge$ 0.50 정답 매핑 기반 Stage 2 (High Precision CNN Classifier)를 결합하여 가짜 오탐 상자를 최대 93.27% 제거.   인터랙티브 웹 데모 구축: Gradio 기반 UI를 통해 Confidence Threshold를 실시간 조정하고 1-Stage 단일 탐지 결과와 2-Stage Cascade 검수 결과를 좌우 비교하는 웹 인터페이스 구현.
---

## 🏗️ 시스템 아키텍처 (System Architecture)
[Raw Road Images]
       │
       ▼
[Data Preprocessing & Augmentation] ──► (HSV, Random Erasing, Mosaic, Clahe)[cite: 4, 6, 8]
       │
       ▼
[Stage 1: YOLOv8 Detector] ─────────► Candidate Bounding Boxes (Low Conf = 0.15, High Recall)
       │
       ▼
[IoU Ground-Truth Mapping & Crop] ──► Patch Extraction (IoU ≥ 0.50 Pothole / IoU < 0.10 Background)
       │
       ▼
[Stage 2: YOLOv8n-cls Classifier] ──► False Positive Filtering & Final Verification[cite: 4, 6, 7]
       │
       ▼
[Final High-Precision Detection] ───► Approved Bounding Boxes Output


## 📊 모델 성능 비교 (Experimental Results)

📊 모델 성능 및 실험 결과 (Experimental Results)1. Single YOLOv8 체급별 고해상도(1024) 
최종 정밀도 비교Model NameMax PrecisionConf (at Max P)Recall (at Max P)mAP50 (at Max P)P ≥ 90% Reached
yolov8n_highres_p90   90.00%0.6557.30%54.63% ✅ YES  
yolov8s_highres_p90   88.16%0.7050.19%47.70%  ❌ NO  
yolov8m_highres_p90   92.52%0.7050.94%        ✅ YES   

yolov8m_highres_p90  0.10     0.8682  0.6413 0.6495
yolov8m_highres_p90  0.20     0.8682  0.6413 0.6255
yolov8m_highres_p90  0.25     0.8682  0.6413 0.6255
yolov8m_highres_p90  0.35     0.8643  0.6442 0.6173
yolov8m_highres_p90  0.40     0.8763  0.6367 0.6085
yolov8m_highres_p90  0.45     0.8935  0.6067 0.5821
yolov8m_highres_p90  0.50     0.9040  0.5993 0.5731
yolov8m_highres_p90  0.55     0.9029  0.5918 0.5731
yolov8m_highres_p90  0.60     0.8994  0.5693 0.5460
yolov8m_highres_p90  0.65     0.9182  0.5468 0.5277
yolov8m_highres_p90  0.70     0.9252  0.5094 0.4906


yolov8m_highres_p90  0.10     0.7991  0.6479 0.6298
yolov8m_highres_p90  0.20     0.8107  0.6416 0.6066
yolov8m_highres_p90  0.25     0.8113  0.6442 0.6066
yolov8m_highres_p90  0.35     0.8410  0.6142 0.5818
yolov8m_highres_p90  0.40     0.8394  0.6067 0.5734
yolov8m_highres_p90  0.45     0.8548  0.5955 0.5650
yolov8m_highres_p90  0.50     0.8729  0.5918 0.5650
yolov8m_highres_p90  0.55     0.8743  0.5730 0.5475
yolov8m_highres_p90  0.60     0.8824  0.5618 0.5387
yolov8m_highres_p90  0.65     0.8797  0.5206 0.5034
yolov8m_highres_p90  0.70     0.8980  0.4944 0.4766

================================================================================
🏆 [YOLOv8 3개 모델 고해상도(1024) 최종 정밀도 성능 비교표]
================================================================================
         Model Name Max Precision  Conf (at Max P) Recall (at Max P) mAP50 (at Max P) P >= 90% Reached
yolov8n_highres_p90        90.00%             0.65            57.30%           54.63%            ✅ YES
yolov8s_highres_p90        88.16%             0.70            50.19%           47.70%             ❌ NO
yolov8m_highres_p90        89.80%             0.70            49.44%           47.66%             ❌ NO
================================================================================
🏆 [YOLOv8 3개 모델 고해상도(1024) 최종 정밀도 성능 비교표]
================================================================================
         Model Name Max Precision  Conf (at Max P) Recall (at Max P) mAP50 (at Max P) P >= 90% Reached
yolov8n_highres_p90        90.00%             0.65            57.30%           54.63%            ✅ YES
yolov8s_highres_p90        88.16%             0.70            50.19%           47.70%             ❌ NO
yolov8m_highres_p90        92.52%             0.70            50.94%           49.06%            ✅ YES
================================================================================
### 2. 5-Fold 교차 검증 종합 성능 (YOLOv8s + Weather Augmentation)

| Fold | Precision | Recall | mAP50 |
| :---: | :---: | :---: | :---: |
| **Fold 1**[cite: 4] | 0.7750 | 0.6589 | 0.6587 |
| **Fold 2**[cite: 4] | 0.7816 | 0.7358 | 0.7361 |
| **Fold 3**[cite: 4] | 0.8121 | 0.7514 | 0.7654 |
| **Fold 4**[cite: 4] | 0.8586 | 0.6747 | 0.7338 |
| **Fold 5**[cite: 4] | 0.7637 | 0.6807 | 0.6927 |
| **Average**[cite: 4] | **0.7982** | **0.7003** | **0.7173** |

### 3. Precision 최적화 및 고해상도 하이퍼파라미터 튜닝 결과

- **YOLOv8s High-Precision Config**: 최고 **Precision 0.8816 (~88.2%)**, **mAP50 0.7883 (~78.8%)** 달성

---
<img width="1784" height="425" alt="image" src="https://github.com/user-attachments/assets/74b1fcbe-6ba1-49d1-b78d-cbe63b50102d" />


2. 5-Fold 교차 검증 OOF Confidence Sweeping 결과 (cls=3.5, imgsz=1024)5-Fold OOF(Out-of-Fold) 검증에서 오탐 방지 손실 가중치(cls=3.5)를 부여하고 Confidence Threshold 스위핑을 수행한 평균 성과입니다.   Conf ThresholdAverage PrecisionAverage RecallAverage mAP500.10   80.94%71.65%70.91%0.20   80.14%72.32%68.93%0.25   81.37%71.24%68.30%0.35   82.25%70.69%67.16%0.40   83.87%68.91%66.00%0.45   85.22%67.60%64.83%0.50   85.80%66.95%63.98%0.60   87.44%64.79%62.06%0.65   88.75%62.60%60.30%0.70 (최적 P 지점)   90.59%58.82%56.68%
## 🔍 오류 및 failure 원인 분석 (Failure Analysis)

3. Single YOLOv8 vs 2-Stage Cascade 파이프라인 정량 평가 (100 Test Images)100장의 테스트 세트 전체에서 IoU $\ge$ 0.50 매핑 단일 YOLOv8 baseline과 2-Stage Cascade 파이프라인을 비교한 최종 정량 평가 지표입니다.   Model ArchitecturePrecisionRecallF1-ScoreTPFP (오탐)FN (미탐)Latency (ms/img)Single YOLOv8 (1-Stage Baseline)   34.45%61.42%0.441516431210318.22 ms   2-Stage Cascade (Balanced Mode, Conf 0.20)   52.97%46.82%0.497012511114254.36 ms   2-Stage Cascade (High-P Mode, Conf 0.50)79.00%29.59%0.4305792118826.42 ms2-Stage Cascade (Extreme Mode, Conf 0.70)   92.86%19.48%0.322052421520.41 ms   💡 2-Stage Cascade 주요 성과High-Precision Mode (conf=0.50): 가짜 오탐(FP) 상자를 312개에서 21개로 93.27% 제거, Precision 79.00% (+44.55%p) 달성[cite: 4, 6, 7].Extreme Precision Mode (conf=0.70): 오탐(FP) 상자를 단 4개만 남기고 98.72% 제거, Precision 92.86% (+58.40%p) 달성.   실시간 연산 효율성: 2-Stage 검수를 포함하더라도 이미지당 20.41 ~ 26.42 ms (약 37~49 FPS)의 빠른 추론 속도 유지[cite: 6, 7].🔍 오류 및 미탐 원인 분석 (Failure Analysis: FN 77개 진단)IoU < 0.25 기준 탐지에 실패한 77개 포트홀 객체의 패치를 크롭하여 영상처리 지표로 동적 분류한 결과입니다.   Reason Code수치 판별 조건   PDF현장 발생 원인 및 원인 분석   PDFSmall Object (Far Distance)   area_ratio < 0.006   원경 미세 포트홀로, Feature Map 다운샘플링 과정에서 정보 손실   Dark Shadow / Occlusion   mean_val < 65   가로수, 건물, 작업자 그림자로 인해 명암 대비 손실   Reflection / Glare   mean_val > 185   우천 후 젖은 노면 및 햇빛 반사광으로 아스팔트 질감 상실   Low Contrast / Blended Surface   std_val < 22   포트홀 내부와 주변 도로의 색상·질감이 유사하여 Edge 특성 저하   Complex Road Texture   기타 조건   도로 노면의 복잡한 균열, 아스팔트 패칭 흔적 등으로 경계선 모호[cite: 6]
