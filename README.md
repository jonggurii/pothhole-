data file link 
https://universe.roboflow.com/brad-dwyer/pothole-voxrl


# 🛣️ 머신러닝 및 딥러닝 기반 도로 노면 손상 분류 기법의 비교 연구
> **A Comparative Study of Machine Learning and Deep Learning Approaches for Road Surface Damage Classification**

본 연구는 도로 노면 손상(포트홀)의 정밀한 감지를 위해 다양한 YOLOv8 모델 체급(Nano, Small, Medium)과 2-Stage Cascade(YOLO Detector + CNN Classifier) 파이프라인, 그리고 도메인 특화 데이터 증강 기법을 비교 및 평가하는 프로젝트입니다

---

## 📌 주요 연구 내용 (Key Contributions)

- **데이터 정제 및 수급 규격화**: Kaggle 및 Roboflow에서 수집한 데이터셋의 Pascal VOC XML 포맷을 YOLO TXT 포맷으로 변환하고, 단일 클래스(`0: pothole`) 매핑 및 수동 라벨 검수를 통한 노이즈 제거 완료[cite: 3, 6, 7].
- **데이터 누수 방지 (Data Leakage Prevention)**: Train, Validation, Test 세트 분할(70:15:15 / 80:10:10) 및 5-Fold 교차 검증(Cross-Validation) 적용.
- **도메인 특화 날씨/조명 데이터 증강**: 악천후(눈/비), 그림자(Shadow), 젖은 노면 반사광(Reflection), 부분 가려짐(Occlusion) 환경을 모사한 증강 기법 구현 (`hsv_v`, `erasing`, `scale` 등)[cite: 4, 6, 9].
- **목표 지표 재설정 (Problem Reformulation)**: 실제 도로 유지보수 현장의 불필요한 출동 비용을 줄이기 위해 목표 지표를 Precision 85% 이상으로 재설정하고, 높은 Recall을 병행 달성하기 위한 2-Stage 필터링 파이프라인 구축.

---

## 🏗️ 시스템 아키텍처 (System Architecture)
[Raw Road Images]
│
▼
[Data Preprocessing & Augmentation] ──► (Weather, Shadow, Occlusion Simulation)
│
▼
[1-Stage YOLOv8 Detector] ──────────► Bounding Box Candidates (High Recall Target)
│
▼
[2-Stage Crop & CNN Classifier] ────► False Positive Filtering (High Precision Target)
│
▼


---

## 📊 모델 성능 비교 (Experimental Results)

### 1. YOLOv8 체급별 초기 검증 성능 (150 Epochs, imgsz=1024)[cite: 4]

| Model | Resolution | Precision | Recall | mAP50 | mAP50-95 | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n (Nano)**[cite: 4] | 1024 | - | - | 0.732 | 0.462 | ~0.150 hrs[cite: 4] |
| **YOLOv8s (Small)**[cite: 4] | 1024 | - | - | 0.742 | 0.464 | ~0.174 hrs[cite: 4] |
| **YOLOv8m (Medium)**[cite: 4] | 1024 | - | - | 0.775 | 0.488 | ~0.328 hrs[cite: 4] |

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



## 🔍 오류 및 failure 원인 분석 (Failure Analysis)

학습된 모델이 현장 평가 시 오탐/미탐을 일으키는 주요 원인 분석 및 대응

1. **그림자 (Shadow)**: 가로수 및 건물의 강한 그림자로 인한 노면 명암 차이 ➔ `hsv_v=0.4` 증강 적용
2. **반사광 (Reflection)**: 우천 후 젖은 아스팔트 및 물웅덩이 표면의 빛 반사 ➔ `hsv_s` 및 색조 변주 증강 적용
3. **부분 가려짐 (Occlusion)**: 차량, 이물질, 눈덩이에 의한 형태 가려짐 ➔ `erasing=0.4` (Random Erasing) 기법 적용
4. **유사 객체 오탐 (False Positives)**: 맨홀 뚜껑, 아스팔트 패칭 흔적 ➔ 2-Stage CNN 분류기 연동으로 2차 필터링

