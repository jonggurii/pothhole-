# 실험 진행 과정별 GitHub 파일 배치

이 문서는 코드/CSV 중심 저장소의 구성 원칙과 단계별 자료 위치를 설명합니다. GitHub에 이미 있는 파일과 추가 업로드할 자료를 구분합니다.

| 단계 | 연구 내용 | GitHub 기존 파일 | 추가 정리 대상(업로드 패키지) |
| --- | --- | --- | --- |
| 1–3주차 | 초기 데이터 처리·YOLOv8 기준모델 | `notebooks/training/project.ipynb` | `notebooks/training/week01_03_baseline.ipynb` |
| 3주차 | 날씨/그림자/모션블러 증강 | `notebooks/training/road_augmentation.ipynb` | `week03_weather_augmentation_legacy.ipynb` |
| 4–5주차 | YOLOv8 ↔ YOLO+CropCNN 비교 | `notebooks/comparison/yolo_vs_cnn_yolo.ipynb`, `week5_kfold5.ipynb` | 초기 비교·평가기 복구 v1/v2 노트북, 5-Fold CSV/YAML |
| 6주차 | YOLOv8s ↔ RT-DETR-L 학습 | `notebooks/training/week6_rtdetr.ipynb` | 기존 코드 보존 |
| 7주차 | FP/image·입력 해상도·분할 복원 | `notebooks/comparison/week7_fp_resolution.ipynb` | `scripts/evaluation/week07/`, `results/week07_fp_resolution/`, `metadata/week07_split/` |
| 8–10주차 | 한국 도로 외부 일반화 검증 | `scripts/evaluation/korean_external_yolo_rtdetr_comparison_v3.py` | `notebooks/external_validation/`, `results/external_eval/` |
| 후속 분석 | 객체 크기, FP/FN, oracle union | `notebooks/analysis/` | `scripts/analysis/`, `results/followup/` |

## 보관 원칙

- **Git에 포함:** 실행 코드(`.py`, `.ipynb`), 설정(`.yaml`, `.json`), 성능 지표(`.csv`), 문서(`.md`).
- **Git에서 제외:** 원본 도로 사진/라벨·모델 가중치·prediction cache·학습 batch 이미지·불완전한 다운로드 파일.
- 새 Notebook은 원래 코드와 설명을 보존하되, 대용량 실행 출력 이미지 및 셀 실행 횟수를 제거합니다.
- CSV에는 이미지 경로와 이름이 일부 포함되므로 공개 저장소 전환 전 개인정보·라이선스 확인이 필요합니다.
- 원본 데이터와 모델 가중치가 없으므로 전체 학습을 이 저장소만으로 즉시 재현할 수는 없습니다.

## 추가 업로드 상태

사용자가 전달한 새 ZIP 파일에서 코드/CSV를 분리하고 대용량 이미지를 제외한 패키지가 준비되어 있습니다. **이 문서가 올라간 사실이 추가 코드/CSV가 GitHub에 업로드되었다는 의미는 아닙니다.** 실제 파일 병합 완료 후 [README](../README.md) 및 `docs/EXPERIMENTS.md`에서 결과 파일별 설명을 확인할 수 있습니다.

미완료 파일 `korean_external_results-20261118T101502Z-1-001.zip.crdownload`은 정상 ZIP이 아니므로 포함하지 않았습니다.
