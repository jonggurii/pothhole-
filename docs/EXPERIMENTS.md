# 포트홀 프로젝트 — 실험 진행 순서 및 코드·결과 파일 가이드

이 문서는 코드와 결과 CSV를 **실험 진행 과정**에 따라 찾을 수 있도록 안내합니다. 기존 저장소에 있는 Notebook/코드와 신규 업로드 파일을 함께 설명합니다.

## 진행 순서

| 시기 | 실험 주제 | 코드 경로 | 결과/설정 경로 |
|---|---|---|---|
| Week 1–3 | 데이터 라벨링·YOLOv8 기준 모델·기초 학습 | `notebooks/training/week01_03_baseline.ipynb`, 기존 `notebooks/training/project.ipynb` | 원본 이미지는 Git에 포함하지 않음 |
| Week 3 | 날씨·그림자·모션 블러 등 증강 | `notebooks/training/week03_weather_augmentation_legacy.ipynb`, 기존 `notebooks/training/road_augmentation.ipynb` | 데이터 증강 결과 이미지는 제외 |
| Week 4–5 | YOLOv8 단독 vs YOLOv8+CropCNN | `notebooks/comparison/week05_yolo_cnn_early.ipynb`, 기존 `notebooks/comparison/yolo_vs_cnn_yolo.ipynb` | `results/kfold/` 학습 로그 |
| Week 5 | 5-fold 비교, 평가기 mAP/alpha 오류 복구 | 기존 `notebooks/comparison/week5_kfold5.ipynb`, `notebooks/comparison/legacy/week05_eval_recovery_v*.ipynb` | 5-fold 결과 CSV와 YAML |
| Week 6 | RT-DETR-L 모델 및 YOLOv8s 비교 | 기존 `notebooks/training/week6_rtdetr.ipynb` | Week 6 결과는 기존 보고서 참고 |
| Week 7 | FP/image 목표·640/800/1024 해상도·분할 복구 | 기존 `notebooks/comparison/week7_fp_resolution.ipynb`, `scripts/evaluation/week07/` | `results/week07_fp_resolution/{summary,curves,detail,metadata}/`, `metadata/week07_split/` |
| Week 8–10 | 한국 도로 외부 일반화 검증 | `notebooks/external_validation/`, 기존 `scripts/evaluation/korean_external_yolo_rtdetr_comparison_v3.py` | `results/external_eval/` |
| 후속 | 포트홀 상대 크기·FP/FN·모델 상호보완성 | 기존 `notebooks/analysis/`, `scripts/analysis/pothole_error_size_analysis.py` | `results/followup/` |
| 보조 | GPU/W&B 파이프라인 및 드라이브 진단 | `notebooks/experimental/` | 주 실험과 분리 관리 |

## 주요 CSV 읽는 방법

- `results/kfold/yolo_800/fold_{1..5}/results.csv`: YOLOv8s 800px에서 Fold별 epoch 학습 기록 (`args.yaml` 동봉)
- `results/kfold/yolo_precision_1024/fold_{1..5}/results.csv`: 정밀도 지향 YOLOv8s 1024px 실험 기록 (`args.yaml` 동봉)
- `results/week07_fp_resolution/summary/`: 탐색 grid, 오탐 정책, 검증/외부 비교 요약 CSV
- `results/week07_fp_resolution/curves/`: Fold/모델/해상도별 validation confidence curve CSV. **CSV만 존재하며 예측 이미지가 아님**
- `results/week07_fp_resolution/detail/`: 작은 포트홀, 이미지별 오류, 예측 품질 분석
- `results/week07_fp_resolution/metadata/`, `metadata/week07_split/`: 결정된 FP 정책·분할·재현성 확인용 메타데이터
- `results/external_eval/`: 체크포인트 요약, 모델별 평균, 대응 Fold 차이, 이미지별 결과
- `results/followup/`: 크기별 미검출율, FP confidence, oracle union, 수동 검수 목록

## 파일 버전 및 주의사항

- 파일명보다 **노트북의 실제 첫 제목/코드 내용**을 기준으로 주차를 매핑했습니다. 예: 원본 `4주차 cnn vs yolo.json`의 첫 제목은 `5주차: YOLOv8 vs YOLOv8 + CNN`이므로 Week 5 초기 구현으로 분류했습니다.
- JSON 형태로 저장된 Jupyter 노트북은 `.ipynb` 확장자로 정상화했습니다.
- 새로 추가한 Notebook의 실행 출력(이미지, 셀 실행기록)은 제거했으나 **코드 및 Markdown 셀은 유지**했습니다. 원본은 업로드 ZIP 안에 그대로 남아 있습니다.
- 이미 GitHub에 역할상 동일한 파일이 있는 경우, 중복 복제 대신 기존 경로를 표기했습니다. 서로 다른 실행 이력까지 코드가 같다고 단정하지는 않습니다.
- CSV 속 `image`, `image_id`, 데이터 폴더 경로 등은 평가 기록일 수 있습니다. 공개 저장소 전환 시 민감하거나 라이선스 제한이 있는 식별자가 없는지 검토하세요.
- 가중치(`*.pt` 등), 도로 사진, batch 시각화, prediction cache, 원본 데이터셋은 Git에 올리지 않습니다.
- `korean_external_results-20261118T101502Z-1-001.zip.crdownload`는 불완전한 다운로드 파일로 판단되어 이번 정리에서 제외했습니다. 완성된 ZIP을 받으면 별도 검증·병합해야 합니다.

## 실행 환경과 재현성

이 저장소는 **코드와 정량 결과 아카이브**입니다. 실행하려면 Google Colab 또는 로컬 PyTorch/Ultralytics 환경, 별도 다운로드한 데이터셋과 체크포인트가 필요합니다. 노트북 내 `/content/drive/MyDrive/...` 같은 경로는 원 실험 환경의 경로이므로 실행 시 사용자의 환경에 맞게 바꿔야 합니다. 실제 버전은 노트북 설치 셀, `results/kfold/**/args.yaml` 및 `metadata/week07_split/` 설정을 확인하세요.

[상위 README](../README.md)에서 프로젝트 동기·성능 비교 수치·결론을 볼 수 있습니다.
