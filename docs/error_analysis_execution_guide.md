# 포트홀 탐지 실패 원인 + 크기별 Recall 분석 실행 가이드

## 제공 파일

- `pothole_error_size_analysis_colab.ipynb` — 구글 Colab에서 바로 열어서 실행할 수 있는 **코드 내장** 노트북
- `pothole_error_size_analysis.py` — 분석 프로그램 전체 코드 (GPU/Ultralytics 재설치 없이 실행 가능)

## 필요한 기존 결과 파일

기존 `korean_external_yolo_rtdetr_comparison_v3.py` 실행이 완료되었다면 다음 파일을 남깁니다.

- `external_.../completed.json`
- `external_.../external_manifest.json`
- `external_.../external_protocol.json`
- `external_.../prediction_cache_index.json`
- `external_.../per_image_metrics.csv`
- `prediction_cache/<sha256>.json` (실제 두 모델의 예측 박스 및 Confidence 캐시)
- 원본 PNG/JPG 파일들 (이미지 특성 추출, 이미지 시각화에 필요)

**현재 대화에 업로드된 CSV만으로는 개별 bbox·Confidence를 복원할 수 없습니다.** 기존 Colab Drive 실행의 외부 평가 폴더와 캐시가 필요합니다.

## 방법 A: Colab 노트북

1. `pothole_error_size_analysis_colab.ipynb`를 Colab에서 엽니다.
2. Google Drive 마운트합니다.
3. `RUN_DIR`에 기존 외부 평가 `external_...` 완료 폴더를 지정합니다. `None`이면 최근 완료 폴더를 자동 선택합니다.
4. 코드 셀을 순서대로 실행합니다. 기존 Fold·threshold 재사용, 새로운 학습/추론은 **없음**.
5. 생성된 `error_size_analysis_YYYYMMDD_HHMMSS` 폴더에 결과가 저장됩니다.

## 방법 B: Python CLI

```bash
python pothole_error_size_analysis.py \
  --run-dir '/content/drive/MyDrive/Pothole/korean_external_results/external_실제폴더명' \
  --cache-root '/content/drive/MyDrive/Pothole/korean_external_results/prediction_cache' \
  --case-folds 1 \
  --case-targets 0.2 0.5 \
  --max-cases-per-folder 20
```

### 특정 실패 이유의 이미지만 추출

```bash
python pothole_error_size_analysis.py \
  --run-dir '/content/drive/MyDrive/Pothole/korean_external_results/external_실제폴더명' \
  --case-folds 1 \
  --case-reasons BELOW_THRESHOLD NO_NEARBY_CANDIDATE \
  --max-cases-per-folder 0
```

상기 명령은 CSV에서는 **모든 모델, Fold, FP 목표의 오류를 그대로 분석**하지만, JPG 추출은 Fold 1 및 선택한 사유만 대상으로 합니다. `--max-cases-per-folder 0`은 모든 해당 이미지를 추출하므로 출력 용량이 커질 수 있습니다.

전 Fold 이미지 추출: `--case-folds 1 2 3 4 5`.

## 가장 중요한 결과

| 파일 | 내용 |
|---|---|
| `gt_object_analysis.csv` | GT 1개마다 모델별 TP/FN, 실패 원인, Confidence/IoU, 상대 크기 |
| `failure_image_index.csv` | **놓친 포트홀 전체 목록**, 원본 이미지 경로, 모델·Fold·threshold·이유 |
| `fp_prediction_analysis.csv` | FP 예측박스별 오탐 조건/위치/Confidence |
| `image_comparison.csv` | 이미지별 YOLO vs RT-DETR 결과 |
| `human_review_template.csv` | 사람이 실제 실패 환경(그림자, 물 반사, GT 오류 등)을 확인할 수 있는 검수표 |
| `size_recall_by_checkpoint.csv` | 각 크기 그룹의 TP/GT/Recall (Fold/모델/목표별) |
| `size_recall_summary.csv` | 크기 그룹별 Fold 평균 및 표준편차 |
| `failure_reasons_by_size.csv` | 작은 포트홀에서 어떤 오류가 자주 나타나는지 분석 |
| `model_disagreement_by_size.csv` | 두 모델 간 크기별 탐지 성공/실패 불일치 |
| `cases/` | **실패 유형별 박스 표시 JPG 이미지** (표본, 사용자가 전부 추출 가능) |
| `case_export_index.csv` | 실제로 추출한 JPG 파일 위치 |

**표시 색상**: GT 초록, TP 초록, FP 빨강, 놓친 GT는 주황 점선.

## 분석 기준

- 고정 Confidence threshold: 기존 7주차 Validation에서 결정된 원래 모델/Fold/FP목표 값만 사용
- GT-예측 박스 TP 판정: score 내림차순으로 unmatched GT 중 최대 IoU가 **0.5 이상**이면 TP
- `BELOW_THRESHOLD`: IoU ≥ 0.5 후보가 있지만 원 threshold 미만
- `MATCHING_CONFLICT`: IoU ≥ 0.5, threshold 이상인 후보가 존재하나 다른 GT에 우선 매칭
- `LOCALIZATION_IOU_LT_05`: threshold 이상 예측의 GT IoU가 0.1 이상 0.5 미만
- `LOW_SCORE_AND_LOCALIZATION`: threshold 미만 후보가 GT에 IoU 0.1 이상 0.5 미만
- `NO_NEARBY_CANDIDATE`: 캐시에 IoU ≥ 0.1 후보가 없음 (모델 내부의 미활성화 증거는 아님)
- 이미지 분위수 크기 구간: 원본 bbox 면적 / 이미지 전체 면적의 Q25, Q75. **COCO Small과 다른 분석 지표**.
- 밝기·대비·선명도는 측정 가능한 특징과 탐지 실패의 **관련성 분석**일 뿐, 원인 입증이 아닙니다.
- 검증: 분석 중 기존 `per_image_metrics.csv`의 **모든 이미지 × Fold × 목표 × 모델의 TP/FP/FN을 반드시 대조**합니다. 하나라도 다르면 종료합니다.

## 검증 범위

코드는 합성 이미지·예측 캐시를 사용하는 전체 파이프라인 테스트, IoU 매칭·오류 유형 유닛 테스트를 통과했습니다. 현재 대화의 CSV에는 원본 이미지와 전체 캐시가 포함되어 있지 않으므로 **실제 745장 재분석은 아직 실행하지 않았습니다**.
