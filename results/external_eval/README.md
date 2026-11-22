# External evaluation CSV results

이 폴더에는 원본 도로 사진이나 모델 가중치 대신, 외부 한국 도로 데이터 평가에서 생성한 CSV 결과만 보관합니다.

| CSV | Description |
| --- | --- |
| `metrics_by_checkpoint.csv` | Fold/model/resolution/FP target별 혼동 지표 및 성능 |
| `model_summary.csv` | 모델별 평균과 표준편차 및 AP 요약 |
| `paired_model_differences.csv` | 동일 Fold에서 RT-DETR과 YOLO 성능 차이 |
| `per_image_metrics.csv` | 이미지별 TP/FP/FN, small GT 및 threshold |

CSV는 원본 사진을 포함하지 않지만 이미지 파일명과 ID가 들어갈 수 있습니다. 공개 저장소로 전환할 경우 경로와 식별자 포함 여부를 검토하세요.

**현재 상태:** CSV 파일은 별도 업로드 대상이며, 이 브랜치에는 아직 들어 있지 않습니다.
