
# Pothole Detection — YOLOv8 and CNN Rescoring

Project repository: [jonggurii/pothhole-](https://github.com/jonggurii/pothhole-)

도로 영상에서 포트홀을 탐지하는 프로젝트입니다. YOLOv8 객체 탐지 결과를 기준선으로 삼고, 5주차 분석에서는 YOLOv8의 후보 박스를 CropCNN으로 다시 점수화해 정확도와 추론 비용을 비교했습니다.

## Week 5 experiment

| Metric (5-fold mean) | YOLOv8 | YOLOv8 + CNN |
|---|---:|---:|
| mAP50–95 | 0.45161 | 0.45013 |
| mAP50 | 0.72216 | 0.72445 |
| Precision | 0.78228 | 0.76310 |
| Recall | 0.64250 | 0.65655 |
| F1 | 0.70415 | 0.70369 |

CNN rescoring increased recall by 1.41 percentage points and reduced precision by 1.92 points. Mean mAP50–95 and F1 did not improve. Across all five outer folds, it added 22 true positives and 41 false positives. This makes the result an accuracy–false-alarm trade-off rather than an overall win for CNN rescoring.

| Inference setup | YOLOv8 | YOLOv8 + CNN |
|---|---:|---:|
| 800 / FP32 | 7.205 ms · 138.809 FPS | 9.463 ms · 105.832 FPS |
| 640 / FP32 | 5.963 ms · 167.718 FPS | 7.933 ms · 126.145 FPS |
| 640 / FP16 | 3.504 ms · 287.965 FPS | 5.875 ms · 173.210 FPS |

Latency measures the in-memory PyTorch pipeline through preprocessing, detection, NMS, CNN crops and score filtering. It excludes disk decoding and visualization. The Colab run did not retain the exact GPU model, so the FPS values are specific to that runtime and are not directly comparable to TensorRT paper benchmarks.

For the fold-by-fold results, error-case images, precision–recall curves, protocol, and interpretation, see [WEEK5_RESULTS.md](WEEK5_RESULTS.md). To regenerate the tables, plots and images or rerun the three speed scenarios, run [`week5_final_analysis.ipynb`](week5_final_analysis.ipynb) in Colab with the existing Drive evaluation files and a GPU runtime.

## Method at a glance

The five-fold evaluation uses the same fold-specific YOLO weights and candidate boxes for both methods. CNN rescoring applies `YOLO score × CNN probability^alpha`; it changes candidate ranking and threshold decisions, but does not generate or refine bounding boxes. Alpha and operating thresholds are selected from each fold's inner validation data and then applied to its outer fold. Object-detection metrics use COCO bbox evaluation; operating-point TP/FP/FN counts use one-to-one matching at IoU 0.5.

A score of `alpha=0` means the CNN has no effect on candidate scores for that fold. Results are reported as fold means with sample standard deviations; the small number of folds and overlapping training sets do not support treating those standard deviations as independent-run significance tests.

## Reproduce the Week 5 analysis

1. Put the analysis notebook in the repository as `week5_final_analysis.ipynb`.
2. Open it in Google Colab and select a GPU runtime.
3. Confirm the Drive paths in the first setup cell point to the completed `Pothole/week5_kfold_results/kfold5/evaluation_repaired_*` run.
4. Run the notebook cells in order. It reuses saved weights and outer predictions; it does not retrain the five folds.
5. Review `speed/status.csv`. All 15 fold × speed-condition entries should show `OK` before reporting the complete speed comparison.

The notebook writes fold metrics, PR data, case-selection evidence, case images, speed measurements, scenario accuracy, an HTML report, and a ZIP under the selected evaluation directory.

## Speed experiment conditions

- A: original 800 input, FP32.
- B: 640 input, FP32.
- C: 640 input, FP16 on CUDA.
- Batch size 1, 30 randomly selected outer images per fold, 3 timed repeats, and 10 warm-up inferences.
- The exact device model is not present in the saved output; record it when reproducing on another machine.

The literature reference is Wang et al., [*YOLOv10: Real-Time End-to-End Object Detection*](https://arxiv.org/html/2405.14458v1), Table 1 and §4.1. Its YOLOv8 latency values use NVIDIA T4 and TensorRT FP16, so they are context rather than a like-for-like speed target for this PyTorch Colab measurement.
