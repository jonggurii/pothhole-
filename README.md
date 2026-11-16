1. 프로젝트 개요
1.1 프로젝트 제목
YOLOv8과 RT-DETR 기반 포트홀 탐지 모델의 성능 비교 및 외부 도로 환경에서의 일반화 성능 분석

영문 제목:

Comparative Evaluation of YOLOv8 and RT-DETR for Pothole Detection and Generalization Analysis in Real-World Road Environments

이 제목은 최종 Week 7~10 연구 내용에 가장 중점을 둔 제목이다. 프로젝트 초기의 YOLOv8 모델 최적화와 CNN 결합 실험도 본 연구의 선행 개발 과정으로 포함한다.
1.2 연구 배경
포트홀(Pothole)은 도로 표면의 일부가 파손되어 움푹 파인 형태의 도로 손상이다. 포트홀은 차량 타이어 및 하부 부품 손상, 운전자의 급격한 회피 조작, 도로 주행 안전성 저하 등의 문제를 일으킬 수 있다.

기존의 도로 포트홀 점검은 관리 인력의 현장 조사나 차량을 활용한 육안 점검에 의존하는 경우가 있다. 이러한 방식은 조사 범위가 넓어질수록 시간과 인력이 많이 필요하며, 도로 손상을 지속적으로 관찰하기 어렵다는 한계가 있다.

이에 따라 차량용 카메라나 도로 영상에서 포트홀을 자동 탐지하는 인공지능 기반 객체 탐지(Object Detection) 기술이 중요해지고 있다.

그러나 실제 도로 영상에는 다양한 탐지 장애 요인이 존재한다.
- 포트홀과 일반 아스팔트 표면의 낮은 색상 대비
- 그림자, 빛 반사, 노면의 보수 흔적
- 원거리에서 작게 촬영되는 포트홀
- 카메라 시점 및 촬영 환경 변화
- 균열, 맨홀, 물웅덩이 등 포트홀과 유사한 시각적 패턴
- 높은 Recall 확보 과정에서 증가할 수 있는 오탐(False Positive)
따라서 단순히 높은 객체 탐지 정확도를 기록하는 것만으로는 실제 도로 환경에서 효과적인 포트홀 탐지 모델이라고 판단하기 어렵다.

포트홀을 놓치지 않는 능력(Recall), 잘못 탐지하지 않는 능력(Precision), 처리 속도(Latency), 새로운 도로 환경에서의 일반화 성능(Generalization)을 함께 고려해야 한다.
1.3 연구 목적
본 프로젝트의 궁극적인 목표는 딥러닝 기반 포트홀 탐지 시스템을 구축하고, 서로 다른 객체 탐지 구조가 도로 손상 탐지 성능에 미치는 영향을 분석하는 것이다.

이를 위해 프로젝트를 단계적으로 진행하였다.
1. 포트홀 이미지 데이터 수집, 라벨 정제 및 모델 학습 환경 구축
2. YOLOv8 계열의 경량·중형 모델 비교 및 하이퍼파라미터 최적화
3. 데이터 증강과 Confidence Threshold 조정을 통한 탐지 성능 분석
4. YOLOv8 후단에 CNN 분류기를 결합한 2단계 검출 구조 구현
5. 기존 CNN 기반 검출 모델과 Transformer 기반 RT-DETR의 비교
6. 동일한 5-fold 평가 조건에서 모델 간 Precision, Recall, mAP, F1 및 오탐 수준 분석
7. 소형 포트홀과 모델별 탐지 실패 사례 분석
8. 학습에 사용하지 않은 외부 한국 도로 이미지에서 일반화 성능 검증
초기에는 탐지 모델 자체의 성능을 높이는 것에 초점을 두었으나, 후반부에는 실제 오탐 부담을 고려한 모델 비교와 외부 데이터에서의 성능 유지 여부를 연구의 핵심 질문으로 확장하였다.

2. Week 1~10 전체 연구 흐름
전체 연구는 다음 네 단계로 구분할 수 있다.
PHASE 1 · WEEK 1~3

데이터 구축 및 YOLOv8 기준 모델 개발

데이터 정제 → YOLOv8n/s/m 비교 → 데이터 증강 → 임계값 분석 → 초기 CNN 결합


PHASE 2 · WEEK 4~5

YOLOv8+CNN 하이브리드 모델 검증

2-Stage Cascade 구축 → 5-fold 정량 비교 → 오탐·미검출 분석 → 추론 속도 비교


PHASE 3 · WEEK 6~8

RT-DETR 도입 및 공정한 비교 평가

Transformer 구조 분석 → RT-DETR 학습 → 동일 5-fold 평가 → FP 예산 및 해상도별 비교


PHASE 4 · WEEK 9~10

외부 데이터 일반화 및 오류 분석

독립 외부 데이터 확보 → 기존 가중치와 임계값 고정 → 성능 평가 → 크기별·실패 유형별 분석


주차	핵심 활동	연구에서의 역할
Week 1	데이터 수집·정제, YOLOv8s Baseline	실험 기반 구축
Week 2	YOLO 모델 비교, 증강, TTA, 임계값 조정	성능 개선 탐색
Week 3	미검출 분석, CNN Cascade 초기 구현	문제점 발견 및 구조 확장
Week 4	YOLO+CNN 재구성, 5-fold 평가	비교 절차 정비
Week 5	정량 비교, 오류·속도 분석	CNN 결합 효과 검증
Week 6	ViT·RT-DETR 학습, 시험 실행	새로운 검출 모델 도입
Week 7	YOLOv8s·RT-DETR-L 5-fold 학습	모델 구조 비교
Week 8	FP 예산, Small Recall, 입력 해상도 분석	운용 조건별 성능 비교
Week 9	745장 외부 이미지 평가	일반화 성능 검증
Week 10	크기별 탐지·실패 원인·상호보완성 분석	최종 연구 결론 도출

주의할 점은 Week 1~3의 초기 실험과 Week 4~8의 5-fold 실험, Week 9~10의 외부 데이터 실험이 모두 같은 평가집합은 아니라는 것이다. 따라서 각 단계의 결과를 그대로 이어 붙여 하나의 연속적인 성능 향상 곡선처럼 표현하면 안 된다.

3. 데이터셋 구축 및 전처리 과정
3.1 데이터 수집
프로젝트 초기에는 Roboflow와 공개 도로 손상 데이터를 활용하여 포트홀 이미지 및 라벨 데이터를 수집하였다.

초기 데이터는 약 600장 이상의 이미지로 구성되었으며, 일부 데이터는 Pascal VOC 형식의 XML annotation을 사용하였다.

초기 Week 1 보고서에는 약 7:2:1 비율의 Train/Validation/Test 분할을 기록하였다. 이후 보다 안정적인 모델 비교를 위해 598장 데이터의 5-fold 교차검증 실험으로 발전시켰다.

최종적으로 연구에서 사용한 평가 데이터는 크게 두 가지이다.
구분	내부 평가 데이터	외부 평가 데이터
이미지 수	598장	745장
정답 포트홀 수	1,586개	803개
주요 목적	모델 학습 및 비교	새로운 환경의 일반화 검증
사용 모델	YOLOv8, YOLO+CNN, RT-DETR	YOLOv8s, RT-DETR
평가 방식	5-fold outer test	기존 5개 체크포인트 반복 적용

외부 평가의 745장은 5개 체크포인트마다 다시 사용한 동일 이미지이며, 3,725개의 서로 다른 외부 이미지가 아니다.
3.2 Annotation 형식 변환
기존 XML 데이터에 저장된 바운딩 박스는 일반적으로 다음 형식을 사용한다.
\[
(x_{\min},y_{\min},x_{\max},y_{\max})
\]

이를 YOLO 학습에서 사용되는 중심 좌표 형식으로 변환하였다.
\[
x_c=\frac{x_{\min}+x_{\max}}{2W}
\]

\[
y_c=\frac{y_{\min}+y_{\max}}{2H}
\]

\[
w=\frac{x_{\max}-x_{\min}}{W},\quad
h=\frac{y_{\max}-y_{\min}}{H}
\]

여기서 \(W,H\)는 이미지 너비와 높이이다.

변환된 좌표는 이미지 크기에 대한 상대 좌표이므로 일반적으로 0~1 범위를 갖는다.

최종 라벨은 다음 형식으로 구성하였다.
class_id center_x center_y width height


포트홀 객체 탐지를 단순화하기 위해 핵심 탐지 대상은 0: pothole 단일 클래스로 지정하였다.
3.3 라벨 검증 및 정제
데이터 전처리 과정에서는 단순 파일 형식 변환뿐 아니라 수동 검수를 수행하였다.

검수 대상은 다음과 같다.
- 실제 포트홀이 아닌 영역이 포트홀로 표시된 사례
- 잘못된 바운딩 박스 위치 및 크기
- 이미지 밖으로 벗어난 좌표
- 클래스 정보 불일치
- 중복되거나 연결되지 않은 이미지·라벨
- 객체 경계에 비해 지나치게 넓은 정답 영역
특히 도로의 그림자, 맨홀, 보수 흔적 등은 포트홀과 시각적으로 혼동될 수 있으므로 정답 라벨을 함께 확인하도록 하였다.

이러한 검증 절차는 모델의 탐지 성능을 정확하게 평가하기 위한 필수 과정이다.
3.4 도로 환경에 특화된 데이터 증강
Week 1~2에서는 도로 영상 특성을 반영한 데이터 증강을 적용하거나 비교하였다.
증강 기법	기록된 설정	목적
Vertical Flip	flipud=0.0	실제 차량 시점과 다른 상하 반전 억제
HSV Brightness	hsv_v=0.4	밝기·조명 변화 대응
Random Erasing	erasing=0.4	부분 가림 환경 모사
Mosaic	close_mosaic=10	학습 후반 Mosaic 비활성화
Weather Augmentation	별도 실험	날씨·그림자 변화 대응

이러한 증강의 의도는 단일 촬영 환경에 모델이 과도하게 적응하는 것을 방지하고 다양한 환경의 포트홀 특징을 학습하도록 하는 것이다.

단, 날씨 증강의 성능 향상은 해당 초기 실험 조건에서 확인된 것이며, 외부 데이터에서 날씨별 성능 향상의 인과관계까지 입증한 것은 아니다.
4. 모델 구조 및 탐지 원리
이 프로젝트에서 중요한 모델은 YOLOv8, YOLOv8+CNN, RT-DETR이다. 세 모델은 모두 도로 영상에서 포트홀을 찾기 위한 것이지만, 객체를 탐지하는 방식과 특징을 처리하는 구조에 차이가 있다.
4.1 YOLOv8 객체 탐지 모델
YOLO(You Only Look Once)는 입력 이미지를 한 번의 신경망 추론 과정으로 처리하여 객체의 위치와 클래스를 예측하는 One-Stage Object Detection 계열 모델이다.

본 프로젝트에서는 YOLOv8n, YOLOv8s, YOLOv8m을 실험하였으며, 후반부 비교에서는 YOLOv8s를 기준 모델로 사용하였다.

YOLOv8의 주요 구조는 크게 Backbone, Neck, Detection Head로 구분된다.
Input Image

도로 영상 · 640 / 800 / 1024 해상도 실험

Backbone

Convolution · C2f · SPPF

포트홀의 경계, 질감, 형태 등 특징 추출

Neck

다중 스케일 특징 융합

작은 객체와 큰 객체의 특징 정보 결합

Detection Head

Bounding Box · Class Score

위치와 포트홀 점수 예측

Detection Results

후보 필터링 및 NMS 후 최종 포트홀 박스 출력


Backbone은 이미지에서 객체를 구분할 수 있는 특징을 추출한다. 초기 계층에서는 경계와 색상 변화 같은 저수준 특징을, 깊은 계층에서는 포트홀의 형태 및 복합적인 특징을 학습한다.

Neck은 서로 다른 크기의 특징맵을 결합한다. 원거리의 작은 포트홀과 가까운 위치의 큰 포트홀을 모두 탐지할 수 있도록 다중 스케일 정보를 활용한다.

Detection Head는 바운딩 박스의 좌표와 클래스 점수를 출력한다. 이후 Confidence Threshold와 NMS 등의 후처리 조건에 따라 최종 검출 결과가 결정된다.

프로젝트에서 YOLOv8s를 주요 기준 모델로 사용한 이유는 경량 모델의 빠른 추론 특성을 활용하면서 CNN 결합 방식과 RT-DETR을 비교하기 위해서였다.
4.2 YOLOv8+CNN 2-Stage Cascade 구조
Week 3~5에서는 YOLOv8이 생성한 포트홀 후보 영역을 별도의 CNN 분류기로 재판별하는 2단계 구조를 개발하였다.

이 구조의 핵심 아이디어는 YOLO가 포트홀로 의심되는 영역을 먼저 찾고, CNN이 해당 영역의 시각적 특징을 다시 검사하는 것이다.
             [도로 이미지 입력]
                     │
                     ▼
         ┌───────────────────────┐
         │ Stage 1: YOLOv8       │
         │ 포트홀 후보 영역 검출   │
         └───────────────────────┘
                     │
                     ▼
            Candidate Bounding Boxes
                     │
                     ▼
             ROI Crop / Resize
                     │
                     ▼
         ┌───────────────────────┐
         │ Stage 2: CropCNN      │
         │ 포트홀 / 배경 분류     │
         └───────────────────────┘
                     │
                     ▼
              CNN Probability
                     │
                     ▼
           YOLO-CNN Score Fusion
                     │
                     ▼
          Threshold 적용 및 결과 출력

          YOLOv8은 바운딩 박스와 Confidence Score를 생성한다. CNN은 YOLO가 출력한 바운딩 박스 안의 이미지를 잘라 입력받으며, 해당 영역이 포트홀인지 배경인지 판별한다.

후반부 정식 5-fold 실험에서 사용한 결합 점수는 다음과 같다.
\[
S_{\text{final}}
=
S_{\text{YOLO}}\times
\max(P_{\text{CNN}},10^{-8})^\alpha
\]

여기서
- \(S_{\text{YOLO}}\): YOLO가 예측한 후보 점수
- \(P_{\text{CNN}}\): CNN이 출력한 포트홀 클래스 확률
- \(\alpha\): CNN 점수를 최종 결과에 반영하는 정도
이다.

\(\alpha=0\)이면 CNN 점수의 영향이 없어져 기존 YOLO 점수를 그대로 사용한다. \(\alpha\)가 증가하면 CNN이 출력한 확률이 최종 점수에 미치는 영향이 커진다.

이때 중요한 사실은 CropCNN이 새로운 바운딩 박스를 생성하거나 기존 박스 좌표를 직접 수정하지 않는다는 점이다.

CNN은 YOLO가 생성한 후보를 재평가한다. 따라서 YOLO가 후보를 전혀 생성하지 못한 포트홀을 CNN만으로 새롭게 검출할 수는 없다.
4.3 CNN 결합 실험의 초기 문제와 수정
Week 3 초기 구현에서는 CNN 학습용 라벨을 생성하는 방법에 문제가 있었다.

초기 코드는 YOLO의 Confidence Score가 0.5 이상이면 포트홀, 미만이면 배경으로 지정하는 방식이었다.
category = "pothole" if conf_score >= 0.50 else "background"



이 방법은 YOLO의 기존 판단을 그대로 CNN의 학습 정답으로 사용하는 문제가 있었다.

결과적으로 CNN은 실제 이미지의 포트홀 특징을 학습하기보다 YOLO가 낮은 Confidence를 부여한 영역을 배경으로 판단하도록 편향될 수 있었다.

초기 실험에서는 558개의 YOLO 후보 가운데 CNN이 1개만 승인하는 극단적인 결과가 나타났다. 이 결과를 높은 성능의 오탐 제거라고 해석할 수는 없으며, 정상 포트홀까지 제거되는 Recall 붕괴 문제가 있었다.

이후 Ground Truth와의 IoU 기반 매칭으로 CNN 학습용 라벨을 변경하였다.

정답 박스와 후보 박스가 IoU 0.5 이상으로 겹치는지 확인하여 포트홀과 배경을 구분하도록 수정하였다.

수정 후 한 초기 실험에서는 558개 후보 중 123개를 최종 승인하였다. 다만 이 단계에서 제거된 후보 435개를 모두 실제 오탐으로 간주할 수는 없으며, 최종적인 성능 우열은 이후 5-fold 평가 결과로 판단해야 한다.

이 과정은 모델 구조뿐 아니라 학습 데이터에 사용하는 정답 라벨의 생성 방법도 탐지 시스템 전체 성능에 중요한 영향을 준다는 사실을 보여준다.
4.4 RT-DETR 객체 탐지 모델
RT-DETR(Real-Time Detection Transformer)은 CNN 기반 특징 추출과 Transformer 기반 특징 처리 및 객체 예측을 결합한 탐지 모델이다.

본 프로젝트에서는 RT-DETR-L을 사용하였다.

RT-DETR의 구조를 프로젝트 수준에서 개략적으로 표현하면 다음과 같다.
Input Road Image

CNN Backbone

다중 스케일 특징맵 생성

Hybrid Encoder

특징 간 관계 학습 및 다중 스케일 정보 융합

Query Selection

객체 후보를 표현할 Query 선택

Transformer Decoder

Query와 이미지 특징의 관계를 이용해 객체 예측 갱신

Object Predictions

Bounding Box · Confidence · Class


RT-DETR의 Encoder는 CNN에서 추출한 특징을 활용하여 다양한 위치와 스케일에 존재하는 이미지 정보 사이의 관계를 처리한다.

Transformer Decoder는 객체 후보를 표현하는 Query를 사용하여 객체의 위치와 클래스 예측을 반복적으로 갱신한다.

YOLOv8과 달리 RT-DETR은 Transformer 기반의 집합 예측 방식을 활용하며, 본 프로젝트의 추론 구현에서는 RT-DETR에 별도의 NMS를 추가하지 않았다.

일반적인 RT-DETR은 순수 ViT Backbone 모델과 구분해야 한다. RT-DETR은 CNN Backbone과 Transformer 구성요소가 결합된 구조이므로, 본 프로젝트의 결과를 순수 Vision Transformer Backbone의 독립적인 성능 검증으로 설명해서는 안 된다.
4.5 YOLOv8과 RT-DETR의 구조적 관계
비교 항목	YOLOv8s	RT-DETR-L
모델 계열	CNN 기반 One-Stage Detector	CNN·Transformer 기반 Detector
특징 추출	CNN Backbone	CNN Backbone
특징 처리	다중 스케일 특징 융합	Hybrid Encoder
객체 예측	Detection Head	Query 기반 Decoder
후처리	NMS 활용	별도 NMS 없이 사용
파라미터 수	약 1,114만	약 3,281만
예상되는 장점	경량 구조 및 빠른 추론	풍부한 특징 처리 및 객체 탐지
연구상 주의점	미검출과 Threshold 민감도	추가 오탐 및 더 큰 모델 규모

파라미터 수는 Week 7 실험에서 사용한 모델의 기록값이다.

두 모델의 성능 차이가 관찰되더라도 이를 Transformer 구조 하나의 효과로 단정할 수는 없다. Backbone, 파라미터 수, 학습 하이퍼파라미터 등이 함께 다르기 때문이다.

5. Week 1~5: YOLOv8 기반 탐지 성능 개선
5.1 Week 1 — 기준 모델 구축
Week 1에서는 데이터 수집, XML→YOLO TXT 변환, 정규화, 라벨 검수, 데이터 증강 설정, 기본 YOLOv8 학습을 진행하였다.

기준 모델은 YOLOv8s이며, 초기 학습 설정에는 이미지 크기 640, Batch Size 16, 50 Epoch, AdamW 및 Cosine Learning Rate Schedule 등이 기록되어 있다.

Week 1 보고서에 기록된 기준 성능은 다음과 같다.
지표	YOLOv8s 초기 결과
mAP50	77.8%
mAP50–95	49.1%
Precision	85.6%
Recall	73.2%

이는 초기 모델 구축 단계의 결과이다. 이후 진행한 5-fold 결과와 평가 절차가 다르므로 직접적인 성능 증가·감소 비교에 사용하지 않는다.

Week 1의 핵심 성과는 포트홀 데이터를 실제로 학습할 수 있도록 표준화하고, 기준 탐지 모델을 구성한 것이다.
5.2 Week 2 — 모델 규모 및 성능 최적화
Week 2에서는 YOLOv8n, YOLOv8s, YOLOv8m을 비교하였다.

n은 Nano, s는 Small, m은 Medium 크기를 의미하며, 모델의 규모와 학습·추론 특성이 다르다.

추가적으로 날씨 증강, 입력 해상도 조정, TTA(Test-Time Augmentation), Confidence 및 IoU Threshold 탐색을 진행하였다.

Week 2의 주요 실험 결과 중 하나는 날씨 증강 모델 비교이다.
모델	Precision	Recall	mAP50	mAP50–95
YOLOv8n + Weather Aug	82.70%	66.62%	71.47%	45.02%
YOLOv8s + Weather Aug	79.54%	67.16%	70.77%	47.06%
YOLOv8m + Weather Aug	74.71%	76.98%	76.96%	49.41%

이 실험에서 YOLOv8m은 높은 Recall 및 mAP를 보였지만 Precision은 다른 모델보다 낮았다.

따라서 모델의 규모를 확대하거나 증강을 적용하더라도 모든 성능 지표가 동시에 개선되는 것은 아니라는 점을 확인하였다.

또한 YOLOv8m에 1024 해상도와 TTA를 적용한 별도 실험에서는 Recall 76.07%, mAP50–95 50.14%가 기록되었다.

해당 실험에서는 고해상도와 TTA를 함께 적용한 설정이 비교 대상 설정보다 유리했으나, 이후 Week 8의 별도 데이터와 모델 실험에서는 1024 해상도가 항상 좋은 결과를 보이지 않았다.

결론적으로 Week 2에서는 모델 규모, 입력 해상도, 증강, 임계값이 Precision과 Recall의 절충 관계에 영향을 준다는 점을 확인하였다.
5.3 Week 3 — 미검출 분석 및 CNN 결합 구조 도입
Week 3에서는 높은 Precision을 달성하는 설정을 분석하고, 임계값을 높였을 때 증가하는 False Negative 문제를 조사하였다.

한 5-fold Confidence Threshold 실험의 결과는 다음과 같다.
Confidence	Precision	Recall	F1
0.10	80.94%	71.65%	76.01%
0.25	81.37%	71.24%	75.97%
0.50	85.80%	66.95%	75.23%
0.70	90.59%	58.82%	71.32%

Confidence Threshold가 높아질수록 더 확실한 예측만 허용하게 되므로, 해당 실험에서는 Precision이 상승하는 반면 Recall은 감소하였다.

또한 IoU 기반으로 미검출 후보를 조사하고, 이미지의 크기·밝기·대비 등에 따라 사례를 분류하였다.

주요 분석 후보에는 작은 객체, 어두운 영역, 높은 밝기, 낮은 대비, 복잡한 노면 질감 등이 포함되었다.

다만 밝기나 대비 기준을 만족했다고 해서 실제 신경망의 실패 원인이 해당 요소라고 입증되는 것은 아니다. 이 분류는 추가 이미지 검수를 위한 분석 기준으로 이해해야 한다.

이후 YOLOv8의 오탐을 줄이기 위해 후단 CNN을 결합하는 구조를 개발하였고, 초기 잘못된 학습 라벨 생성 방식을 IoU 기반 GT 매칭 방식으로 수정하였다.
5.4 Week 4~5 — YOLOv8과 YOLOv8+CNN 비교
Week 4~5에서는 초기 구조를 정비하고, 5-fold 교차검증 기반으로 YOLOv8 단독 모델과 CNN 점수 결합 모델을 비교하였다.

평가는 총 598장, 정답 포트홀 1,586개를 대상으로 진행하였다.

동일한 YOLOv8 가중치와 후보 박스를 공유하고, CNN을 추가했을 때 최종 점수와 탐지 결과가 어떻게 달라지는지 확인하였다.

결합 가중치 \(\alpha\)와 Confidence Threshold는 각 fold의 inner validation에서 선택하고 outer test에 고정하여 적용하였다.
Week 5 핵심 성능 결과
지표	YOLOv8	YOLOv8+CNN	변화
mAP50–95	45.161%	45.013%	−0.148%p
mAP50	72.216%	72.445%	+0.229%p
Precision	78.228%	76.310%	−1.918%p
Recall	64.250%	65.655%	+1.405%p
F1	70.415%	70.369%	−0.047%p
FP/image	0.4784	0.5472	+0.0688


지표	YOLOv8	YOLOv8+CNN
TP	1,017	1,039
FP	286	327
FN	569	547

CNN 결합 모델은 YOLOv8 단독 모델보다 22개의 정답 포트홀을 더 검출하고 미검출을 22개 줄였다.

그러나 오탐은 41개 증가하였다.

따라서 CNN 추가를 통해 Recall은 소폭 향상되었지만, Precision은 낮아졌으며 mAP50–95와 F1에서 실질적인 평균 개선은 확인되지 않았다.
CNN 추가에 따른 추론 속도 변화
설정	YOLOv8	YOLOv8+CNN
800, FP32	7.205 ms	9.463 ms
640, FP32	5.963 ms	7.933 ms
640, FP16	3.504 ms	5.875 ms

800 FP32 기준으로 CNN 추가에 따른 평균 지연시간은 약 2.258ms, 상대적인 지연시간 증가는 약 31.3%였다.

이 측정은 해당 Colab 실행 환경에서 전처리와 후처리를 포함하여 수행한 것으로, 다른 장비의 성능을 그대로 예측하는 수치는 아니다.

Week 5 결론

YOLOv8에 CNN을 추가하는 2-Stage Cascade는 Recall을 높일 수 있지만, 반드시 Precision과 종합적인 탐지 성능을 개선하는 것은 아니다.

후단 CNN이 추가되면 추론 과정도 복잡해지고 처리 시간이 증가한다. 따라서 이러한 구조를 적용하기 위해서는 오탐 감소나 Recall 개선이 추가 연산비용을 정당화할 만큼 충분한지 검증해야 한다.

이 결과를 바탕으로 Week 6부터는 후처리 CNN을 추가하는 방식이 아니라, 서로 다른 특징 처리 구조를 가진 RT-DETR을 비교 대상으로 도입하였다.

6. Week 6~8: RT-DETR 도입 및 모델 비교
6.1 Week 6 — Transformer 기반 객체 탐지 모델 학습 준비
Week 6에서는 기존 YOLOv8 및 YOLOv8+CNN 실험 이후, Transformer 기반 검출 모델인 RT-DETR-L의 구현과 학습을 진행하였다.

먼저 Vision Transformer의 기본 개념인 Patch Embedding, Token, Attention, Transformer 구조 등을 학습하고 RT-DETR의 객체 탐지 구조를 분석하였다.

Vision Transformer의 일반적인 처리 방식은 입력 이미지를 Patch 단위로 분할하여 Token으로 표현하고, Attention을 이용하여 이미지 영역 사이의 관계를 학습하는 것이다.

본 프로젝트에서는 이러한 Transformer 개념을 검출 모델에 적용한 RT-DETR을 이용하였으며, 기존 YOLOv8 모델과 동일한 데이터를 사용하도록 학습 환경을 구성하였다.

RT-DETR의 초기 시험 학습 조건은 다음과 같다.
항목	설정
모델	RT-DETR-L
초기 가중치	COCO 사전학습
입력 해상도	800 × 800
Batch Size	2
Optimizer	AdamW
초기 Learning Rate	0.0001
정밀도	FP32
GPU	NVIDIA Tesla T4
시험 학습	3 Epoch
본 학습 예산	최대 100 Epoch

학습 과정에서는 Checkpoint 저장 및 재개 기능도 검증하였다.

첫 Epoch가 종료된 후 학습을 중단하고, 저장된 Checkpoint의 Optimizer 상태를 포함해 2번째 Epoch부터 재개하여 총 3 Epoch의 시험 학습을 수행하였다.

이 과정은 Colab 환경에서 학습이 중단되더라도 이전 학습 상태를 복구할 수 있도록 하기 위한 것이다.
Week 6 시험 학습 결과
Epoch	GIoU Loss	Classification Loss	L1 Loss	mAP50
1	1.0210	2.0550	0.7494	14.90%
2	0.6280	0.7983	0.4108	59.90%
3	0.5512	0.6056	0.3737	62.60%

학습이 진행되면서 GIoU Loss, Classification Loss, L1 Loss가 감소하고 검증 AP가 상승하였다.

3 Epoch 시험 모델을 공통 COCO 평가기로 평가한 결과는 다음과 같다.
지표	결과
mAP50–95	41.399%
mAP50	62.478%
Precision	66.135%
Recall	58.657%
F1	62.172%
Small AP	16.013%
Medium AP	39.817%
Large AP	59.770%

단, 이 값은 96장의 inner validation에서 얻은 시험 학습 결과이다. 따라서 최종 5-fold outer test 성능과 직접 비교해서는 안 된다.

Week 6의 주요 성과는 모델의 성능 우열을 확정한 것이 아니라 RT-DETR의 학습·검증·Checkpoint 복구·공통 평가 절차가 정상적으로 작동하는지 확인한 것이다.
6.2 Week 7 — 동일한 5-fold 교차검증 구축
Week 7에서는 YOLOv8s와 RT-DETR-L을 비교하기 위해 동일한 5-fold 교차검증 환경을 사용하였다.

5-fold 교차검증은 데이터를 다섯 부분으로 나누어 각 부분을 한 번씩 평가에 사용하는 방식이다.

본 프로젝트에서는 각 outer fold에서 test 데이터를 분리하고, 나머지 데이터 안에 inner validation을 구성하였다.

각 Fold의 데이터 구성은 다음과 같다.
Fold	Train	Inner Validation	Outer Test
1	382	96	120
2	382	96	120
3	382	96	120
4	383	96	119
5	383	96	119

이렇게 하면 전체 598장 이미지가 5개 outer test에 한 번씩 포함된다.

연구에서는 모델 선택, Confidence Threshold 결정 등을 inner validation에서 수행하고, outer test에서는 결정된 설정을 변경하지 않도록 하였다.

이는 모델이 평가 데이터에 맞춰 과도하게 조정되어 실제 성능보다 높게 평가되는 문제를 줄이기 위한 절차이다.
모델 학습 및 평가 조건
구분	YOLOv8s	RT-DETR-L
데이터 분할	동일 5-fold	동일 5-fold
입력 크기	800	800
최대 Epoch	100	100
Train Batch	16	2
초기 Learning Rate	0.001	0.0001
파라미터	11,135,987	32,808,131
주요 평가 지표	mAP, P, R, F1	동일
모델 선택 기준	Inner Validation	동일 원칙

데이터와 평가 기준은 동일하지만 학습 Batch Size 및 Learning Rate까지 동일한 것은 아니다.

따라서 이 결과는 동일한 평가 환경에서 두 모델의 실험 설정을 비교한 것이며, 파라미터 수나 아키텍처 한 요소만 독립적으로 통제한 실험은 아니다.
6.3 Week 7 최종 모델 성능 비교
다음 값은 Week 7 보고서의 5-fold 평균이다.
지표	YOLOv8s	RT-DETR-L	RT-DETR − YOLO
mAP50–95	45.17%	53.34%	+8.17%p
mAP50	72.89%	80.44%	+7.55%p
mAP75	49.69%	59.69%	+10.01%p
Precision	78.26%	79.01%	+0.76%p
Recall	64.77%	74.63%	+9.86%p
F1	70.75%	76.51%	+5.76%p
AP Small	30.26%	35.84%	+5.58%p
FP/image	0.4766	0.5540	+0.0774

RT-DETR-L은 YOLOv8s에 비해 mAP50–95가 약 8.17%p, Recall이 약 9.86%p, F1이 약 5.76%p 높았다.

특히 Recall의 차이가 크게 나타났다는 것은 RT-DETR이 동일한 평가 데이터에서 더 많은 포트홀을 탐지했음을 의미한다.

하지만 RT-DETR의 FP/image는 0.5540으로 YOLOv8s의 0.4766보다 높았다.

즉, RT-DETR은 상대적으로 높은 객체 탐지 성능을 보였지만, 오탐 부담도 함께 증가하였다.
6.4 객체 크기별 성능 비교
Week 7에서는 Small, Medium, Large 객체의 AP도 분석하였다.
객체 크기	YOLOv8s AP	RT-DETR AP
Small	30.26%	35.84%
Medium	41.59%	47.39%
Large	57.57%	67.45%

두 모델 모두 작은 객체에서 AP가 낮고 큰 객체에서 AP가 높게 나타났다.

RT-DETR은 모든 크기 구간에서 더 높은 AP를 기록하였다.

이는 단순한 전체 평균 성능 차이를 넘어, 객체 크기별로도 RT-DETR의 우위가 관찰되었다는 점에서 의미가 있다.

다만 작은 포트홀에 대한 성능은 두 모델 모두 상대적으로 낮았기 때문에, 소형 포트홀 탐지는 후속 분석의 주요 과제로 선정하였다.
6.5 Week 8 — 오탐 수준을 고려한 Recall 비교
Week 8에서는 단순한 mAP나 F1 비교에서 더 나아가 허용 가능한 오탐 수준에서 실제 포트홀을 얼마나 검출하는가라는 문제를 분석하였다.

이를 위해 validation 데이터에서 FP/image 목표를 설정하고, 해당 목표에 맞춰 Confidence Threshold를 선택하였다.

사용한 FP/image 목표는 두 가지이다.
- 0.2: 이미지 한 장당 평균 오탐 0.2개를 목표로 하는 비교
- 0.5: 이미지 한 장당 평균 오탐 0.5개를 목표로 하는 비교
다만 이 목표는 validation에서 threshold를 선택하기 위한 기준이며, 실제 test에서 동일한 FP/image가 보장되는 것은 아니다.
기준 해상도 800의 Pooled Outer-Test 결과
모델	Validation FP 목표	Recall	Small Recall	실제 FP/image
YOLOv8s	0.2	52.40%	32.97%	0.1555
RT-DETR	0.2	66.02%	46.22%	0.2207
YOLOv8s	0.5	63.75%	48.65%	0.4080
RT-DETR	0.5	74.46%	60.54%	0.5100

FP 목표가 0.2인 경우 RT-DETR은 전체 Recall에서 13.62%p, Small Recall에서 약 13.24%p 높았다.

FP 목표가 0.5인 경우에도 RT-DETR은 전체 Recall에서 약 10.72%p, Small Recall에서 약 11.89%p 높았다.

이는 RT-DETR이 validation에서 동일한 오탐 목표를 기준으로 threshold를 선택했을 때 YOLOv8s보다 많은 포트홀을 탐지했다는 결과이다.

그러나 실제 outer test의 FP/image는 모델별로 차이가 있기 때문에 완전히 동일한 test 오탐 수준에서 RT-DETR의 Recall이 더 높다고 결론 내릴 수는 없다.

이를 검증하려면 실제 FP/image와 Recall의 관계를 나타내는 곡선에서 동일한 오탐 수준을 기준으로 비교해야 한다.
6.6 Week 8 — 입력 해상도별 성능 비교
추가적으로 입력 해상도를 640, 800, 1024로 변경하여 성능을 비교하였다.

FP 목표 0.2의 결과는 다음과 같다.
모델	해상도	Recall	Small Recall	실제 FP/image
YOLOv8s	640	55.49%	30.54%	0.1555
YOLOv8s	800	52.40%	32.97%	0.1555
YOLOv8s	1024	48.49%	32.97%	0.1522
RT-DETR	640	63.56%	41.62%	0.2258
RT-DETR	800	66.02%	46.22%	0.2207
RT-DETR	1024	59.96%	40.54%	0.2508

Week 8 해상도별 전체 Recall
Validation FP 목표 0.2, 단위 %


YOLOv8sRT-DETR

40%47.5%55%62.5%70%6408001024














RT-DETR은 기준 해상도 800에서 가장 높은 전체 Recall과 Small Recall을 기록하였다.

반면 1024 해상도로 증가시켰을 때 전체 Recall과 Small Recall이 모두 감소하였다.

YOLOv8s에서는 640 해상도가 전체 Recall 측면에서 높았지만, Small Recall은 800과 1024에서 더 높았다.

따라서 입력 해상도를 증가시킨다고 항상 탐지 성능이 개선되는 것은 아니며, 최적 해상도는 모델 구조와 평가 조건에 따라 달라질 수 있다.

추가로 이 실험에서는 해상도별 validation 결과에 따라 threshold도 다시 선택했기 때문에, 결과 차이는 해상도 변화와 threshold 재선택 효과를 함께 포함한다.
6.7 Validation 기반 해상도 선택 정책
Week 8에서는 입력 크기를 항상 800으로 사용하는 기준 정책과, 각 Fold의 validation Small Recall이 가장 높은 해상도를 선택하는 정책도 비교하였다.

RT-DETR에서 validation 기반 해상도 선택 정책을 사용한 결과는 다음과 같다.
FP 목표	정책	전체 Recall	Small Recall
0.2	기준 800	66.02%	46.22%
0.2	Validation 선택	63.18%	44.59%
0.5	기준 800	74.46%	60.54%
0.5	Validation 선택	72.32%	57.03%

Validation을 기준으로 해상도를 선택했더라도 outer test에서 더 나은 결과가 보장되지는 않았다.

이는 제한된 검증 데이터에서 좋은 결과를 보인 설정이 다른 평가 데이터에서도 반드시 우수한 것은 아니라는 사실을 보여준다.

따라서 본 프로젝트에서는 해상도 800을 주요 기준 설정으로 유지하고, 검증 데이터에 따른 정책 선택 효과는 별도의 분석으로 구분하였다.

Week 6~8 종합 결론

RT-DETR-L은 YOLOv8s보다 높은 mAP와 Recall을 기록하였으며, 특히 작은 포트홀 탐지에서 상대적인 우위를 나타냈다.

그러나 FP/image도 높았고, 입력 해상도 변경이나 validation 기반 해상도 선택이 항상 성능을 개선하지는 않았다.

따라서 모델의 정확도를 하나의 지표로만 판단하기보다, 탐지율·오탐 수준·객체 크기·운용 설정을 종합적으로 평가해야 한다는 결론에 도달하였다.

7. Week 9~10: 외부 한국 도로 데이터 일반화 성능 검증
7.1 외부 평가의 목적
Week 9~10의 연구 목적은 기존 내부 데이터로 학습한 YOLOv8s와 RT-DETR 모델이 새로운 한국 도로 이미지에서도 포트홀을 탐지할 수 있는지를 확인하는 것이었다.

기존 학습 및 검증 데이터에서 높은 성능을 기록한 모델도 새로운 촬영 환경에서는 성능이 저하될 수 있다.

특히 도로 촬영 위치, 카메라 특성, 날씨, 노면 상태, 포트홀 크기와 형태 등은 학습 당시와 다를 수 있다.

따라서 모델 개발 과정에서 사용하지 않은 새로운 데이터를 이용해 탐지 성능을 평가하였다.

이 실험의 핵심은 외부 이미지에 맞춰 모델을 추가 학습하거나 Confidence Threshold를 다시 조정하지 않는다는 것이다.

기존 Week 7 실험에서 결정한 모델 가중치와 validation 기반 threshold를 그대로 사용하여 외부 데이터 평가를 수행하였다.

이를 통해 새로운 데이터에 대한 모델의 일반화 성능을 보다 독립적으로 확인하고자 하였다.
7.2 외부 데이터 평가 구성
항목	설정
데이터	한국 도로 외부 이미지
이미지 수	745장
정답 포트홀 수	803개
비교 모델	YOLOv8s, RT-DETR
체크포인트	모델별 5개
FP/image 목표	0.2 및 0.5
입력 해상도	800 × 800
IoU 평가 기준	0.5
추론 Batch Size	1
정밀도	FP32
추가 학습	없음
Threshold 재조정	없음

5개 체크포인트와 2개 모델, 2개 Threshold 정책을 조합하여 총 20개의 체크포인트별 평가 결과가 생성되었다.

여기서 745장은 고유한 외부 이미지 수이며, 동일한 외부 이미지를 각각의 체크포인트로 반복 평가하였다.
7.3 외부 데이터 평가 파이프라인
프로젝트의 외부 평가 프로그램은 다음과 같은 구조로 구성하였다.
          [외부 도로 이미지 + JSON 라벨]
                        │
                        ▼
             데이터 파일 목록 구성
                   inventory()
                        │
                        ▼
           이미지와 JSON 연결 관계 검증
          filename_binding() / validate_bindings()
                        │
                        ▼
              포트홀 정답 라벨 파싱
                   parse_label()
                        │
                        ▼
                데이터 무결성 검사
                     prepare()
                        │
                        ▼
         기존 Week 7 모델과 Threshold 복원
                   load_week7()
                        │
               ┌────────┴────────┐
               ▼                 ▼
          YOLOv8s 추론      RT-DETR 추론
               │                 │
               └────────┬────────┘
                        ▼
              예측 Bounding Box 생성
                        │
                        ▼
             Confidence Threshold 적용
                        │
                        ▼
                 IoU 기반 매칭
                     match()
                        │
                        ▼
               TP / FP / FN 계산
                        │
                        ▼
           Precision / Recall / F1 / AP
                        │
                        ▼
            크기별 성능 및 실패 분석
                        │
                        ▼
               CSV / 그래프 저장

               7.4 데이터 무결성 검증
외부 평가에서 가장 먼저 수행한 작업은 이미지와 정답 JSON 파일의 연결 관계 확인이었다.

외부 데이터에서는 실제 이미지 파일 이름과 JSON 내부의 이미지 파일 이름이 완전히 일치하지 않는 경우가 있었다.

이를 해결하기 위해 날짜, 시간, 날씨, 프레임 번호 등을 확인하여 이미지와 JSON의 연결 관계를 검증하도록 구성하였다.

또한 단순한 파일 개수 검사를 넘어 다음 항목들을 확인하였다.
1. 이미지와 JSON 파일의 일대일 연결 관계
2. 동일한 이미지 및 JSON 중복 여부
3. 이미지 크기와 JSON 기록의 일치 여부
4. 포트홀 클래스 이름
5. Bounding Box 좌표 형식 및 유효 범위
6. 면적이 0이거나 음수인 박스
7. 동일 픽셀 이미지의 중복 여부
8. 파일의 SHA-256 해시 및 실험 기록
외부 JSON 데이터의 바운딩 박스는 설정된 xywh 픽셀 좌표 형식을 기준으로 파싱하였다.
\[
x_2=x+w,\qquad y_2=y+h
\]

이렇게 변환한 박스가 이미지 영역 안에 존재하는지 확인하였다.

잘못된 좌표나 누락된 라벨을 임의로 정상 데이터로 처리하지 않고 검증 오류로 분류하도록 하였다.

이 과정은 모델 성능뿐 아니라 평가 결과의 신뢰성을 확보하기 위한 중요한 개발 단계이다.
7.5 기존 가중치와 Threshold 고정
모델 평가 시 Week 7의 protocol.json, decisions_locked.json, completed.json 등 실험 기록을 불러왔다.

모델 Checkpoint의 SHA-256 해시를 비교하여 이전 실험과 동일한 가중치가 사용되는지도 확인하였다.

또한 기존 inner validation에서 설정한 Confidence Threshold를 외부 데이터에 그대로 적용하였다.

이 정책의 목적은 외부 데이터의 정답을 확인하면서 모델 설정을 다시 조정하는 평가 편향을 줄이는 것이다.

단, 기존 validation에서 선택한 FP/image 목표가 외부 데이터에서 실제로 달성되는 것은 보장되지 않는다.
7.6 외부 데이터 전체 성능 비교
외부 데이터 745장과 정답 포트홀 803개를 대상으로 한 평가 결과는 다음과 같다.

표의 Recall, Precision 및 F1은 5개 체크포인트의 평균이다.
모델	FP 목표	Recall	Precision	F1
YOLOv8s	0.2	37.41%	69.97%	48.28%
RT-DETR	0.2	52.73%	64.66%	58.01%
YOLOv8s	0.5	50.61%	54.33%	52.20%
RT-DETR	0.5	64.86%	49.45%	55.80%

FP 목표 0.2에서는 RT-DETR의 Recall이 YOLOv8s보다 15.32%p 높았다.

FP 목표 0.5에서도 RT-DETR의 Recall이 약 14.25%p 높게 나타났다.

반면 두 FP 목표 모두 Precision은 YOLOv8s가 더 높았다.

이 결과는 RT-DETR이 더 많은 포트홀을 찾아내는 대신 잘못된 탐지 결과도 상대적으로 더 많이 발생시켰음을 보여준다.
외부 데이터 AP 비교
지표	YOLOv8s	RT-DETR	차이
AP50	49.07%	58.99%	+9.92%p
AP50–95	24.20%	32.48%	+8.28%p
RT-DETR은 AP50과 AP50–95에서 모두 YOLOv8s보다 높은 성능을 보였다.

이는 외부 평가 데이터에서도 RT-DETR의 상대적인 탐지 성능 우위가 관찰되었다는 것을 의미한다.

다만 AP는 여러 Confidence 수준의 정밀도·재현율 관계를 요약한 지표이므로, 특정 운영 Threshold에서의 Recall과는 구분해서 해석해야 한다.
7.7 외부 데이터 오탐 수준 비교
모델	FP 목표 0.2에서 실제 FP/image	FP 목표 0.5에서 실제 FP/image
YOLOv8s	0.182	0.468
RT-DETR	0.314	0.743
RT-DETR은 두 목표에서 모두 YOLOv8s보다 높은 FP/image를 기록하였다.

특히 FP 목표가 0.2인 조건에서 RT-DETR의 실제 외부 FP/image는 0.314로 나타나 목표를 초과하였다.

따라서 RT-DETR의 높은 Recall만으로 해당 모델이 실제 도로 서비스 환경에 무조건 더 적합하다고 판단할 수 없다.

오탐이 발생하면 실제 포트홀이 없는 상황에서 잘못된 경고가 전달될 수 있기 때문이다.

이 연구에서 중요한 결론은 외부 데이터에서 RT-DETR의 탐지 성능 우위가 관찰되었지만, 높은 Recall은 더 많은 오탐과 함께 나타났다는 것이다.
7.8 기상 조건별 성능 분석
Week 9~10에서는 외부 이미지의 기상 조건에 따른 Recall과 FP/image도 분석하였다.

다음은 FP 목표 0.5 조건에서의 결과이다.
기상 조건	이미지 수	YOLO Recall	RT-DETR Recall
맑음	666장	48.98%	63.69%
흐림	60장	69.09%	74.24%
우천	19장	48.18%	74.55%

RT-DETR은 맑음, 흐림, 우천으로 구분한 모든 조건에서 YOLOv8s보다 높은 Recall을 기록하였다.

그러나 기상 조건별 이미지 수가 크게 다르다.

특히 우천 데이터는 19장에 불과하므로 이 결과만으로 RT-DETR이 모든 우천 도로 환경에서 안정적인 성능을 보인다고 일반화해서는 안 된다.

기상 조건과 촬영 위치, 도로 표면, 포트홀 크기 등의 요소도 함께 영향을 줄 수 있다.
7.9 외부 데이터의 객체 크기별 Recall
초기 외부 평가에서는 원본 이미지 기준 \(32\times32\) 픽셀 미만의 COCO Small 객체가 존재하지 않아 해당 기준의 Small Recall을 계산할 수 없었다.

이에 따라 Week 9~10 후속 분석에서는 전체 803개의 정답 포트홀을 이미지 대비 상대적인 Bounding Box 면적에 따라 세 그룹으로 분류하였다.
크기 구분	기준	객체 수
Small	상대 면적 하위 25%	201개
Medium	중간 50%	401개
Large	상대 면적 상위 25%	201개

이 구분은 COCO 공식 Small/Medium/Large 기준과 다르다.
FP 목표 0.2의 상대 크기별 Recall
크기	YOLOv8s	RT-DETR	RT-DETR 우위
Small	27.76%	43.58%	+15.82%p
Medium	37.06%	52.72%	+15.66%p
Large	47.76%	61.89%	+14.13%p
