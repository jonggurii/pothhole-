"""Validation-locked FP/image comparison for single-class pothole detection.

No training. Requires the original five-fold train/inner-val/outer-test lists
and the corresponding local YOLO and RT-DETR checkpoints. See the notebook.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

VERSION = "1.1.0"
PREDICTION_POLICY = "retain finite zero-area predictions as FP; reject inverted boxes, nonfinite values and invalid scores"
MODELS = ("yolo", "rtdetr")
SPLITS = ("train", "inner_val", "outer_test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REJECT_ALL = float(np.nextafter(1.0, np.inf))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remap_path(value, cfg, base=None):
    s = str(value)
    for old, new in sorted(cfg.get("path_remap", {}).items(), key=lambda x: -len(x[0])):
        old = old.rstrip("/")
        if s == old or s.startswith(old + "/"):
            s = str(new).rstrip("/") + s[len(old):]
            break
    p = Path(s).expanduser()
    if not p.is_absolute():
        if base is None:
            raise ValueError(f"절대 경로가 필요합니다: {value}")
        p = Path(base) / p
    return p.resolve()


def fold_number(path):
    found = {int(m.group(1)) for part in Path(path).parts
             if (m := re.fullmatch(r"fold[_-]?(\d+)", part, re.I))}
    if len(found) > 1:
        raise ValueError(f"경로에 서로 다른 fold가 있습니다: {path}")
    return next(iter(found), None)


def discover(cfg):
    """Read-only inventory. Never chooses the latest/first run from duplicates."""
    rows = []
    for model in MODELS:
        root_value = cfg.get(f"{model}_root", "")
        if not root_value:
            print(f"{model}_root를 실제 실행 폴더로 설정하세요.")
            continue
        root = remap_path(root_value, cfg)
        if not root.is_dir():
            print(f"현재 보이지 않는 폴더: {root}")
            continue
        for p in sorted(root.rglob("best.pt")):
            fold = fold_number(p)
            yaml_paths = []
            for parent in p.parents:
                if parent == root.parent:
                    break
                candidate = parent / "data.yaml"
                if candidate.is_file():
                    yaml_paths.append(str(candidate))
            rows.append({"model": model, "fold": fold, "weights": str(p),
                         "data_yaml_candidates": json.dumps(yaml_paths, ensure_ascii=False)})
    return pd.DataFrame(rows, columns=["model", "fold", "weights", "data_yaml_candidates"])


def inspect_metadata(root, limit=15):
    """Print only schema hints when an old manifest needs an explicit adapter."""
    paths = sorted(Path(root).rglob("dataset_manifest.json"))[:limit]
    for p in paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        print(str(p))
        if isinstance(obj, dict):
            print("  top-level keys:", list(obj)[:30])
            for key, value in list(obj.items())[:12]:
                hint = f"len={len(value)}" if isinstance(value, (dict, list)) else type(value).__name__
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    hint += f", first-item keys={list(value[0])[:20]}"
                print(" ", key, hint)
        elif isinstance(obj, list):
            print("  list length:", len(obj), "; first-item keys:",
                  list(obj[0]) if obj and isinstance(obj[0], dict) else "not a mapping")


def resolve_folds(cfg):
    inventory = discover(cfg)
    resolved = []
    overrides = cfg.get("fold_overrides", {})
    for fold in range(1, int(cfg.get("folds", 5)) + 1):
        override = overrides.get(fold, overrides.get(str(fold), {}))
        row = {"fold": fold}
        for model in MODELS:
            key = f"{model}_weights"
            if override.get(key):
                p = remap_path(override[key], cfg)
            else:
                candidates = inventory[(inventory.model == model) & (inventory.fold == fold)]
                if len(candidates) != 1:
                    raise ValueError(f"fold{fold} {model}: best.pt 후보 {len(candidates)}개. "
                                     f"fold_overrides[{fold}]['{key}']에 정확한 경로를 지정하세요.\n"
                                     + candidates.to_string(index=False))
                p = Path(candidates.iloc[0].weights)
            if not p.is_file():
                raise FileNotFoundError(p)
            if fold_number(p) not in (None, fold):
                raise ValueError(f"fold 번호 불일치: fold{fold}, {p}")
            row[key] = str(p)
        if not cfg.get("split_csv"):
            if override.get("data_yaml"):
                row["data_yaml"] = str(remap_path(override["data_yaml"], cfg))
            else:
                weights = Path(row["yolo_weights"])
                root = remap_path(cfg["yolo_root"], cfg)
                candidates = []
                for parent in weights.parents:
                    if parent == root.parent:
                        break
                    p = parent / "data.yaml"
                    if p.is_file():
                        candidates.append(p)
                if len(candidates) != 1:
                    raise ValueError(f"fold{fold}: data.yaml 후보={list(map(str, candidates))}. "
                                     "fold_overrides에 data_yaml을 지정하거나 split_csv를 지정하세요.")
                row["data_yaml"] = str(candidates[0])
        resolved.append(row)
    return resolved


def label_for(image):
    parts = list(Path(image).parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"images→labels 경로를 결정할 수 없습니다: {image}. "
                         "label 열이 있는 split_csv를 사용하세요.")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def expand_images(value, root, cfg):
    out = []
    for item in value if isinstance(value, list) else [value]:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"이미지 목록/폴더 경로가 필요합니다: {item!r}")
        p = remap_path(item, cfg, root)
        if p.is_dir():
            out.extend(sorted(x.resolve() for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS))
        elif p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                if Path(line).is_absolute():
                    q = remap_path(line, cfg)
                elif line.startswith("./"):
                    q = remap_path(line, cfg, p.parent)
                else:
                    # Explicitly reject ambiguous relative paths instead of guessing.
                    options = {remap_path(line, cfg, b) for b in (p.parent, root)}
                    existing = [x for x in options if x.is_file()]
                    if len(existing) != 1:
                        raise ValueError(f"목록의 상대 경로를 확정할 수 없습니다: {p}: {line}. "
                                         "이미지 목록을 절대 경로로 수정하거나 path_remap을 지정하세요.")
                    q = existing[0]
                if not q.is_file():
                    raise FileNotFoundError(q)
                out.append(q)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
        else:
            raise FileNotFoundError(f"분할 경로가 없습니다: {p}. 기존 data.yaml의 경로와 path_remap을 확인하세요.")
    if not out or len(out) != len(set(out)):
        raise ValueError(f"분할이 비어 있거나 중복 경로가 있습니다: {value}")
    return out


def read_split_rows(folds, cfg):
    """Supported inputs: original per-fold YAML train/val/test, or explicit CSV."""
    raw, sources = [], []
    if cfg.get("split_csv"):
        p = remap_path(cfg["split_csv"], cfg)
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        required = {"fold", "split", "image", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"split_csv 필수 열: {sorted(required)}; 선택 열: group")
        for r in df.to_dict("records"):
            split = {"val": "inner_val", "test": "outer_test"}.get(r["split"], r["split"])
            if split not in SPLITS:
                raise ValueError(f"알 수 없는 split: {split}")
            if not r["label"] or not r["image"]:
                raise ValueError("image/label 경로를 비워 둘 수 없습니다. 배경 이미지에도 빈 라벨 파일이 필요합니다.")
            raw.append({"fold": int(r["fold"]), "split": split,
                        "image": str(remap_path(r["image"], cfg, p.parent)),
                        "label": str(remap_path(r["label"], cfg, p.parent)),
                        "group": r.get("group", "")})
        sources.append({"path": str(p), "sha256": sha256_file(p)})
    else:
        for fold in folds:
            p = Path(fold["data_yaml"])
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or any(not data.get(k) for k in ("train", "val", "test")):
                raise ValueError(f"{p}에 train/val/test가 모두 있어야 합니다. 여기서 test는 해당 fold의 "
                                 "outer_test여야 합니다. 원래 별도 test 67장을 넣으면 안 됩니다. "
                                 "기존 dataset_manifest.json의 분할을 CSV로 명시할 수도 있습니다. "
                                 "폴더를 새로 나누거나 파일명으로 추정하지 않습니다.")
            root = remap_path(data.get("path", str(p.parent)), cfg, p.parent)
            sources.append({"path": str(p), "sha256": sha256_file(p)})
            for key, split in zip(("train", "val", "test"), SPLITS):
                for img in expand_images(data[key], root, cfg):
                    raw.append({"fold": fold["fold"], "split": split,
                                "image": str(img), "label": str(label_for(img)), "group": ""})
    if {r["fold"] for r in raw} != {f["fold"] for f in folds}:
        raise ValueError("분할 목록의 fold 집합과 모델 fold 집합이 다릅니다.")
    return raw, sources


def decode_rgb(path):
    with Image.open(path) as im:
        if im.getexif().get(274, 1) not in (None, 1):
            raise ValueError(f"EXIF 회전이 있는 이미지: {path}. 원래 라벨 좌표와 방향을 확인하세요.")
        return np.array(im.convert("RGB"))


def load_gt(path, width, height, tolerance=1e-6):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"라벨 파일이 없습니다: {p}. 누락을 배경으로 간주하지 않습니다.")
    boxes, corrected = [], 0
    for line_no, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        a = np.asarray([float(x) for x in line.split()], dtype=float)
        if len(a) != 5 or not np.isfinite(a).all() or a[0] != 0:
            raise ValueError(f"class=0인 YOLO bbox 라벨 5열이 필요합니다: {p}:{line_no}")
        _, cx, cy, w, h = a
        b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        if w <= 0 or h <= 0 or b.min() < -tolerance or b.max() > 1 + tolerance:
            raise ValueError(f"유효하지 않은 GT bbox: {p}:{line_no}: {line}")
        clipped = b.clip(0, 1)
        corrected += int(not np.array_equal(b, clipped))
        if np.any(clipped[2:] <= clipped[:2]):
            raise ValueError(f"면적이 0인 GT bbox: {p}:{line_no}")
        boxes.append((clipped * [width, height, width, height]).tolist())
    return boxes, corrected


def prepare(folds, cfg):
    raw, sources = read_split_rows(folds, cfg)
    cache, rows, boundary_corrections = {}, [], 0
    for r in raw:
        key = (r["image"], r["label"])
        if key not in cache:
            rgb = decode_rgb(r["image"])
            h, w = rgb.shape[:2]
            identity = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            gt, fixes = load_gt(r["label"], w, h)
            cache[key] = {"image_id": identity, "width": w, "height": h,
                          "gt": gt, "label_sha256": sha256_file(r["label"])}
            boundary_corrections += fixes
        rows.append({**r, **cache[key]})
    by_fold = {}
    canonical = {}
    for row in rows:
        old = canonical.setdefault(row["image_id"], row)
        if not np.array_equal(np.asarray(old["gt"]), np.asarray(row["gt"])):
            raise ValueError(f"동일 pixel 이미지에 서로 다른 GT가 있습니다: {row['image']}")
    all_groups = [bool(r["group"]) for r in rows]
    if any(all_groups) and not all(all_groups):
        raise ValueError("group은 모든 행에 지정하거나 모두 비워 두세요.")
    if all(all_groups):
        for r in rows:
            if canonical[r["image_id"]]["group"] != r["group"]:
                raise ValueError("동일 이미지의 group 값이 fold마다 다릅니다.")
    pools, tests, summaries = [], [], []
    for fold in folds:
        k = fold["fold"]
        split_rows = {s: [r for r in rows if r["fold"] == k and r["split"] == s] for s in SPLITS}
        identities = {}
        for split, items in split_rows.items():
            ids = {r["image_id"] for r in items}
            if not items or len(ids) != len(items):
                raise ValueError(f"fold{k} {split}: 빈 분할 또는 동일 pixel 중복 이미지")
            identities[split] = ids
            summaries.append({"fold": k, "split": split, "images": len(items),
                              "objects": sum(len(r["gt"]) for r in items)})
        for a, b in (("train", "inner_val"), ("train", "outer_test"), ("inner_val", "outer_test")):
            if identities[a] & identities[b]:
                raise ValueError(f"fold{k}: {a}/{b} 간 동일 이미지 누수")
            if all(all_groups) and ({r['group'] for r in split_rows[a]} & {r['group'] for r in split_rows[b]}):
                raise ValueError(f"fold{k}: {a}/{b} 간 촬영 group 누수")
        pools.append(set.union(*identities.values()))
        tests.extend(identities["outer_test"])
        by_fold[k] = split_rows
    if any(pool != pools[0] for pool in pools):
        raise ValueError("fold별 전체 이미지 pool이 다릅니다. 원래 5-fold 분할을 확인하세요.")
    if len(tests) != len(set(tests)) or set(tests) != pools[0]:
        raise ValueError("outer_test는 CV pool의 각 이미지를 정확히 한 번 포함해야 합니다. "
                         "공통 test 67장을 모든 fold에 지정하지 마세요.")
    n_images = len(canonical)
    n_objects = sum(len(r["gt"]) for r in canonical.values())
    for name, actual in (("expected_images", n_images), ("expected_objects", n_objects)):
        if cfg.get(name) is not None and cfg[name] != actual:
            raise ValueError(f"{name}: 기대값 {cfg[name]}, 실제 {actual}. 분할/데이터가 이전 실험과 같은지 확인하세요.")
    print(f"분할 검증 완료: {n_images} images, {n_objects} objects; 경계 미세 보정 {boundary_corrections}건")
    if not all(all_groups):
        print("촬영 group 미제공: 동일 pixel 누수만 검사했습니다. 인접 프레임/유사 장면 누수는 별도 확인이 필요합니다.")
    return {"by_fold": by_fold, "rows": rows, "summary": summaries, "sources": sources,
            "setup_sha256": digest({"folds": folds, "config": cfg}),
            "audit": {"images": n_images, "objects": n_objects,
                      "boundary_corrections": boundary_corrections, "groups_checked": all(all_groups)}}


def validate_prediction(pred, label="prediction"):
    boxes = np.asarray(pred["boxes"], dtype=float).reshape(-1, 4)
    scores = np.asarray(pred["scores"], dtype=float)
    if scores.ndim != 1 or len(boxes) != len(scores):
        raise ValueError(f"bbox/score 길이 불일치: {label}")
    if not np.isfinite(boxes).all():
        bad = np.flatnonzero(~np.isfinite(boxes).all(axis=1))[:5]
        raise ValueError(f"Nonfinite bbox: {label}; indices={bad.tolist()}, boxes={boxes[bad].tolist()}")
    if not np.isfinite(scores).all():
        bad = np.flatnonzero(~np.isfinite(scores))[:5]
        raise ValueError(f"Nonfinite score: {label}; indices={bad.tolist()}, scores={scores[bad].tolist()}")
    invalid_scores = (scores < 0) | (scores > 1)
    if invalid_scores.any():
        bad = np.flatnonzero(invalid_scores)[:5]
        raise ValueError(f"Score outside [0,1]: {label}; indices={bad.tolist()}, scores={scores[bad].tolist()}")
    inverted = np.any(boxes[:, 2:] < boxes[:, :2], axis=1)
    if inverted.any():
        bad = np.flatnonzero(inverted)[:5]
        raise ValueError(f"Inverted bbox: {label}; indices={bad.tolist()}, boxes={boxes[bad].tolist()}")
    # Zero width/height can result from clipping to the image boundary.
    # Keep the original box and confidence; it cannot match any GT and counts as FP.
    return boxes, scores


def prediction_quality(rows, predictions):
    records = []
    if {r['image_id'] for r in rows} != set(predictions):
        raise ValueError('예측 품질 기록의 이미지 집합이 분할과 다릅니다.')
    for row in rows:
        boxes, scores = validate_prediction(predictions[row['image_id']], row['image'])
        zero = np.any(boxes[:, 2:] == boxes[:, :2], axis=1)
        records.append({'image_id':row['image_id'],'image':row['image'],
                        'prediction_count':len(boxes),'zero_area_count':int(zero.sum()),
                        'zero_area_boxes':json.dumps(boxes[zero].tolist()),
                        'zero_area_scores':json.dumps(scores[zero].tolist())})
    return records


def iou_one(box, gt):
    xy1 = np.maximum(box[:2], gt[:, :2])
    xy2 = np.minimum(box[2:], gt[:, 2:])
    wh = np.maximum(0, xy2 - xy1)
    intersection = wh[:, 0] * wh[:, 1]
    area = np.prod(box[2:] - box[:2])
    areas = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-15)


def detection_events(rows, predictions, match_iou=0.5):
    """Score-ordered greedy matching to the best *unmatched* GT in each image.

    Matching prefixes do not change when lower-score detections are removed.
    Duplicate predictions count as FP. Negative images count in the denominator.
    Equal scores retain the detector's output order within each image.
    """
    if not 0 < match_iou <= 1:
        raise ValueError("match_iou must be in (0,1]")
    if len(rows) == 0 or {r["image_id"] for r in rows} != set(predictions):
        raise ValueError("예측 이미지 집합이 분할 이미지 집합과 다릅니다.")
    scores_out, tp_out, total_gt = [], [], 0
    for row in rows:
        boxes, scores = validate_prediction(predictions[row["image_id"]], row["image"])
        gt = np.asarray(row["gt"], dtype=float).reshape(-1, 4)
        if not np.isfinite(gt).all() or np.any(gt[:, 2:] <= gt[:, :2]):
            raise ValueError(f"Invalid GT: {row['image']}")
        used = np.zeros(len(gt), dtype=bool)
        total_gt += len(gt)
        for j in np.argsort(-scores, kind="stable"):
            hit = 0
            if len(gt) and np.all(boxes[j, 2:] > boxes[j, :2]):
                overlaps = iou_one(boxes[j], gt)
                overlaps[used] = -1
                best = int(np.argmax(overlaps))
                if overlaps[best] >= match_iou:
                    used[best], hit = True, 1
            scores_out.append(float(scores[j]))
            tp_out.append(hit)
    if total_gt == 0:
        raise ValueError("이 분할에는 GT가 없어 Recall을 정의할 수 없습니다.")
    return np.asarray(scores_out), np.asarray(tp_out, dtype=int), total_gt, len(rows)


def metrics(tp, fp, gt, images):
    tp, fp, gt, images = map(int, (tp, fp, gt, images))
    precision = tp / (tp + fp) if tp + fp else 0.0  # documented zero-prediction convention
    recall = tp / gt
    return {"tp": tp, "fp": fp, "fn": gt - tp, "gt": gt, "images": images,
            "recall": recall, "precision": precision,
            "precision_defined": bool(tp + fp),
            "f1": 2 * tp / (2 * tp + fp + gt - tp), "fp_per_image": fp / images}


def validation_curve(events):
    scores, labels, gt, images = events
    rows = [{"threshold": REJECT_ALL, **metrics(0, 0, gt, images)}]
    if len(scores):
        order = np.argsort(-scores, kind="stable")
        scores, labels = scores[order], labels[order]
        tp, fp = labels.cumsum(), (1 - labels).cumsum()
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
        for end in ends:
            rows.append({"threshold": float(scores[end]), **metrics(tp[end], fp[end], gt, images)})
    return pd.DataFrame(rows)


def select_threshold(curve, target, conf_floor):
    if not math.isfinite(target) or target < 0:
        raise ValueError("FP/image 목표는 유한한 0 이상의 값이어야 합니다.")
    images = int(curve.iloc[0].images)
    budget = math.floor(float(target) * images + 1e-9)
    feasible = curve[curve.fp <= budget]
    # Maximize recall; ties: fewer FP, then higher threshold. No interpolation or partial score ties.
    chosen = feasible.sort_values(["tp", "fp", "threshold"], ascending=[False, True, False],
                                  kind="stable").iloc[0].to_dict()
    for key in ("tp", "fp", "fn", "gt", "images"):
        chosen[key] = int(chosen[key])
    chosen["precision_defined"] = bool(chosen["precision_defined"])
    chosen.update(target_fp_per_image=float(target), allowed_val_fp=budget,
                  all_cached_candidates_fit_budget=bool(curve.iloc[-1].fp <= budget),
                  confidence_floor=float(conf_floor))
    return chosen


def evaluate_threshold(events, threshold):
    scores, labels, gt, images = events
    keep = scores >= threshold
    return metrics(labels[keep].sum(), keep.sum() - labels[keep].sum(), gt, images)


def environment(cfg):
    import torch
    import ultralytics
    required = cfg.get("required_ultralytics")
    if required and ultralytics.__version__ != required:
        raise RuntimeError(f"Ultralytics 현재={ultralytics.__version__}, 요구={required}. "
                           "노트북의 설치 셀을 실행한 뒤 런타임을 재시작하세요. "
                           "기존 실행의 실제 버전이 다르면 설정과 기록을 함께 수정하세요.")
    device = cfg.get("device", 0)
    gpu = str(device).lower() != "cpu"
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU가 없습니다. Colab 런타임 → 런타임 유형 변경 → T4 GPU를 선택하세요.")
    torch.manual_seed(43)
    if gpu:
        torch.cuda.manual_seed_all(43)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"python": sys.version, "ultralytics": ultralytics.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(int(device)) if gpu else None,
            "numpy": np.__version__, "pillow": importlib.metadata.version("Pillow")}


def inference_signature(cfg):
    return {"imgsz": int(cfg.get("imgsz", 800)), "conf": float(cfg.get("conf_floor", 0.001)),
            "iou": float(cfg.get("yolo_nms_iou", 0.7)), "max_det": int(cfg.get("max_det", 300)),
            "batch": 1, "quantize": 32, "rect": False, "augment": False, "classes": [0],
            "agnostic_nms": False}


def predict_split(model_name, weights, weight_sha, rows, cfg, env):
    import torch
    from ultralytics import YOLO, RTDETR
    settings = inference_signature(cfg)
    signature = {"code_version": VERSION, "code_sha256": sha256_file(__file__),
                 "model": model_name, "weight_sha256": weight_sha,
                 "prediction_policy": PREDICTION_POLICY,
                 "environment": env, "settings": settings,
                 "images": [{k: r[k] for k in ("image_id", "label_sha256", "width", "height")}
                            for r in rows]}
    key = digest(signature)
    cache = Path(cfg["output_root"]) / "prediction_cache" / f"{key}.json"
    if cache.is_file():
        obj = json.loads(cache.read_text(encoding="utf-8"))
        if obj["signature"] != signature:
            raise ValueError(f"예측 캐시 설정 불일치: {cache}")
        predictions = obj["predictions"]
        if set(predictions) != {r["image_id"] for r in rows}:
            raise ValueError(f"캐시에 누락된 이미지: {cache}")
        for image_id, pred in predictions.items():
            validate_prediction(pred, image_id)
        print(f"  캐시 사용: {model_name}, {len(rows)} images")
        return predictions, str(cache)
    model = None
    predictions = {}
    try:
        model = (YOLO if model_name == "yolo" else RTDETR)(str(weights))
        if len(model.names) != 1:
            raise ValueError(f"포트홀 단일 클래스 학습 가중치가 아닙니다: names={model.names}, {weights}")
        for index, row in enumerate(rows, 1):
            rgb = decode_rgb(row["image"])
            actual_id = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            if actual_id != row["image_id"]:
                raise ValueError(f"검증 이후 이미지가 변경되었습니다: {row['image']}")
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            result = model.predict(source=bgr, device=cfg.get("device", 0), verbose=False,
                                   save=False, **settings)[0]
            boxes = result.boxes
            if boxes is None:
                raise ValueError("bbox 결과가 없습니다. detect 체크포인트인지 확인하세요.")
            cls = boxes.cls.detach().cpu().numpy()
            if np.any(cls != 0):
                raise ValueError("예측 class id가 0이 아닙니다.")
            pred = {"boxes": boxes.xyxy.detach().cpu().numpy().astype(float).tolist(),
                    "scores": boxes.conf.detach().cpu().numpy().astype(float).tolist()}
            try:
                validate_prediction(pred, row["image"])
            except ValueError as exc:
                diagnostic = Path(cfg['output_root'])/'prediction_diagnostics'/f'{key}_{row["image_id"][:16]}.json'
                dump_json(diagnostic, {'model':model_name,'weights':str(weights),'image':row['image'],
                                      'settings':settings,'error':str(exc),
                                      'boxes_repr':repr(pred['boxes']),'scores_repr':repr(pred['scores'])})
                raise ValueError(f'{exc}\n진단 파일: {diagnostic}') from exc
            order = np.argsort(-np.asarray(pred["scores"]), kind="stable")[:settings["max_det"]]
            pred = {key: [values[int(i)] for i in order] for key, values in pred.items()}
            predictions[row["image_id"]] = pred
            if index % 20 == 0 or index == len(rows):
                print(f"  {model_name}: {index}/{len(rows)}")
        quality = prediction_quality(rows, predictions)
        total_zero = sum(r['zero_area_count'] for r in quality)
        if total_zero:
            print(f'  {model_name}: 면적 0인 후보 {total_zero}개 유지. threshold 이상이면 FP로 집계합니다.')
        dump_json(cache, {"signature": signature, "predictions": predictions, "quality": quality})
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions, str(cache)


def summarize(results):
    summaries, paired = [], []
    for (model, target), group in results.groupby(["model", "target_fp_per_image"], sort=True):
        row = {"model": model, "target_fp_per_image": target, "folds": len(group)}
        for metric in ("recall", "precision", "f1", "fp_per_image"):
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        pooled = metrics(group['tp'].sum(), group['fp'].sum(), group['gt'].sum(), group['images'].sum())
        row.update({"pooled_" + k: v for k, v in pooled.items()})
        row["test_budget_exceeded_folds"] = int(group.test_budget_exceeded.sum())
        summaries.append(row)
    for (fold, target), group in results.groupby(["fold", "target_fp_per_image"]):
        indexed = group.set_index("model")
        if len(indexed) != 2 or set(indexed.index) != set(MODELS):
            raise ValueError("paired 비교에는 각 fold·각 목표에 두 모델의 결과가 모두 필요합니다.")
        y, r = indexed.loc["yolo"], indexed.loc["rtdetr"]
        paired.append({"fold": fold, "target_fp_per_image": target,
                       "recall_delta_pp": 100 * (r.recall - y.recall),
                       "fp_per_image_delta": r.fp_per_image - y.fp_per_image,
                       "yolo_recall": y.recall, "rtdetr_recall": r.recall,
                       "yolo_fp_per_image": y.fp_per_image, "rtdetr_fp_per_image": r.fp_per_image})
    return pd.DataFrame(summaries), pd.DataFrame(paired)


def write_plots(out, summary, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"yolo": "#2563eb", "rtdetr": "#e15d36"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        rows = summary[summary.model == model].sort_values("target_fp_per_image")
        ax.plot(rows.pooled_fp_per_image, 100 * rows.pooled_recall, "o-", color=colors[model], label=model)
        for _, r in rows.iterrows():
            ax.annotate(f"val target={r.target_fp_per_image:g}",
                        (r.pooled_fp_per_image, 100 * r.pooled_recall), xytext=(5, 6),
                        textcoords="offset points", fontsize=8)
    ax.set(xlabel="Achieved outer-test FP / image", ylabel="Pooled outer-test recall (%)",
           title="Validation-locked operating points (IoU = 0.50)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "outer_test_operating_points.png", dpi=180)
    plt.close(fig)
    targets = sorted(summary.target_fp_per_image.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, model in enumerate(MODELS):
        rows = summary[summary.model == model].set_index("target_fp_per_image").loc[targets]
        x = np.arange(len(targets)) + (i - 0.5) * 0.35
        axes[0].bar(x, rows.recall_mean * 100, 0.35, yerr=rows.recall_sd * 100,
                    label=model, color=colors[model], capsize=4)
        axes[1].bar(x, rows.fp_per_image_mean, 0.35, yerr=rows.fp_per_image_sd,
                    label=model, color=colors[model], capsize=4)
    for ax in axes:
        ax.set_xticks(range(len(targets)), [f"{t:g}" for t in targets])
        ax.set_xlabel("Inner-validation target FP / image")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Outer-test recall (%), mean ± sample SD")
    axes[1].set_ylabel("Achieved outer-test FP / image, mean ± sample SD")
    fig.tight_layout()
    fig.savefig(out / "recall_and_actual_fp.png", dpi=180)
    plt.close(fig)
    pivot = paired.pivot(index="fold", columns="target_fp_per_image", values="recall_delta_pp")
    fig, ax = plt.subplots(figsize=(7, 4))
    limit = max(1, float(np.max(np.abs(pivot.to_numpy()))))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    color="white" if abs(value) > limit * .65 else "black")
    ax.set_xticks(range(len(pivot.columns)), [f"{t:g}" for t in pivot.columns])
    ax.set_yticks(range(len(pivot)), [f"Fold {i}" for i in pivot.index])
    ax.set(xlabel="Inner-validation target FP / image", title="Recall difference: RT-DETR − YOLO (pp)")
    fig.colorbar(im, ax=ax, label="Percentage points")
    fig.tight_layout()
    fig.savefig(out / "paired_recall_delta.png", dpi=180)
    plt.close(fig)


def run(cfg, folds=None, prepared=None):
    cfg = dict(cfg)
    targets = [float(t) for t in cfg.get("targets", [0.2, 0.5])]
    if not targets or len(set(targets)) != len(targets) or any(not math.isfinite(t) or t < 0 for t in targets):
        raise ValueError("targets는 중복 없는 유한한 0 이상 숫자 목록이어야 합니다.")
    if not 0 < float(cfg.get("conf_floor", .001)) < 1:
        raise ValueError("conf_floor는 0과 1 사이여야 합니다.")
    if float(cfg.get("match_iou", .5)) != .5:
        raise ValueError("이 실험은 IoU=0.50으로 고정합니다. 다른 IoU는 별도 실험으로 구현하세요.")
    folds = resolve_folds(cfg) if folds is None else folds
    prepared = prepare(folds, cfg) if prepared is None else prepared
    if prepared["setup_sha256"] != digest({"folds": folds, "config": cfg}):
        raise ValueError("분할 검증 후 설정이 변경되었습니다. prepare 셀부터 다시 실행하세요.")
    for source in prepared["sources"]:
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("분할 검증 후 원본 목록이 변경되었습니다. prepare 셀부터 다시 실행하세요.")
    env = environment(cfg)
    root = remap_path(cfg["output_root"], cfg)
    cfg["output_root"] = str(root)
    out = root / datetime.now(timezone.utc).strftime("fp_matched_%Y%m%d_%H%M%S_%f_UTC")
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(prepared["summary"]).to_csv(out / "split_summary.csv", index=False)
    pd.DataFrame([{k: r[k] for k in ("fold", "split", "image", "label", "image_id", "group", "label_sha256")}
                  for r in prepared["rows"]]).to_csv(out / "verified_splits.csv", index=False)
    weight_hashes = {(f["fold"], model): sha256_file(f[f"{model}_weights"])
                     for f in folds for model in MODELS}
    for model in MODELS:
        hashes = [weight_hashes[(f["fold"], model)] for f in folds]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{model}: 여러 fold에 동일 체크포인트가 지정되었습니다.")
    protocol = {"code_version": VERSION, "code_sha256": sha256_file(__file__),
                "config": cfg, "environment": env,
                "folds": folds, "split_sources": prepared["sources"], "data_audit": prepared["audit"],
                "weight_sha256": {f"fold{k}/{m}": h for (k, m), h in weight_hashes.items()},
                "inference_settings": inference_signature(cfg),
                "selection": "inner_val only; max TP under floor(target*n_images); ties min FP then max threshold",
                "matching": "class0, IoU>=0.5, score-ordered one-to-one greedy; no crowd/ignore labels",
                "prediction_policy": PREDICTION_POLICY,
                "postprocessing": "native YOLO NMS; native RT-DETR NMS-free; no added NMS",
                "limits": ["FP budget is empirical on inner_val, not guaranteed on outer_test",
                           "conf_floor and max_det limit candidate coverage",
                           "No training, AP, latency measurement or significance test",
                           "Same input size does not imply identical architecture-specific preprocessing",
                           "5-fold sample SD is descriptive; folds are not independent repeats"]}
    dump_json(out / "protocol.json", protocol)
    thresholds, cache_index = [], []
    # Complete and persist every validation decision BEFORE reading any test predictions.
    print("PHASE 1/2: inner validation에서만 threshold 결정")
    for fold in folds:
        k = fold["fold"]
        for model in MODELS:
            print(f"Fold {k}/{len(folds)} {model} inner_val")
            rows = prepared["by_fold"][k]["inner_val"]
            predictions, path = predict_split(model, fold[f"{model}_weights"], weight_hashes[k, model], rows, cfg, env)
            cache_index.append({"fold": k, "model": model, "split": "inner_val", "cache": path})
            events = detection_events(rows, predictions, .5)
            curve = validation_curve(events)
            curve.to_csv(out / f"val_curve_fold{k}_{model}.csv", index=False)
            for target in targets:
                selected = select_threshold(curve, target, cfg.get("conf_floor", .001))
                thresholds.append({"fold": k, "model": model, **selected})
                print(f"  target={target:g}; threshold={selected['threshold']:.6g}; "
                      f"val FP/img={selected['fp_per_image']:.4f}; recall={selected['recall']:.4f}")
                if selected["all_cached_candidates_fit_budget"]:
                    print("  모든 수집 후보가 FP 예산 이내: 더 낮은 score 후보는 수집되지 않았으므로 도달 가능한 Recall이 제한될 수 있습니다.")
    lock = {"protocol_sha256": digest(protocol), "thresholds": thresholds}
    lock["lock_sha256"] = digest(lock)
    dump_json(out / "thresholds_locked.json", lock)
    pd.DataFrame(thresholds).to_csv(out / "thresholds_inner_val.csv", index=False)
    results = []
    print("PHASE 2/2: 고정 threshold로 outer_test 평가")
    for fold in folds:
        k = fold["fold"]
        for model in MODELS:
            print(f"Fold {k}/{len(folds)} {model} outer_test")
            rows = prepared["by_fold"][k]["outer_test"]
            predictions, path = predict_split(model, fold[f"{model}_weights"], weight_hashes[k, model], rows, cfg, env)
            cache_index.append({"fold": k, "model": model, "split": "outer_test", "cache": path})
            events = detection_events(rows, predictions, .5)
            for selected in thresholds:
                if selected["fold"] == k and selected["model"] == model:
                    measured = evaluate_threshold(events, selected["threshold"])
                    target = selected["target_fp_per_image"]
                    results.append({"fold": k, "model": model, "target_fp_per_image": target,
                                    "threshold": selected["threshold"],
                                    "val_fp_per_image": selected["fp_per_image"],
                                    "val_recall": selected["recall"], **measured,
                                    "test_budget_exceeded": bool(measured["fp_per_image"] > target + 1e-12)})
            pd.DataFrame(results).to_csv(out / "outer_test_results_partial.csv", index=False)
    # No adaptation to test counts: thresholds above are the only operating points.
    if json.loads((out / "thresholds_locked.json").read_text(encoding="utf-8")) != lock:
        raise ValueError("평가 중 threshold 파일이 변경되었습니다.")
    results = pd.DataFrame(results)
    summary, paired = summarize(results)
    results.to_csv(out / "outer_test_by_fold.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_differences.csv", index=False)
    dump_json(out / "prediction_cache_index.json", cache_index)
    write_plots(out, summary, paired)
    lines = ["오탐 목표를 validation에서 고정한 성능 비교", "", "아래 값은 pooled outer-test 결과입니다.",
             "각 fold의 threshold는 다를 수 있습니다. 목표 FP/image는 validation의 제약이며 test 실측값과 다를 수 있습니다.", ""]
    for target in targets:
        rows = summary[summary.target_fp_per_image == target].set_index("model")
        y, r = rows.loc["yolo"], rows.loc["rtdetr"]
        lines.append(f"목표 {target:g} FP/image: YOLO Recall={100*y.pooled_recall:.2f}%, FP/img={y.pooled_fp_per_image:.4f}; "
                     f"RT-DETR Recall={100*r.pooled_recall:.2f}%, FP/img={r.pooled_fp_per_image:.4f}; "
                     f"Recall 차이={100*(r.pooled_recall-y.pooled_recall):+.2f} percentage points.")
    lines += ["", "두 모델의 실제 test FP/image를 함께 해석하세요. test 오탐이 서로 다르면 정확히 같은 오탐 조건의 우열로 단정하지 않습니다.",
              "본 비교는 설정된 후보 수/score 하한과 검증 표본에 조건부입니다. 기존 test 결과를 보고 목표나 threshold를 다시 선택하지 마세요.",
              "오차 막대는 fold 간 표본 표준편차이며 신뢰구간이나 통계적 유의성을 나타내지 않습니다."]
    (out / "interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
    dump_json(out / "completed.json", {"status": "complete", "threshold_lock_sha256": lock["lock_sha256"]})
    print("\n완료:", out)
    print("\n".join(lines))
    return {"out": out, "thresholds": pd.DataFrame(thresholds), "results": results,
            "summary": summary, "paired": paired}
