"""Validation-locked FP budgets + inference resolution / small-object study.

Reuses the original per-fold checkpoints. Does not retrain or select on test.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import math

import numpy as np
import pandas as pd
import week7_fp_matched as core

VERSION = '2.2.0'
SIZES = ('small', 'medium', 'large')


def safe_json(obj):
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return safe_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path, obj):
    core.dump_json(path, safe_json(obj))


def area_bins(boxes):
    """Disjoint bins in ORIGINAL pixels; never resize GT to assign size bins.

    small < 32^2; medium [32^2,96^2); large >=96^2.
    These are size-stratified Recall bins, not the COCO AP/AR evaluator.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    names = np.where(area < 32**2, 'small', np.where(area < 96**2, 'medium', 'large'))
    return area, names


def trace_split(rows, predictions):
    if {r['image_id'] for r in rows} != set(predictions) or not rows:
        raise ValueError('분할과 예측의 이미지 집합이 다릅니다.')
    items, all_scores, all_hits = [], [], []
    total_gt = 0
    for row in rows:
        boxes, scores = core.validate_prediction(predictions[row['image_id']], row['image'])
        gt = np.asarray(row['gt'], dtype=float).reshape(-1, 4)
        if not np.isfinite(gt).all() or np.any(gt[:, 2:] <= gt[:, :2]):
            raise ValueError(f"유효하지 않은 GT: {row['image']}")
        area, bins = area_bins(gt)
        used = np.zeros(len(gt), dtype=bool)
        gt_score = np.full(len(gt), -np.inf)
        pred_gt = np.full(len(boxes), -1, dtype=int)
        for index in np.argsort(-scores, kind='stable'):
            if len(gt) and np.all(boxes[index, 2:] > boxes[index, :2]):
                overlap = core.iou_one(boxes[index], gt)
                overlap[used] = -1
                g = int(overlap.argmax())
                if overlap[g] >= .5:
                    used[g], gt_score[g], pred_gt[index] = True, scores[index], g
        best_iou = np.zeros(len(gt))
        best_valid_score = np.full(len(gt), -np.inf)
        for g, box in enumerate(gt):
            if len(boxes):
                overlaps = core.iou_one(box, boxes)
                best_iou[g] = overlaps.max()
                if np.any(overlaps >= .5):
                    best_valid_score[g] = scores[overlaps >= .5].max()
        items.append({'row': row, 'boxes': boxes, 'scores': scores, 'pred_gt': pred_gt,
                      'gt_score': gt_score, 'area': area, 'bins': bins,
                      'best_iou': best_iou, 'best_valid_score': best_valid_score})
        all_scores.extend(scores.tolist())
        all_hits.extend((pred_gt >= 0).astype(int).tolist())
        total_gt += len(gt)
    if total_gt == 0:
        raise ValueError('전체 GT가 없어 Recall을 정의할 수 없습니다.')
    events = (np.asarray(all_scores), np.asarray(all_hits, dtype=int), total_gt, len(rows))
    return {'items': items, 'events': events}


def score_trace(trace, threshold, details=False):
    measured = core.evaluate_threshold(trace['events'], threshold)
    measured['zero_area_fp'] = 0
    counts = {name: [0, 0] for name in SIZES}
    gt_rows, image_rows = [], []
    for item in trace['items']:
        hit = item['gt_score'] >= threshold
        keep = item['scores'] >= threshold
        zero_area_fp = int((keep & np.any(item['boxes'][:, 2:] == item['boxes'][:, :2], axis=1)).sum())
        measured['zero_area_fp'] += zero_area_fp
        bins = item['bins']
        per_image = {'image_id': item['row']['image_id'], 'image': item['row']['image'],
                     'tp': int(hit.sum()), 'fp': int(keep.sum() - hit.sum()),
                     'fn': int(len(hit) - hit.sum()), 'zero_area_fp':zero_area_fp}
        for name in SIZES:
            subset = bins == name
            tp, n = int((hit & subset).sum()), int(subset.sum())
            counts[name][0] += tp
            counts[name][1] += n
            per_image.update({f'{name}_tp': tp, f'{name}_gt': n, f'{name}_fn': n-tp})
        if details:
            image_rows.append(per_image)
            for g in range(len(hit)):
                if hit[g]:
                    reason = 'detected'
                elif item['best_valid_score'][g] >= threshold:
                    reason = 'matching_competition'
                elif item['best_iou'][g] >= .5:
                    reason = 'below_threshold'
                elif item['best_iou'][g] > 0:
                    reason = 'overlap_below_iou'
                else:
                    reason = 'no_overlapping_cached_candidate'
                gt_rows.append({'image_id': item['row']['image_id'], 'image': item['row']['image'],
                                'gt_index': g, 'area_original_px2': float(item['area'][g]),
                                'size': str(bins[g]), 'detected': bool(hit[g]), 'reason': reason,
                                'max_iou_cached': float(item['best_iou'][g]),
                                'max_score_iou_ge_05': float(item['best_valid_score'][g])
                                if np.isfinite(item['best_valid_score'][g]) else None})
    for name, (tp, n) in counts.items():
        measured.update({f'{name}_tp': tp, f'{name}_gt': n, f'{name}_fn': n-tp,
                         f'{name}_recall': tp/n if n else None})
    return measured, gt_rows, image_rows


def choose_resolution(validation_records):
    """The only resolution-selection function. Accepts validation rows only."""
    frame = pd.DataFrame(validation_records)
    if not len(frame) or set(frame['split']) != {'inner_val'}:
        raise ValueError('해상도 선택에는 inner_val 결과만 전달해야 합니다.')
    decisions = []
    for (fold, model, target), group in frame.groupby(['fold', 'model', 'target_fp_per_image']):
        if group['small_gt'].nunique() != 1 or group['gt'].nunique() != 1:
            raise ValueError('해상도에 따라 GT 평가 대상이 달라졌습니다.')
        if (group['fp_per_image'] > target + 1e-12).any():
            raise ValueError('오탐 예산을 넘는 validation 후보가 있습니다.')
        use_small = int(group['small_gt'].iloc[0]) > 0
        keys = ['small_recall', 'recall', 'fp_per_image', 'imgsz'] if use_small else ['recall', 'fp_per_image', 'imgsz']
        ascending = [False, False, True, True] if use_small else [False, True, True]
        selected = group.sort_values(keys, ascending=ascending, kind='stable').iloc[0]
        decisions.append({'fold': int(fold), 'model': str(model), 'target_fp_per_image': float(target),
                          'selected_imgsz': int(selected['imgsz']), 'threshold': float(selected['threshold']),
                          'selection_objective': 'small_recall' if use_small else 'overall_recall_no_small_gt',
                          'val_small_gt': int(selected['small_gt']),
                          'val_small_recall': selected['small_recall'] if use_small else None,
                          'val_recall': float(selected['recall']),
                          'val_fp_per_image': float(selected['fp_per_image'])})
    return safe_json(decisions)


def summarize(frame, keys):
    records = []
    for values, group in frame.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row['folds'] = len(group)
        pooled = core.metrics(group['tp'].sum(), group['fp'].sum(), group['gt'].sum(), group['images'].sum())
        row.update({f'pooled_{k}': v for k, v in pooled.items()})
        row['pooled_zero_area_fp'] = int(group['zero_area_fp'].sum())
        for size in SIZES:
            tp, n = int(group[f'{size}_tp'].sum()), int(group[f'{size}_gt'].sum())
            row.update({f'pooled_{size}_tp': tp, f'pooled_{size}_gt': n,
                        f'pooled_{size}_fn': n-tp, f'pooled_{size}_recall': tp/n if n else None})
        for metric in ('recall', 'precision', 'f1', 'fp_per_image', 'small_recall', 'medium_recall', 'large_recall'):
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            row[f'{metric}_valid_folds'] = len(values)
            row[f'{metric}_mean'] = float(values.mean()) if len(values) else None
            row[f'{metric}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        row['test_budget_exceeded_folds'] = int(group['test_budget_exceeded'].sum())
        records.append(row)
    return pd.DataFrame(records)


def policy_tables(grid, decisions, baseline):
    rows, differences = [], []
    for decision in decisions:
        mask = (grid.fold == decision['fold']) & (grid.model == decision['model']) & (grid.target_fp_per_image == decision['target_fp_per_image'])
        group = grid[mask]
        base = group[group.imgsz == baseline]
        chosen = group[group.imgsz == decision['selected_imgsz']]
        if len(base) != 1 or len(chosen) != 1:
            raise ValueError('baseline 또는 validation 선택 해상도의 test 결과가 누락되었습니다.')
        base, chosen = base.iloc[0].to_dict(), chosen.iloc[0].to_dict()
        for policy, entry in [('baseline', base), ('val_selected', chosen)]:
            rows.append({**entry, 'policy': policy})
        difference = {k: decision[k] for k in ('fold', 'model', 'target_fp_per_image', 'selected_imgsz', 'selection_objective')}
        difference.update(baseline_imgsz=baseline,
                          recall_delta_pp=100*(chosen['recall']-base['recall']),
                          fp_per_image_delta=chosen['fp_per_image']-base['fp_per_image'],
                          small_fn_reduction=base['small_fn']-chosen['small_fn'],
                          small_gt=base['small_gt'])
        difference['small_recall_delta_pp'] = (100*(chosen['small_recall']-base['small_recall'])
                                                if base['small_gt'] else None)
        differences.append(difference)
    policy = pd.DataFrame(rows)
    model_pairs = []
    for (fold, target, name), group in policy.groupby(['fold', 'target_fp_per_image', 'policy']):
        indexed = group.set_index('model')
        y, r = indexed.loc['yolo'], indexed.loc['rtdetr']
        model_pairs.append({'fold': fold, 'target_fp_per_image': target, 'policy': name,
                            'yolo_imgsz': int(y.imgsz), 'rtdetr_imgsz': int(r.imgsz),
                            'recall_delta_pp_rtdetr_minus_yolo': 100*(r.recall-y.recall),
                            'small_recall_delta_pp_rtdetr_minus_yolo': 100*(r.small_recall-y.small_recall) if y.small_gt else None,
                            'fp_per_image_delta_rtdetr_minus_yolo': r.fp_per_image-y.fp_per_image})
    return policy, pd.DataFrame(differences), pd.DataFrame(model_pairs)


def plot_results(out, grid_summary, policy_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'yolo': '#2563eb', 'rtdetr': '#e15d36'}
    targets = sorted(grid_summary.target_fp_per_image.unique())
    for target in targets:
        data = grid_summary[grid_summary.target_fp_per_image == target]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for model in core.MODELS:
            rows = data[data.model == model].sort_values('imgsz')
            for ax, metric, label, scale in zip(axes,
                    ['pooled_recall', 'pooled_small_recall', 'pooled_fp_per_image'],
                    ['Overall recall (%)', 'Small-object recall (%)', 'Actual test FP / image'], [100, 100, 1]):
                ax.plot(rows.imgsz, pd.to_numeric(rows[metric], errors='coerce')*scale,
                        'o-', color=colors[model], label=model)
                ax.set(xlabel='Inference image size', ylabel=label)
                ax.set_xticks(sorted(data.imgsz.unique()))
                ax.grid(alpha=.2)
            if rows.pooled_small_gt.sum() == 0:
                axes[1].text(.05, .5, 'No small GT: recall is undefined', transform=axes[1].transAxes)
        axes[2].axhline(target, color='gray', linestyle='--', label='validation budget')
        for ax in axes:
            ax.legend()
        fig.suptitle(f'Validation FP/image target = {target:g} | pooled outer-test results')
        fig.tight_layout()
        fig.savefig(out/f'resolution_target_{target:g}.png', dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for model in core.MODELS:
        for policy, style in [('baseline', '--'), ('val_selected', '-')]:
            rows = policy_summary[(policy_summary.model == model) & (policy_summary.policy == policy)].sort_values('target_fp_per_image')
            for ax, metric in zip(axes, ['pooled_recall', 'pooled_small_recall']):
                ax.plot(rows.pooled_fp_per_image, pd.to_numeric(rows[metric], errors='coerce')*100,
                        marker='o', linestyle=style, color=colors[model], label=f'{model} {policy}')
                ax.grid(alpha=.2)
                ax.set_xlabel('Achieved outer-test FP / image')
    axes[0].set_ylabel('Overall recall (%)')
    axes[1].set_ylabel('Small-object recall (%)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Baseline vs validation-selected resolution; thresholds locked before test')
    fig.tight_layout()
    fig.savefig(out/'policy_operating_points.png', dpi=170)
    plt.close(fig)


def case_candidates(gt_status, decisions, baseline, target):
    cases = []
    for decision in decisions:
        if decision['target_fp_per_image'] != target:
            continue
        selected = decision['selected_imgsz']
        subset = gt_status[(gt_status.fold == decision['fold']) & (gt_status.model == decision['model']) &
                           (gt_status.target_fp_per_image == target) & (gt_status['size'] == 'small')]
        a = subset[subset.imgsz == baseline]
        b = subset[subset.imgsz == selected]
        joined = a.merge(b, on=['image_id', 'gt_index'], suffixes=('_baseline', '_selected'), validate='one_to_one')
        for r in joined.to_dict('records'):
            old, new = bool(r['detected_baseline']), bool(r['detected_selected'])
            category = 'recovered' if new and not old else 'regressed' if old and not new else 'still_missed' if not new else None
            if category:
                cases.append({'fold': decision['fold'], 'model': decision['model'], 'target_fp_per_image': target,
                              'baseline_imgsz': baseline, 'selected_imgsz': selected, 'image_id': r['image_id'],
                              'image': r['image_baseline'], 'gt_index': r['gt_index'], 'category': category,
                              'baseline_reason': r['reason_baseline'], 'selected_reason': r['reason_selected'],
                              'baseline_threshold': r['threshold_baseline'], 'selected_threshold': r['threshold_selected']})
    columns = ['fold', 'model', 'target_fp_per_image', 'baseline_imgsz', 'selected_imgsz', 'image_id', 'image',
               'gt_index', 'category', 'baseline_reason', 'selected_reason', 'baseline_threshold', 'selected_threshold']
    return pd.DataFrame(cases, columns=columns)


def render_cases(out, cases, rows_by_fold, cache_index, limit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    (out/'cases').mkdir(exist_ok=True)
    index = {(r['fold'], r['model'], r['imgsz']): r['cache'] for r in cache_index if r['split'] == 'outer_test'}
    loaded = {}
    produced = []
    # deterministic illustrative sample; never used for resolution or threshold selection
    for category in ['recovered', 'regressed', 'still_missed']:
        subset = cases[cases.category == category].sort_values(['model', 'fold', 'image_id', 'gt_index'])
        subset = subset.drop_duplicates(['model', 'image_id']).head(limit)
        for c in subset.to_dict('records'):
            row = next(r for r in rows_by_fold[c['fold']]['outer_test'] if r['image_id'] == c['image_id'])
            rgb = core.decode_rgb(row['image'])
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            for ax, label, size, threshold in zip(axes, ['Baseline', 'Validation-selected'],
                    [c['baseline_imgsz'], c['selected_imgsz']], [c['baseline_threshold'], c['selected_threshold']]):
                key = (c['fold'], c['model'], size)
                if key not in loaded:
                    loaded[key] = json.loads(Path(index[key]).read_text(encoding='utf-8'))['predictions']
                prediction = loaded[key][row['image_id']]
                trace = trace_split([row], {row['image_id']: prediction})
                item = trace['items'][0]
                metrics, _, _ = score_trace(trace, threshold)
                ax.imshow(rgb)
                for g, box in enumerate(row['gt']):
                    x1, y1, x2, y2 = box
                    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                          edgecolor='#ffd000' if g == c['gt_index'] else '#30c878', linewidth=2))
                for j in np.flatnonzero(item['scores'] >= threshold):
                    x1, y1, x2, y2 = item['boxes'][j]
                    color = '#39b7ff' if item['pred_gt'][j] >= 0 else '#ff4040'
                    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.5, linestyle='--'))
                    ax.text(x1, y1, f"{item['scores'][j]:.2f}", color='white', fontsize=7,
                            bbox={'facecolor': color, 'alpha': .65, 'pad': .5})
                ax.set_title(f"{label} imgsz={size}, conf>={threshold:.4g}\nTP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} | small TP/GT={metrics['small_tp']}/{metrics['small_gt']}", fontsize=10)
                ax.axis('off')
            fig.suptitle(f"{c['model']} fold {c['fold']} | {category} | GT #{c['gt_index']} | validation FP target={c['target_fp_per_image']:g}")
            fig.text(.5, .01, 'GT: green; focus small GT: yellow; prediction TP: blue dashed; prediction FP: red dashed', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .035, 1, .94))
            name = f"{category}_{c['model']}_fold{c['fold']}_{c['image_id'][:12]}.png"
            fig.savefig(out/'cases'/name, dpi=150)
            plt.close(fig)
            produced.append({**c, 'file': f'cases/{name}'})
    pd.DataFrame(produced, columns=[*cases.columns, 'file']).to_csv(out/'case_images.csv', index=False)
    return produced


def run(cfg, folds=None, prepared=None):
    cfg = dict(cfg)
    resolutions = cfg.get('resolutions', [640, 800, 1024])
    baseline = cfg.get('baseline_imgsz', 800)
    targets = [float(t) for t in cfg.get('targets', [.2, .5])]
    if not resolutions or any(type(s) is not int or s <= 0 or s % 32 for s in resolutions) or len(set(resolutions)) != len(resolutions) or baseline not in resolutions:
        raise ValueError('resolutions는 중복 없는 양의 32배수 정수 목록이며 baseline_imgsz를 포함해야 합니다.')
    if not targets or len(set(targets)) != len(targets) or any(not math.isfinite(t) or t < 0 for t in targets):
        raise ValueError('targets는 중복 없는 유한한 0 이상 값이어야 합니다.')
    if not 0 < cfg.get('conf_floor', .001) < 1 or cfg.get('match_iou', .5) != .5:
        raise ValueError('0<conf_floor<1, match_iou=0.5 조건이 필요합니다.')
    limit = cfg.get('case_images_per_category', 3)
    case_target = cfg.get('case_target', targets[-1])
    if type(limit) is not int or limit < 0 or case_target not in targets:
        raise ValueError('사례 수는 0 이상의 정수이며 case_target은 targets에 포함되어야 합니다.')
    folds = core.resolve_folds(cfg) if folds is None else folds
    prepared = core.prepare(folds, cfg) if prepared is None else prepared
    if prepared['setup_sha256'] != core.digest({'folds': folds, 'config': cfg}):
        raise ValueError('분할 검증 후 설정이 바뀌었습니다. prepare 셀부터 다시 실행하세요.')
    for source in prepared['sources']:
        if core.sha256_file(source['path']) != source['sha256']:
            raise ValueError('분할 파일이 변경되었습니다. prepare 셀부터 다시 실행하세요.')
    env = core.environment(cfg)
    root = core.remap_path(cfg['output_root'], cfg)
    cfg['output_root'] = str(root)
    out = root/datetime.now(timezone.utc).strftime('fp_resolution_%Y%m%d_%H%M%S_%f_UTC')
    out.mkdir(parents=True, exist_ok=False)
    (out/'code').mkdir()
    for source in (Path(core.__file__), Path(__file__)):
        (out/'code'/source.name).write_bytes(source.read_bytes())
    if cfg.get('recovery_audit'):
        audit_path = Path(cfg['recovery_audit'])
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        recovery_code = audit_path.parent/'week7_split_recovery.py'
        if core.sha256_file(recovery_code) != audit['code_sha256']:
            raise ValueError('분할 복구 코드의 해시가 기록과 다릅니다.')
        (out/'recovery_audit.json').write_bytes(audit_path.read_bytes())
        (out/'code'/recovery_code.name).write_bytes(recovery_code.read_bytes())
    hashes = {(f['fold'], model): core.sha256_file(f[f'{model}_weights']) for f in folds for model in core.MODELS}
    for model in core.MODELS:
        if len({hashes[f['fold'], model] for f in folds}) != len(folds):
            raise ValueError(f'{model}: 같은 체크포인트가 여러 fold에 지정되었습니다.')
    protocol = {'version': VERSION, 'code_sha256': core.sha256_file(__file__),
                'core_sha256': core.sha256_file(core.__file__), 'config': cfg, 'environment': env,
                'folds': folds, 'split_sources': prepared['sources'], 'data_audit': prepared['audit'],
                'weights_sha256': {f'fold{k}/{m}': h for (k,m),h in hashes.items()},
                'inference_by_resolution': {str(s): core.inference_signature({**cfg, 'imgsz': s}) for s in resolutions},
                'prediction_policy': core.PREDICTION_POLICY,
                'threshold_rule': 'inner_val: max overall TP under integer FP budget; tie min FP then max threshold',
                'resolution_rule': 'inner_val: max small recall; tie max overall recall, min FP/image, min imgsz; no-small fallback overall recall',
                'size_rule': 'ORIGINAL bbox area: small<1024; medium[1024,9216); large>=9216; not COCO AP/AR',
                'limits': ['No retraining, AP or latency measurement', 'Outer-test budgets are not guaranteed',
                           'No extra NMS for RT-DETR; architecture-native preprocessing/postprocessing',
                           'Small-object strata are fixed across resolutions; global FP includes all predictions',
                           'Scores below conf_floor or beyond max_det are unobserved',
                           'All-grid outer results are descriptive; final policy is validation-selected',
                           'A resolution policy includes recalibration of its confidence threshold',
                           'Do not interpret fold sample SD as independent-trial significance']}
    save_json(out/'protocol.json', protocol)
    pd.DataFrame(prepared['summary']).to_csv(out/'split_summary.csv', index=False)
    pd.DataFrame([{k:r[k] for k in ('fold','split','image','label','image_id','group','label_sha256')} for r in prepared['rows']]).to_csv(out/'verified_splits.csv', index=False)
    val_records, cache_index, quality_records = [], [], []
    print('PHASE 1: 모든 모델·해상도의 inner validation 평가 / threshold 선택')
    for fold in folds:
        k = fold['fold']
        for model in core.MODELS:
            for size in resolutions:
                print(f'Fold {k}/{len(folds)} {model} imgsz={size} inner_val')
                local_cfg = {**cfg, 'imgsz': size}
                pred, cache = core.predict_split(model, fold[f'{model}_weights'], hashes[k, model], prepared['by_fold'][k]['inner_val'], local_cfg, env)
                cache_index.append({'fold': k, 'model': model, 'imgsz': size, 'split': 'inner_val', 'cache': cache})
                quality_records.extend({'fold':k,'model':model,'imgsz':size,'split':'inner_val',**r}
                                       for r in core.prediction_quality(prepared['by_fold'][k]['inner_val'],pred))
                pd.DataFrame(quality_records).to_csv(out/'prediction_quality.csv',index=False)
                trace = trace_split(prepared['by_fold'][k]['inner_val'], pred)
                curve = core.validation_curve(trace['events'])
                curve.to_csv(out/f'val_curve_fold{k}_{model}_{size}.csv', index=False)
                for target in targets:
                    selected = core.select_threshold(curve, target, cfg.get('conf_floor', .001))
                    stats, _, _ = score_trace(trace, selected['threshold'])
                    val_records.append({'fold': k, 'model': model, 'imgsz': size, 'split': 'inner_val', **selected, **stats})
                    if selected['all_cached_candidates_fit_budget']:
                        print(f'  target={target:g}: 모든 수집 후보가 FP 예산 이내입니다. conf_floor/max_det 하한·상한에 의해 추가 Recall이 제한될 수 있습니다.')
                pd.DataFrame(val_records).to_csv(out/'validation_grid_partial.csv', index=False)
    decisions = choose_resolution(val_records)
    lock = safe_json({'protocol_sha256': core.digest(protocol), 'thresholds': val_records, 'resolution_selection': decisions})
    lock['lock_sha256'] = core.digest(lock)
    save_json(out/'decisions_locked.json', lock)
    pd.DataFrame(val_records).to_csv(out/'validation_grid.csv', index=False)
    pd.DataFrame(decisions).to_csv(out/'selected_resolution_inner_val.csv', index=False)
    print('threshold와 해상도 선택 저장 완료. 이제 outer test를 평가합니다.')
    print(pd.DataFrame(decisions).to_string(index=False))
    grid_records, gt_records, image_records = [], [], []
    print('PHASE 2: 고정된 설정으로 outer-test 평가; 결과를 보고 재선택하지 않음')
    for fold in folds:
        k = fold['fold']
        for model in core.MODELS:
            for size in resolutions:
                print(f'Fold {k}/{len(folds)} {model} imgsz={size} outer_test')
                pred, cache = core.predict_split(model, fold[f'{model}_weights'], hashes[k, model], prepared['by_fold'][k]['outer_test'], {**cfg, 'imgsz': size}, env)
                cache_index.append({'fold': k, 'model': model, 'imgsz': size, 'split': 'outer_test', 'cache': cache})
                quality_records.extend({'fold':k,'model':model,'imgsz':size,'split':'outer_test',**r}
                                       for r in core.prediction_quality(prepared['by_fold'][k]['outer_test'],pred))
                pd.DataFrame(quality_records).to_csv(out/'prediction_quality.csv',index=False)
                trace = trace_split(prepared['by_fold'][k]['outer_test'], pred)
                for selected in val_records:
                    if (selected['fold'], selected['model'], selected['imgsz']) != (k, model, size):
                        continue
                    target = selected['target_fp_per_image']
                    meta = {'fold':k,'model':model,'imgsz':size,'target_fp_per_image':target,'threshold':selected['threshold']}
                    stats, gt_detail, image_detail = score_trace(trace, selected['threshold'], details=True)
                    grid_records.append({**meta, **stats, 'val_fp_per_image':selected['fp_per_image'],
                                         'test_budget_exceeded': stats['fp_per_image'] > target+1e-12})
                    gt_records.extend({**meta, **r} for r in gt_detail)
                    image_records.extend({**meta, **r} for r in image_detail)
                pd.DataFrame(grid_records).to_csv(out/'outer_grid_partial.csv', index=False)
    if json.loads((out/'decisions_locked.json').read_text(encoding='utf-8')) != lock:
        raise ValueError('평가 도중 고정한 결정 파일이 변경되었습니다.')
    grid, gt_status, per_image = pd.DataFrame(grid_records), pd.DataFrame(gt_records), pd.DataFrame(image_records)
    grid_summary = summarize(grid, ['model','imgsz','target_fp_per_image'])
    policy, deltas, model_pairs = policy_tables(grid, decisions, baseline)
    policy_summary = summarize(policy, ['model','policy','target_fp_per_image'])
    for name, frame in [('outer_grid_by_fold',grid), ('grid_summary',grid_summary), ('gt_status',gt_status),
                        ('image_results',per_image), ('policy_by_fold',policy), ('policy_summary',policy_summary),
                        ('selected_vs_baseline',deltas), ('rtdetr_vs_yolo',model_pairs)]:
        frame.to_csv(out/f'{name}.csv', index=False)
    reasons = gt_status.groupby(['model','imgsz','target_fp_per_image','size','reason']).size().rename('objects').reset_index()
    reasons.to_csv(out/'miss_reason_counts.csv', index=False)
    quality_frame = pd.DataFrame(quality_records)
    quality_summary = quality_frame.groupby(['fold','model','imgsz','split']).agg(
        images=('image_id','size'), predictions=('prediction_count','sum'), zero_area_candidates=('zero_area_count','sum')
    ).reset_index()
    quality_summary.to_csv(out/'prediction_quality_summary.csv',index=False)
    save_json(out/'prediction_cache_index.json',cache_index)
    plot_results(out, grid_summary, policy_summary)
    cases = case_candidates(gt_status, decisions, baseline, case_target)
    cases.to_csv(out/'case_candidates.csv', index=False)
    produced = render_cases(out,cases,prepared['by_fold'],cache_index,limit) if limit else []
    lines = ['오탐 예산 + 입력 해상도 실험 완료', '',
             '아래 값은 pooled outer-test 지표입니다. small은 원본 bbox 면적<32²이며 APsmall이 아닙니다.',
             '해상도와 threshold는 inner validation에서 고정했습니다. test의 실제 FP/image는 목표와 다를 수 있습니다.', '']
    for _, r in policy_summary.iterrows():
        small = f"{100*r.pooled_small_recall:.2f}%" if r.pooled_small_gt else 'N/A (small GT=0)'
        lines.append(f"{r.model}, {r.policy}, validation FP target={r.target_fp_per_image:g}: "
                     f"Recall={100*r.pooled_recall:.2f}%, small Recall={small}, "
                     f"actual FP/image={r.pooled_fp_per_image:.4f}, small FN={r.pooled_small_fn}/{r.pooled_small_gt}")
    lines += ['', '전체 해상도 표에서 test 성능이 가장 높은 해상도를 사후 선택하지 마세요.',
              '면적 0인 예측은 삭제하지 않고 threshold 이상일 때 FP로 집계했습니다. zero_area_fp/pooled_zero_area_fp 열에서 확인할 수 있습니다.',
              'baseline 대비 선택 정책의 변화에는 입력 해상도와 validation threshold 보정이 함께 반영됩니다.',
              'recovered/regressed/still_missed 그림은 정성 사례이며 전체 효과는 CSV의 GT 수와 Recall로 판단합니다.',
              '후보 기반 미검출 분류는 구조적 원인이나 그림자/조명 등의 인과관계를 증명하지 않습니다.']
    (out/'interpretation.txt').write_text('\n'.join(lines), encoding='utf-8')
    save_json(out/'completed.json', {'status':'complete','lock_sha256':lock['lock_sha256'],'case_images':len(produced)})
    print('\n'.join(lines))
    print('저장 폴더:',out)
    return {'out':out,'validation':pd.DataFrame(val_records),'decisions':pd.DataFrame(decisions),
            'grid':grid,'grid_summary':grid_summary,'policy':policy,'policy_summary':policy_summary,
            'deltas':deltas,'model_pairs':model_pairs,'cases':cases}
