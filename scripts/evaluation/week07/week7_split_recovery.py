"""Restore the user's saved Week-5 splits without recreating image staging.

Schema verified against week7_split_metadata.zip:
source_manifest.json = ordered list; fold_indices.json['folds'] = train/val/test
index lists; each selected run has dataset_manifest.json with pixel/label hashes.
Original labels and pixels are verified; staging labels are regenerated only if
their bytes exactly match the saved staging-label SHA256. Images are not copied.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import numpy as np
import pandas as pd
import week7_fp_matched as core

VERSION = '1.0.1'
ROLE = {'train': 'train', 'val': 'inner_val', 'test': 'outer_test'}


def read_json(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'복구에 필요한 분할 기록이 없습니다: {p}')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def legacy_pixel_sha256(rgb):
    # Verified on an original image against its uploaded source/staging hashes.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    h = hashlib.sha256(np.asarray(bgr.shape, dtype='<i8').tobytes())
    h.update(bgr.tobytes())
    return h.hexdigest()


def manifest_path(fold, cfg):
    if fold.get('data_yaml'):
        p = Path(fold['data_yaml']).parent / 'dataset_manifest.json'
        if p.is_file():
            return p
    weight = Path(fold['yolo_weights']).resolve()
    root = core.remap_path(cfg['yolo_root'], cfg)
    candidates = []
    for parent in weight.parents:
        if parent == root.parent:
            break
        p = parent / 'dataset_manifest.json'
        if p.is_file():
            candidates.append(p)
    if len(candidates) != 1:
        raise ValueError(f"fold{fold['fold']}: 체크포인트에 연결된 dataset_manifest.json을 확정할 수 없습니다: {candidates}")
    return candidates[0]


def load_metadata(cfg, folds):
    root = core.remap_path(cfg['yolo_root'], cfg)
    paths = [root/'source_manifest.json', root/'fold_indices.json', root/'cv_config.json']
    source, index_doc, cv_config = [read_json(p) for p in paths]
    if not isinstance(source, list) or not source or not isinstance(index_doc, dict):
        raise ValueError('첨부 ZIP에서 확인한 source_manifest/fold_indices 형식과 다릅니다.')
    required = {'path', 'relative_path', 'label_sha256', 'pixel_sha256', 'width', 'height', 'group', 'provided_group'}
    for i, row in enumerate(source):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f'source_manifest[{i}] 필드가 누락되었습니다.')
        rel = PurePosixPath(row['relative_path'])
        if rel.is_absolute() or '..' in rel.parts or 'images' not in rel.parts:
            raise ValueError(f'유효하지 않은 원본 상대 경로: {rel}')
        for key in ('label_sha256', 'pixel_sha256'):
            value = row[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'source_manifest[{i}]: 유효하지 않은 {key}')
    n = len(source)
    if len({r['pixel_sha256'] for r in source}) != n or len({r['relative_path'] for r in source}) != n:
        raise ValueError('원본 manifest의 pixel hash 또는 상대 경로가 중복됩니다.')
    if cfg.get('expected_images') is not None and cfg['expected_images'] != n:
        raise ValueError(f"원본 이미지 수 불일치: {n} != {cfg['expected_images']}")
    split_defs = index_doc.get('folds')
    if not isinstance(split_defs, list) or len(split_defs) != len(folds):
        raise ValueError('fold_indices의 fold 개수가 선택한 가중치 개수와 다릅니다.')
    ids = [f['fold'] for f in split_defs]
    if len(set(ids)) != len(ids) or set(ids) != {f['fold'] for f in folds}:
        raise ValueError('fold_indices의 fold 번호가 선택한 가중치와 다릅니다.')
    by_id = {f['fold']: f for f in folds}
    expected_set = set(range(n))
    test_indices, assignments, stage_reference, summary = [], [], None, []
    for split_def in sorted(split_defs, key=lambda f:f['fold']):
        k = split_def['fold']
        used, groups = set(), {}
        for role in ROLE:
            indices = split_def.get(role)
            if not isinstance(indices, list) or not indices or any(type(i) is not int or i not in expected_set for i in indices):
                raise ValueError(f'fold{k}/{role}: 유효하지 않은 source index 목록')
            if len(indices) != len(set(indices)) or used.intersection(indices):
                raise ValueError(f'fold{k}: split 내부 또는 split 사이에 source index 중복')
            used.update(indices)
            groups[role] = {source[i]['group'] for i in indices}
            assignments.extend({'fold':k,'split':ROLE[role],'source_index':i} for i in indices)
            if role == 'test':
                test_indices.extend(indices)
        if used != expected_set:
            raise ValueError(f'fold{k}: 전체 source pool을 덮지 않습니다.')
        for a,b in [('train','val'),('train','test'),('val','test')]:
            if groups[a] & groups[b]:
                raise ValueError(f'fold{k}: 저장된 group 기준 {a}/{b} 누수')
        p = manifest_path(by_id[k], cfg)
        paths.append(p)
        stage = read_json(p)
        if not isinstance(stage, list) or len(stage) != n:
            raise ValueError(f'fold{k}: dataset_manifest의 행 수 불일치')
        required_stage = {'split','path','label_sha256','pixel_sha256','objects'}
        if any(not isinstance(r,dict) or not required_stage.issubset(r) for r in stage):
            raise ValueError(f'fold{k}: dataset_manifest 형식이 확인된 형식과 다릅니다.')
        stage_by_pixel = {r['pixel_sha256']:r for r in stage}
        if len(stage_by_pixel) != n or set(stage_by_pixel) != {r['pixel_sha256'] for r in source}:
            raise ValueError(f'fold{k}: source와 staged 이미지 집합 불일치')
        ordered = [stage_by_pixel[r['pixel_sha256']] for r in source]
        for role in ROLE:
            for i in split_def[role]:
                row = ordered[i]
                if row['split'] != role:
                    raise ValueError(f'fold{k} source#{i}: fold_indices와 dataset_manifest의 split 불일치')
                if type(row['objects']) is not int or row['objects'] < 0:
                    raise ValueError(f'fold{k} source#{i}: 유효하지 않은 객체 수')
            summary.append({'fold':k,'split':ROLE[role],'images':len(split_def[role]),
                            'objects':sum(ordered[i]['objects'] for i in split_def[role])})
        signature = [(r['pixel_sha256'],r['label_sha256'],r['objects']) for r in ordered]
        if stage_reference is None:
            stage_reference = ordered
        elif signature != [(r['pixel_sha256'],r['label_sha256'],r['objects']) for r in stage_reference]:
            raise ValueError('동일 source 이미지의 staged 라벨/객체 수가 fold마다 다릅니다.')
    if len(test_indices) != n or set(test_indices) != expected_set:
        raise ValueError('outer_test가 전체 source 이미지를 정확히 한 번 덮지 않습니다.')
    object_count = sum(r['objects'] for r in stage_reference)
    if cfg.get('expected_objects') is not None and cfg['expected_objects'] != object_count:
        raise ValueError(f"객체 수 불일치: {object_count} != {cfg['expected_objects']}")
    return {'source':source,'assignments':assignments,'stage':stage_reference,'cv_config':cv_config,
            'summary':summary,'sources':[{'path':str(p),'sha256':core.sha256_file(p)} for p in paths]}


def stage_label_bytes(original_label, gt, width, height, expected_sha, corrected):
    # Two known 12-significant-digit serializations; only a byte-for-byte hash match is accepted.
    def encode(rows):
        return ''.join('0 '+' '.join(f'{v:.12g}' for v in values)+'\n' for values in rows).encode('utf-8')
    normalized = []
    for x1,y1,x2,y2 in gt:
        normalized.append([(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height])
    candidates = [('xyxy_roundtrip_12sig',encode(normalized))]
    if corrected == 0:
        raw = [[float(v) for v in line.split()[1:]] for line in original_label.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        candidates.append(('source_normalized_12sig',encode(raw)))
    for method, content in candidates:
        if hashlib.sha256(content).hexdigest() == expected_sha:
            return content, method
    raise ValueError(f'원본 라벨은 확인됐지만 이전 staged 라벨 해시를 재현하지 못했습니다: {original_label}. '
                     '다른 형식으로 임의 저장하지 않았습니다.')


def verify_source(source, stage, data_root, eps):
    image = (data_root / source['relative_path']).resolve()
    if not image.is_relative_to(data_root.resolve()):
        raise ValueError('원본 상대 경로가 data_root 밖을 가리킵니다.')
    label = core.label_for(image)
    if not image.is_file() or not label.is_file():
        raise FileNotFoundError(f'원본 이미지/라벨이 없습니다. CFG[data_root]를 확인하세요.\nimage={image}\nlabel={label}')
    if core.sha256_file(label) != source['label_sha256']:
        raise ValueError(f'원본 라벨이 저장된 실행 당시와 다릅니다: {label}')
    rgb = core.decode_rgb(image)
    height,width = rgb.shape[:2]
    if (width,height) != (source['width'],source['height']):
        raise ValueError(f'원본 이미지 크기 변경: {image}')
    pixel_sha = legacy_pixel_sha256(rgb)
    if pixel_sha != source['pixel_sha256']:
        raise ValueError(f'원본 pixel hash 불일치: {image}. 이미지 변경 또는 디코딩 차이를 확인해야 합니다.')
    gt,corrected = core.load_gt(label,width,height,tolerance=eps)
    if len(gt) != stage['objects']:
        raise ValueError(f'기존 실행과 객체 수가 다릅니다: {label}')
    content,method = stage_label_bytes(label,gt,width,height,stage['label_sha256'],corrected)
    return {'image':str(image),'original_label':str(label),'relative_path':source['relative_path'],
            'pixel_sha256':pixel_sha,'source_label_sha256':source['label_sha256'],
            'stage_label_sha256':stage['label_sha256'],'objects':len(gt),'width':width,'height':height,
            'label_reconstruction':method,'boundary_corrections':corrected},content


def write_identical_or_new(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'기존 복구 파일의 내용이 다릅니다. 덮어쓰지 않았습니다: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_name(path.name+'.tmp')
        temp.write_bytes(content)
        temp.replace(path)


def recover(cfg, folds):
    meta = load_metadata(cfg,folds)
    data_root = core.remap_path(cfg.get('data_root') or meta['cv_config']['data_root'],cfg)
    eps = float(meta['cv_config'].get('coordinate_eps',1e-6))
    if not 0 <= eps <= 1e-6:
        raise ValueError('확인된 경계 보정 허용 범위를 넘는 coordinate_eps입니다.')
    source = meta['source']
    has_capture_groups = all(r['provided_group'] is not None and str(r['provided_group']) != '' for r in source)
    key = core.digest({'version':VERSION,'sources':meta['sources'],'data_root':str(data_root),
                       'code_sha256':core.sha256_file(__file__)})[:16]
    output = core.remap_path(cfg['output_root'],cfg) / f'recovered_splits_{key}'
    verified, label_bytes = [], []
    print(f'기존 분할 기록 대조 완료: {len(source)} images, {sum(r["objects"] for r in meta["stage"])} objects')
    print('원본 이미지·라벨 해시와 이전 staged 라벨을 검증합니다. 이미지와 가중치는 복사하지 않습니다.')
    for i,(row,stage) in enumerate(zip(source,meta['stage'])):
        info,content = verify_source(row,stage,data_root,eps)
        info.update(source_index=i,original_group=row['group'])
        verified.append(info)
        label_bytes.append(content)
        if (i+1)%50 == 0 or i+1 == len(source):
            print(f'원본 검증: {i+1}/{len(source)}')
    output.mkdir(parents=True,exist_ok=True)
    for i,content in enumerate(label_bytes):
        write_identical_or_new(output/'labels'/f'{i:06d}.txt',content)
    records = []
    for assignment in meta['assignments']:
        i = assignment['source_index']
        records.append({**assignment,'image':verified[i]['image'],
                        'label':str(output/'labels'/f'{i:06d}.txt'),
                        'group':source[i]['group'] if has_capture_groups else '',
                        'original_group':source[i]['group']})
    csv_path = output/'splits.csv'
    content = pd.DataFrame(records).to_csv(index=False,lineterminator='\n').encode('utf-8')
    write_identical_or_new(csv_path,content)
    pd.DataFrame(verified).to_csv(output/'verification.csv',index=False)
    pd.DataFrame(meta['summary']).to_csv(output/'split_summary.csv',index=False)
    audit = {'recovery_version':VERSION,'data_root':str(data_root),'csv':str(csv_path),
             'csv_sha256':core.sha256_file(csv_path),'images':len(source),
             'objects':sum(r['objects'] for r in verified),'csv_rows':len(records),
             'original_image_pixels_verified':True,'original_label_bytes_verified':True,
             'staged_label_bytes_reproduced':True,'all_fold_memberships_verified':True,
             'pixel_hash_method':'SHA256(shape little-endian int64 + decoded BGR bytes)',
             'boundary_corrections':sum(r['boundary_corrections'] for r in verified),
             'capture_groups_provided':has_capture_groups,
             'automatic_groups_checked_in_saved_indices':True,
             'note':'Stored group IDs alone do not verify adjacent video frame independence.',
             'sources':meta['sources'],'code_sha256':core.sha256_file(__file__)}
    core.dump_json(output/'recovery_audit.json',audit)
    write_identical_or_new(output/'week7_split_recovery.py',Path(__file__).read_bytes())
    for item in meta['sources']:
        if core.sha256_file(item['path']) != item['sha256']:
            raise ValueError('복구 도중 입력 metadata가 변경되었습니다.')
    provenance = [*meta['sources'],{'path':str(output/'recovery_audit.json'),
                                   'sha256':core.sha256_file(output/'recovery_audit.json')}]
    print(f'복구 완료: {len(source)} images / {audit["objects"]} objects / CSV {len(records)}행')
    if not has_capture_groups:
        print('촬영 그룹은 제공되지 않았습니다. 자동 group ID를 촬영/영상 그룹으로 간주하지 않습니다.')
    print('복구 CSV:',csv_path)
    return {'csv':csv_path,'out':output,'audit':audit,'sources':provenance,'summary':pd.DataFrame(meta['summary'])}
