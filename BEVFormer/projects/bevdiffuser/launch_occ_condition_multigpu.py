"""Launch and merge a bounded, disjoint multi-GPU occupancy condition probe."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BEVFormer.projects.bevdiffuser.analyze.analyze_occ_condition_ablation import (
    BEV, class_rows, selection_rows, sha256, summarize,
)


EXPECTED_ARMS = ('clean', 'unconditional', 'conditional', 'guided_cfg2', 'conditional_t300_cfg4')
EXPECTED_ARMS_6 = (*EXPECTED_ARMS[:-1], 'conditional_t100_cfg4', EXPECTED_ARMS[-1])
SCRIPT = Path(__file__).with_name('analyze_occ_condition_ablation.py')


def aggregate_frames(frames, expected_arms=EXPECTED_ARMS):
    totals = {arm: np.zeros((18, 18), dtype=np.int64) for arm in expected_arms}
    ce_totals = {arm: {'all': [0.0, 0], 'nonfree': [0.0, 0]} for arm in expected_arms}
    for frame in frames:
        if set(frame['arms']) != set(expected_arms):
            raise ValueError(f"Missing or extra arms for {frame['token']}")
        if frame['noise_sha256'] != frame['noise_t300_sha256'] or frame['x_t_sha256'] == frame['x_t_t300_sha256']:
            raise ValueError(f"Noise pairing failed for {frame['token']}")
        for arm in expected_arms:
            item = frame['arms'][arm]
            hist = np.asarray(item['confusion'], dtype=np.int64)
            if hist.shape != (18, 18) or (hist < 0).any():
                raise ValueError(f"Invalid confusion for {frame['token']} {arm}")
            if int(hist.sum()) != item['ce_all_count']:
                raise ValueError(f"CE/confusion visible count differs for {frame['token']} {arm}")
            totals[arm] += hist
            for region in ('all', 'nonfree'):
                count = item[f'ce_{region}_count']
                value = item[f'ce_{region}_sum']
                if count < 0 or not np.isfinite(value):
                    raise ValueError(f"Invalid CE for {frame['token']} {arm}")
                ce_totals[arm][region][0] += value
                ce_totals[arm][region][1] += count
    arm_summary, pooled = summarize(totals, ce_totals)
    return {'frames': len(frames), 'scenes': len({frame['scene_token'] for frame in frames}),
            'arms': arm_summary}, pooled


def merge_shards(output, rows, gpu_ids, wall_seconds, reference_count=24, first_pass_count=None,
                 include_t100_cfg4=False):
    if not 0 < reference_count < len(rows):
        raise ValueError('Reference count must fall inside selected rows')
    if first_pass_count is not None and not reference_count < first_pass_count < len(rows):
        raise ValueError('First-pass count must follow reference and precede selection end')
    selected = [row['token'] for row in rows]
    frames_by_token = {}
    provenances = []
    for index in range(len(gpu_ids)):
        shard = output / f'shard_{index:02d}'
        provenance = json.loads((shard / 'provenance.json').read_text())
        provenances.append(provenance)
        frame_rows = [json.loads(line) for line in (shard / 'per_frame.jsonl').read_text().splitlines()]
        expected = selected[index::len(gpu_ids)]
        if [frame['token'] for frame in frame_rows] != expected:
            raise ValueError(f'Shard {index} does not match deterministic token slice')
        if provenance['selected_tokens'] != expected:
            raise ValueError(f'Shard {index} provenance token list differs')
        for frame in frame_rows:
            if frame['token'] in frames_by_token:
                raise ValueError(f"Duplicate frame {frame['token']}")
            frames_by_token[frame['token']] = frame
    if set(frames_by_token) != set(selected) or len(frames_by_token) != len(selected):
        raise ValueError('Shard union does not equal requested selection')
    reference = provenances[0]
    expected_arms = EXPECTED_ARMS_6 if include_t100_cfg4 else EXPECTED_ARMS
    if set(reference['arms']) != set(expected_arms) - {'clean'}:
        raise ValueError('Shard arms differ from requested run variant')
    shared_fields = ('selection_csv', 'selection_sha256', 'selection_limit', 'num_shards',
                     'cache_frame', 'seed_rule', 'queue_length', 'torch_version', 'arms',
                     'sampler_config', 'effective_sampler', 'include_t300_cfg4',
                     'actual_ddim_grid', 'actual_ddim_grid_t300')
    for index, provenance in enumerate(provenances):
        if provenance['shard_index'] != index:
            raise ValueError(f'Shard provenance index mismatch: {index}')
        if bool(provenance.get('include_t100_cfg4', False)) != include_t100_cfg4:
            raise ValueError(f'Shard {index} t100 CFG4 flag differs')
        if any(provenance[key] != reference[key] for key in shared_fields):
            raise ValueError(f'Shard {index} shared provenance differs')
        for path, digest in reference['files_sha256'].items():
            if path in provenance['files_sha256'] and provenance['files_sha256'][path] != digest:
                raise ValueError(f'Shard {index} input hash differs: {path}')
    ordered = [frames_by_token[token] for token in selected]
    if [frame['selection_index'] for frame in ordered] != list(range(len(selected))):
        raise ValueError('Selection indices are incomplete or out of order')
    all_summary, pooled = aggregate_frames(ordered, expected_arms)
    old_summary, _ = aggregate_frames(ordered[:reference_count], expected_arms)
    new_summary, _ = aggregate_frames(ordered[reference_count:], expected_arms)
    all_summary.update({'actual_ddim_grid': reference['actual_ddim_grid'],
                        'actual_ddim_grid_t300': reference['actual_ddim_grid_t300'],
                        'wall_seconds': wall_seconds, 'gpu_ids': gpu_ids,
                        'interpretation': 'Fixed-head selected-frame compatibility, not Stage 2 transfer.'})
    with (output / 'per_frame.jsonl').open('x') as stream:
        for frame in ordered:
            stream.write(json.dumps(frame, allow_nan=False) + '\n')
    with (output / 'pooled_class.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=pooled[0].keys())
        writer.writeheader()
        writer.writerows(pooled)
    summaries = [('summary.json', all_summary),
                 (f'summary_old{reference_count}.json', old_summary),
                 (f'summary_new{len(rows) - reference_count}.json', new_summary)]
    if first_pass_count is not None:
        first_summary, _ = aggregate_frames(ordered[:first_pass_count], expected_arms)
        second_summary, _ = aggregate_frames(ordered[first_pass_count:], expected_arms)
        middle_summary, _ = aggregate_frames(ordered[reference_count:first_pass_count], expected_arms)
        summaries.extend([(f'summary_first{first_pass_count}.json', first_summary),
                          (f'summary_second{len(rows) - first_pass_count}.json', second_summary),
                          (f'summary_new_scenes{first_pass_count - reference_count}.json', middle_summary)])
    for filename, payload in summaries:
        with (output / filename).open('x') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    merged_provenance = {key: reference[key] for key in shared_fields}
    merged_provenance.update({'gpus': gpu_ids, 'selected_tokens': selected,
                              'reference_count': reference_count,
                              'first_pass_count': first_pass_count,
                              'include_t100_cfg4': include_t100_cfg4,
                              'group_files': [filename for filename, _ in summaries],
                              'selection_sha256_verified': sha256(output / 'selected_frames.csv'),
                              'shards': [str(output / f'shard_{i:02d}') for i in range(len(gpu_ids))],
                              'worker_provenance': [str(output / f'shard_{i:02d}' / 'provenance.json')
                                                    for i in range(len(gpu_ids))],
                              'wall_seconds': wall_seconds,
                              'subset_not_full_validation': True})
    with (output / 'provenance.json').open('x') as stream:
        json.dump(merged_provenance, stream, indent=2, allow_nan=False)
    return all_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-csv', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--reference-count', type=int, default=24)
    parser.add_argument('--first-pass-count', type=int, default=None)
    parser.add_argument('--include-t100-cfg4', action='store_true')
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    selection = args.selection_csv.resolve()
    gpu_ids = [item.strip() for item in args.gpus.split(',')]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(not item.isdigit() for item in gpu_ids):
        raise ValueError('Supply unique numeric GPU IDs')
    if not output.is_dir() or not selection.is_file() or selection.parent != output:
        raise ValueError('Output directory must already contain selection CSV')
    if args.limit < 1:
        raise ValueError('Limit must be positive')
    rows = selection_rows(selection, args.limit)
    if len(rows) < len(gpu_ids):
        raise ValueError('Every GPU shard must receive at least one selected frame')
    if not 0 < args.reference_count < len(rows):
        raise ValueError('Reference count must fall inside selected rows')
    if args.first_pass_count is not None and not args.reference_count < args.first_pass_count < len(rows):
        raise ValueError('First-pass count must follow reference and precede selection end')
    if any((output / name).exists() for name in ('summary.json', 'provenance.json', 'per_frame.jsonl')):
        raise FileExistsError('Merged output already exists')
    start = time.monotonic()
    if not args.merge_only:
        if any((output / f'shard_{i:02d}').exists() or (output / f'shard_{i:02d}.log').exists()
               for i in range(len(gpu_ids))):
            raise FileExistsError('Shard output already exists; use --merge-only after completed workers')
        processes = []
        log_files = []
        try:
            for index, gpu in enumerate(gpu_ids):
                logfile = (output / f'shard_{index:02d}.log').open('x')
                log_files.append(logfile)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = gpu
                env['OMP_NUM_THREADS'] = '2'
                command = [sys.executable, str(SCRIPT), '--selection-csv', str(selection),
                           '--limit', str(args.limit), '--num-shards', str(len(gpu_ids)),
                           '--shard-index', str(index), '--include-t300-cfg4',
                           '--output-dir', str(output / f'shard_{index:02d}')]
                if args.include_t100_cfg4:
                    command.append('--include-t100-cfg4')
                processes.append(subprocess.Popen(command, cwd=BEV, env=env,
                                                  stdout=logfile, stderr=subprocess.STDOUT))
                print(f'Started shard {index} on GPU {gpu}: pid {processes[-1].pid}', flush=True)
            statuses = [process.wait() for process in processes]
        finally:
            for logfile in log_files:
                logfile.close()
        if any(status != 0 for status in statuses):
            raise RuntimeError(f'Worker exit codes: {statuses}; inspect shard logs')
    summary = merge_shards(output, rows, gpu_ids, time.monotonic() - start,
                           args.reference_count, args.first_pass_count, args.include_t100_cfg4)
    print(json.dumps({'output': str(output), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
