import numpy as np
import pytest
import torch

from BEVFormer.projects.bevdiffuser.analyze.analyze_occ_condition_ablation import (
    ARMS, paired_teacher_forward, selection_rows, summarize, visible_cross_entropy,
)
from projects.bevdiffuser.launch_occ_condition_multigpu import (
    EXPECTED_ARMS, EXPECTED_ARMS_6, aggregate_frames, merge_shards,
)
import json


class FakeScheduler:
    def add_noise(self, sample, noise, timesteps):
        assert timesteps.item() == 100
        return sample + noise


class FakeTeacher:
    def __init__(self):
        self.cfg_scale = 2.0
        self.noise_scheduler = FakeScheduler()

    def __call__(self, clean, metas, semantic_prob):
        noise = torch.randn_like(clean)
        x_t = self.noise_scheduler.add_noise(clean, noise, torch.tensor(100))
        # Analytic CFG: unconditional prediction 0, conditional prediction 1.
        return x_t + self.cfg_scale * semantic_prob.mean()


def test_paired_cfg_uses_identical_noise_and_restores_teacher():
    teacher = FakeTeacher()
    clean = torch.zeros(1, 2, 2, 2)
    semantic = torch.ones(1, 17, 1, 1)
    outputs, hashes = paired_teacher_forward(teacher, clean, {}, semantic, 'fixed-token')
    assert set(outputs) == set(ARMS)
    assert torch.allclose(outputs['conditional'] - outputs['unconditional'], torch.ones_like(clean))
    assert torch.allclose(outputs['guided_cfg2'] - outputs['unconditional'], 2 * torch.ones_like(clean))
    assert len(hashes['noise']) == len(hashes['x_t']) == 64
    assert teacher.cfg_scale == 2.0
    assert teacher.noise_scheduler.add_noise.__func__ is FakeScheduler.add_noise
    repeated, repeated_hashes = paired_teacher_forward(teacher, clean, {}, semantic, 'fixed-token')
    assert repeated_hashes == hashes
    assert torch.equal(repeated['unconditional'], outputs['unconditional'])


def test_single_cfg4_arm_reuses_seed_and_restores_scale():
    teacher = FakeTeacher()
    clean = torch.zeros(1, 2, 2, 2)
    semantic = torch.ones(1, 17, 1, 1)
    single, single_hash = paired_teacher_forward(
        teacher, clean, {}, semantic, 'fixed-token', {'conditional_cfg4': 4.0})
    original, original_hash = paired_teacher_forward(teacher, clean, {}, semantic, 'fixed-token')
    assert set(single) == {'conditional_cfg4'}
    assert single_hash == original_hash
    assert torch.allclose(single['conditional_cfg4'] - original['unconditional'],
                          4 * torch.ones_like(clean))
    assert teacher.cfg_scale == 2.0


def test_visible_ce_separates_free_and_nonfree():
    logits = torch.zeros(1, 1, 2, 2, 18)
    logits[0, 0, 0, 0, 4] = 5
    logits[0, 0, 0, 1, 17] = 5
    logits[0, 0, 1, 0, 8] = 5
    labels = np.array([[[4, 17], [8, 4]]])
    visible = np.array([[[True, True], [False, True]]])
    values = visible_cross_entropy(logits, labels, visible)
    per_voxel = torch.nn.functional.cross_entropy(logits.reshape(-1, 18),
                                                   torch.as_tensor(labels).reshape(-1), reduction='none')
    assert values['all'][1] == 3 and values['nonfree'][1] == 2
    assert values['all'][0] == pytest.approx(float(per_voxel[[0, 1, 3]].sum()))
    assert values['nonfree'][0] == pytest.approx(float(per_voxel[[0, 3]].sum()))
    empty = visible_cross_entropy(logits, labels, np.zeros_like(visible))
    assert empty['all'] == (0.0, 0) and empty['nonfree'] == (0.0, 0)
    with pytest.raises(ValueError):
        visible_cross_entropy(logits[..., :17], labels, visible)


def test_undefined_class_iou_is_null():
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[4, 4] = 2
    totals = {'clean': hist}
    ce = {'clean': {'all': [4.0, 2], 'nonfree': [2.0, 1]}}
    summary, classes = summarize(totals, ce)
    assert summary['clean']['miou_nonfree_defined_classes'] == 1.0
    assert summary['clean']['defined_classes'] == 1
    assert summary['clean']['visible_ce_all'] == 2.0
    assert classes[1]['iou'] is None


def test_50_frame_eight_shards_cover_selection_once(tmp_path):
    selection = tmp_path / 'selected_frames.csv'
    with selection.open('w') as stream:
        stream.write('token,scene_token,timestamp,gt\n')
        for index in range(50):
            stream.write(f'token{index},scene{index},{index},gt{index}\n')
    shards = [selection_rows(selection, 50, index, 8) for index in range(8)]
    assert [len(shard) for shard in shards] == [7, 7, 6, 6, 6, 6, 6, 6]
    assert sorted(row['_selection_index'] for shard in shards for row in shard) == list(range(50))
    assert [row['token'] for row in shards[0]] == [f'token{i}' for i in range(0, 50, 8)]


def test_merge_uses_raw_confusion_and_ce_counts():
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[4, 4] = 2
    hist[17, 17] = 3
    item = {'confusion': hist.tolist(), 'ce_all_sum': 10.0, 'ce_all_count': 5,
            'ce_nonfree_sum': 6.0, 'ce_nonfree_count': 2}
    base = {'scene_token': 'scene_a', 'noise_sha256': 'a', 'noise_t300_sha256': 'a',
            'x_t_sha256': 'b', 'x_t_t300_sha256': 'c',
            'arms': {arm: item for arm in EXPECTED_ARMS}}
    first = dict(base, token='one')
    second = dict(base, token='two', scene_token='scene_b')
    summary, pooled = aggregate_frames([first, second])
    assert summary['frames'] == 2 and summary['scenes'] == 2
    assert summary['arms']['clean']['visible_ce_all'] == 2.0
    assert summary['arms']['clean']['visible_ce_nonfree'] == 3.0
    assert summary['arms']['clean']['miou_nonfree_defined_classes'] == 1.0
    assert next(row for row in pooled if row['system'] == 'clean' and row['class_id'] == 4)['tp'] == 4
    assert next(row for row in pooled if row['system'] == 'clean' and row['class_id'] == 1)['iou'] is None


def test_200_frame_eight_shards_and_repeated_scenes_merge(tmp_path):
    selection = tmp_path / 'selected_frames.csv'
    with selection.open('w') as stream:
        stream.write('token,scene_token,timestamp,gt\n')
        for index in range(200):
            scene = index if index < 150 else index - 150
            stream.write(f'token{index},scene{scene},{index},gt{index}\n')
    rows = selection_rows(selection, 200)
    shards = [selection_rows(selection, 200, index, 8) for index in range(8)]
    assert [len(shard) for shard in shards] == [25] * 8
    assert len({row['token'] for row in rows}) == 200
    assert len({row['scene_token'] for row in rows}) == 150
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[4, 4] = 1
    item = {'confusion': hist.tolist(), 'ce_all_sum': 2.0, 'ce_all_count': 1,
            'ce_nonfree_sum': 2.0, 'ce_nonfree_count': 1}
    common = {'selection_csv': str(selection), 'selection_sha256': 'test',
              'selection_limit': 200, 'num_shards': 8, 'cache_frame': 'ego',
              'seed_rule': 'test', 'queue_length': 4, 'torch_version': 'test',
              'arms': {arm: 1 for arm in EXPECTED_ARMS if arm != 'clean'},
              'sampler_config': {}, 'effective_sampler': {}, 'include_t300_cfg4': True,
              'actual_ddim_grid': [81, 61, 41, 21, 1],
              'actual_ddim_grid_t300': [241, 181, 121, 61, 1],
              'files_sha256': {str(selection): 'test'}}
    for index, shard_rows in enumerate(shards):
        folder = tmp_path / f'shard_{index:02d}'
        folder.mkdir()
        provenance = dict(common, shard_index=index,
                          selected_tokens=[row['token'] for row in shard_rows])
        (folder / 'provenance.json').write_text(json.dumps(provenance))
        with (folder / 'per_frame.jsonl').open('w') as stream:
            for row in shard_rows:
                frame = {'token': row['token'], 'scene_token': row['scene_token'],
                         'selection_index': row['_selection_index'],
                         'noise_sha256': 'same', 'noise_t300_sha256': 'same',
                         'x_t_sha256': 't100', 'x_t_t300_sha256': 't300',
                         'arms': {arm: item for arm in EXPECTED_ARMS}}
                stream.write(json.dumps(frame) + '\n')
    summary = merge_shards(tmp_path, rows, [str(index) for index in range(8)], 1.0, 50, 150)
    assert summary['frames'] == 200 and summary['scenes'] == 150
    expected = {'summary_old50.json': (50, 50),
                'summary_new150.json': (150, 150),
                'summary_first150.json': (150, 150),
                'summary_second50.json': (50, 50),
                'summary_new_scenes100.json': (100, 100)}
    for filename, counts in expected.items():
        subgroup = json.loads((tmp_path / filename).read_text())
        assert (subgroup['frames'], subgroup['scenes']) == counts
        assert subgroup['arms']['clean']['visible_ce_all'] == 2.0
    merged = json.loads((tmp_path / 'provenance.json').read_text())
    assert merged['reference_count'] == 50 and merged['first_pass_count'] == 150


def test_500_frame_eight_shards_cover_repeated_scenes(tmp_path):
    selection = tmp_path / 'selected_frames.csv'
    with selection.open('w') as stream:
        stream.write('token,scene_token,timestamp,gt\n')
        for index in range(500):
            scene = index % 150 if index < 450 else index - 450
            stream.write(f'token{index},scene{scene},{index},gt{index}\n')
    shards = [selection_rows(selection, 500, index, 8) for index in range(8)]
    assert [len(shard) for shard in shards] == [63, 63, 63, 63, 62, 62, 62, 62]
    assert sorted(row['_selection_index'] for shard in shards for row in shard) == list(range(500))
    assert len({row['scene_token'] for shard in shards for row in shard}) == 150


def test_six_arm_merge_requires_explicit_six_arm_variant():
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[4, 4] = 1
    item = {'confusion': hist.tolist(), 'ce_all_sum': 1.0, 'ce_all_count': 1,
            'ce_nonfree_sum': 1.0, 'ce_nonfree_count': 1}
    base = {'token': 'one', 'scene_token': 'scene', 'noise_sha256': 'a',
            'noise_t300_sha256': 'a', 'x_t_sha256': 'b', 'x_t_t300_sha256': 'c'}
    five = dict(base, arms={arm: item for arm in EXPECTED_ARMS})
    six = dict(base, arms={arm: item for arm in EXPECTED_ARMS_6})
    assert len(aggregate_frames([five])[0]['arms']) == 5
    with pytest.raises(ValueError, match='Missing or extra arms'):
        aggregate_frames([six])
    with pytest.raises(ValueError, match='Missing or extra arms'):
        aggregate_frames([five], EXPECTED_ARMS_6)
    assert len(aggregate_frames([six], EXPECTED_ARMS_6)[0]['arms']) == 6
