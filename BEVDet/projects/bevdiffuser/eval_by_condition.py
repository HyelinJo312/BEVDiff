"""
Per-condition nuScenes detection evaluation (sunny / rainy / day / night).

Predictions are condition-independent, so this script does NOT re-run inference.
It reuses an existing `results_nusc.json` (produced by a normal test run) and
re-evaluates it on subsets of samples grouped by the nuScenes scene description.

Scene descriptions are free text; we use the standard keyword convention:
    rainy : 'rain'  in description
    sunny : 'rain'  NOT in description   (i.e. not raining / clear)
    night : 'night' in description
    day   : 'night' NOT in description
The 'sunny/rainy' and 'day/night' splits are two independent partitions of the
full val set, so a sample appears in exactly one of {sunny,rainy} and one of
{day,night}.

Usage:
    python eval_by_condition.py \
        --bev_config ../configs/diff_bevformer/layout_tiny_seg_v4_2.py \
        --result_path ../../test/<cfg>/<run>/<ckpt>/5_5_5/results_nusc.json
"""

import argparse
from collections import Counter
import csv
import json
import os
import sys

# Make the BEVFormer root importable. The mmdet3d plugin's datasets/__init__.py
# imports projects.bevdiffuser.data_utils, which registers the custom diffusion
# dataset types — so we must NOT import data_utils directly here (double-register).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

from mmcv import Config
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes

from projects.mmdet3d_plugin.datasets.nuscnes_eval import (
    NuScenesEval_custom,
    SCENE_CONDITIONS,
    evaluate_by_scene_condition,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-condition nuScenes detection evaluation."
    )
    parser.add_argument("--bev_config", required=True, help="test config file path")
    parser.add_argument(
        "--result_path",
        required=True,
        help="path to results_nusc.json produced by a previous test run",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="folder for eval scratch output (default: alongside result_path)",
    )
    return parser.parse_args()


def load_plugin(cfg):
    """Import plugin modules so build_dataset can resolve custom dataset types."""
    if cfg.get("plugin", False) and cfg.get("plugin_dir", None):
        import importlib

        module_path = ".".join(os.path.dirname(cfg.plugin_dir).split("/"))
        importlib.import_module(module_path)


def build_test_dataset(cfg, data_root_abs):
    source = cfg.data.test
    dataset_cfg = dict(
        type="CustomNuScenesDataset",
        data_root=data_root_abs,
        ann_file=os.path.join(data_root_abs, os.path.basename(source.ann_file)),
        pipeline=[],
        classes=source.get("classes", cfg.class_names),
        modality=source.get("modality", cfg.input_modality),
        test_mode=True,
    )
    for key in ("eval_version", "load_interval"):
        if key in source:
            dataset_cfg[key] = source[key]
    return build_dataset(dataset_cfg)


def save_condition_rows(rows, nusc_eval, nusc, config_path, result_path, eval_set, output_dir):
    tokens = sorted(nusc_eval.all_gt.sample_tokens)
    sample_scenes = {token: nusc.get("sample", token)["scene_token"] for token in tokens}
    descriptions = {scene: nusc.get("scene", scene)["description"].lower()
                    for scene in set(sample_scenes.values())}
    predicates = {"overall": lambda description: True, **SCENE_CONDITIONS}
    records = []
    for condition, num_samples, mean_ap, nds in rows:
        valid = [token for token in tokens
                 if predicates[condition](descriptions[sample_scenes[token]])]
        support = Counter(box.detection_name for token in valid
                          for box in nusc_eval.all_gt[token])
        records.append(dict(condition=condition, num_samples=num_samples,
                            mAP=mean_ap, NDS=nds,
                            num_scenes=len({sample_scenes[token] for token in valid}),
                            class_support={name: support[name]
                                           for name in nusc_eval.cfg.class_names}))

    payload = dict(
        config_path=config_path, result_path=result_path, eval_set=eval_set,
        condition_definition={"sunny": "no 'rain' in lowercased scene description",
                              "rainy": "'rain' in lowercased scene description",
                              "day": "no 'night' in lowercased scene description",
                              "night": "'night' in lowercased scene description"},
        evaluated_tokens=tokens, rows=records,
    )
    with open(os.path.join(output_dir, "conditions.json"), "x") as stream:
        json.dump(payload, stream, indent=2)
    with open(os.path.join(output_dir, "conditions.csv"), "x", newline="") as stream:
        fields = ["condition", "num_samples", "mAP", "NDS", "num_scenes"]
        fields += ["gt_count_" + name for name in nusc_eval.cfg.class_names]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record[key] for key in fields[:5]},
                             **{"gt_count_" + name: count
                                for name, count in record["class_support"].items()}})


def main():
    args = parse_args()

    # Resolve user paths before chdir, then run from the BEVFormer root.
    args.bev_config = os.path.abspath(args.bev_config)
    args.result_path = os.path.abspath(args.result_path)
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
    bevformer_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(bevformer_root)
    # The real nuScenes lives at <BEVFormer>/data/nuscenes regardless of the
    # data_root string in the config.
    data_root_abs = os.path.join(bevformer_root, "data", "nuscenes") + "/"

    cfg = Config.fromfile(args.bev_config)
    load_plugin(cfg)

    dataset = build_test_dataset(cfg, data_root_abs)

    with open(args.result_path) as stream:
        prediction_tokens = set(json.load(stream)["results"])
    dataset_tokens = {info["token"] for info in dataset.data_infos}
    assert prediction_tokens == dataset_tokens, "Prediction tokens differ from evaluation dataset"

    nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=True)
    eval_set_map = {"v1.0-mini": "mini_val", "v1.0-trainval": "val"}

    output_dir = args.output_dir or os.path.join(
        *os.path.split(args.result_path)[:-1], "by_condition"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions + GT once; all_gt / all_preds hold the full filtered set.
    nusc_eval = NuScenesEval_custom(
        nusc,
        config=dataset.eval_detection_configs,
        result_path=args.result_path,
        eval_set=eval_set_map[dataset.version],
        output_dir=output_dir,
        verbose=False,
        overlap_test=False,
        data_infos=dataset.data_infos,
    )

    rows = evaluate_by_scene_condition(nusc_eval, nusc, result_path=args.result_path)
    save_condition_rows(rows, nusc_eval, nusc, args.bev_config, args.result_path,
                        eval_set_map[dataset.version], output_dir)


if __name__ == "__main__":
    main()
