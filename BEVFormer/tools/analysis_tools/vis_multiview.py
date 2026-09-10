"""Render the 6 surround-view camera images of a nuScenes sample as a single
packed 3x2 grid (top row: front-left/front/front-right, bottom row:
back-left/back/back-right), matching the layout of the `*_camera_gt.png`
figures produced by `visual_v2.py` but without any box overlay.

Samples are picked per map location (e.g. one Boston + one Singapore) using the
`log` table, which is joined through `scene_token` stored in the info pkl, so
the full NuScenes devkit database never has to be loaded.
"""

import argparse
import json
import os
import os.path as osp
import pickle

from PIL import Image

# Camera order for the 2x3 surround grid (top row: front, bottom row: back)
SURROUND_CAMS = [
    'CAM_FRONT_LEFT',  'CAM_FRONT',  'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',   'CAM_BACK',   'CAM_BACK_RIGHT',
]


def _load_scene_meta(version_dir):
    """scene_token -> {'location', 'name', 'description'}."""
    with open(osp.join(version_dir, 'scene.json')) as f:
        scenes = json.load(f)
    with open(osp.join(version_dir, 'log.json')) as f:
        logs = {l['token']: l for l in json.load(f)}
    return {
        s['token']: {
            'location': logs[s['log_token']]['location'],
            'name': s['name'],
            'description': s['description'],
        }
        for s in scenes if s['log_token'] in logs
    }


def _resolve(data_path, dataroot):
    """Info-pkl paths are stored relative to a './data/nuscenes' cwd."""
    if osp.isabs(data_path) and osp.exists(data_path):
        return data_path
    return osp.join(dataroot, data_path.split('nuscenes/', 1)[-1].lstrip('./'))


def render_grid(info, dataroot, out_path):
    tiles = [Image.open(_resolve(info['cams'][c]['data_path'], dataroot)).convert('RGB')
             for c in SURROUND_CAMS]
    w, h = tiles[0].size
    canvas = Image.new('RGB', (3 * w, 2 * h))
    for idx, tile in enumerate(tiles):
        if tile.size != (w, h):
            tile = tile.resize((w, h), Image.BICUBIC)
        canvas.paste(tile, ((idx % 3) * w, (idx // 3) * h))
    canvas.save(out_path)
    return canvas.size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', default='BEVFormer/data/nuscenes')
    p.add_argument('--info-pkl', default=None,
                   help='default: <dataroot>/nuscenes_infos_temporal_val.pkl')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--out-dir', default='visualize/multiview')
    p.add_argument('--locations', nargs='+', default=['boston', 'singapore'],
                   help='case-insensitive substrings matched against log location')
    p.add_argument('--per-location', type=int, default=1)
    p.add_argument('--tokens', nargs='+', default=None,
                   help='explicit sample tokens; overrides --locations')
    p.add_argument('--scenes', nargs='+', default=None,
                   help="scene names (e.g. scene-0104); overrides --locations")
    p.add_argument('--frame-ratio', type=float, default=0.5,
                   help='which frame of the scene to render, 0=first 1=last')
    p.add_argument('--suffix', default='multiview')
    return p.parse_args()


def main():
    args = parse_args()
    info_pkl = args.info_pkl or osp.join(args.dataroot,
                                         'nuscenes_infos_temporal_val.pkl')
    with open(info_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data

    meta = _load_scene_meta(osp.join(args.dataroot, args.version))
    scene_loc = {k: v['location'] for k, v in meta.items()}
    name2token = {v['name']: k for k, v in meta.items()}
    os.makedirs(args.out_dir, exist_ok=True)

    if args.tokens:
        by_token = {i['token']: i for i in infos}
        picked = [(scene_loc.get(by_token[t].get('scene_token'), 'unknown'),
                   by_token[t]) for t in args.tokens]
    elif args.scenes:
        # Scene frames are stored consecutively in the info pkl; `frame_idx`
        # gives the position within the scene.
        picked = []
        for name in args.scenes:
            tok = name2token.get(name)
            frames = sorted((i for i in infos if i.get('scene_token') == tok),
                            key=lambda i: i.get('frame_idx', 0))
            if not frames:
                print(f'[warn] scene "{name}" not in this info pkl')
                continue
            sel = frames[min(len(frames) - 1,
                             int(round(args.frame_ratio * (len(frames) - 1))))]
            picked.append((f"{meta[tok]['location']}_{name}", sel))
    else:
        picked = []
        for key in args.locations:
            n = 0
            for info in infos:
                loc = scene_loc.get(info.get('scene_token'), '')
                if key.lower() not in loc.lower():
                    continue
                picked.append((loc, info))
                n += 1
                if n >= args.per_location:
                    break
            if n == 0:
                print(f'[warn] no sample found for location "{key}"')

    for loc, info in picked:
        out_path = osp.join(args.out_dir, f'{loc}_{info["token"]}_{args.suffix}.png')
        size = render_grid(info, args.dataroot, out_path)
        print(f'{loc:24s} {info["token"]}  ->  {out_path}  {size[0]}x{size[1]}')


if __name__ == '__main__':
    main()
