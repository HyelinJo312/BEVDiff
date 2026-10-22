from typing import Any, Dict, Tuple

import logging

import torch
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap

# shapely가 nuScenes 맵 폴리곤 처리 시 INFO 레벨로 쏟아내는
# "Self-intersection at or near point ..." 로그를 억제한다.
logging.getLogger('shapely.geos').setLevel(logging.WARNING)
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class BEVFusionSegmentation:
    """
    获取BEVFusion论文中使用的分割标注

    """
    def __init__(
        self,
        dataset_root,
        map_grid_conf,
        classes=('drivable_area', 'ped_crossing',
                 'walkway', 'stop_line', 'carpark_area',
                 'divider'),
    ):
        super().__init__()
        xbound = map_grid_conf['xbound']
        ybound = map_grid_conf['ybound']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def show_seg(self, labels):

        mask_colors = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(len(self.classes) + 1)
        ]
        img = np.zeros((200, 400, 3))

        for index, mask_ in enumerate(labels):
            color_mask = mask_colors[index]
            mask_ = mask_.astype(bool)
            img[mask_] = color_mask

        return img

    def __call__(self, results):

        location, ego2global_translation, ego2global_rotation = \
            results['location'], results['ego2global_translation'], results['ego2global_rotation']
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        # masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        semantic_masks = labels  # 这里已经进行了one_hot编码了
        num_cls = semantic_masks.shape[0]
        indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = np.sum(semantic_masks * indices, axis=0)
        semantic_indices = np.where(semantic_indices > 5, 6, semantic_indices)

        results.update({
            'semantic_map': torch.from_numpy(semantic_masks),
            'semantic_indices': torch.from_numpy(semantic_indices).long(),
        })
        return results


@PIPELINES.register_module()
class BEVFusionSegmentationV2:
    """BEVFusion-style segmentation label generator with priority-based flatten.

    Differences vs ``BEVFusionSegmentation``:
      * Uses an explicit per-class priority to flatten the 6-channel binary
        masks into a single-channel index map. The previous version did
        ``sum(masks * indices)`` which corrupts overlapping pixels (e.g.
        drivable_area + ped_crossing produced the index of walkway).
      * Output index layout: 0 = background, then classes ordered by
        ascending priority (highest priority overwrites last).

    Default priority (low -> high, last wins):
        drivable_area < carpark_area < walkway < stop_line
                      < ped_crossing < divider
    Index mapping with default priority:
        0: background
        1: drivable_area
        2: carpark_area
        3: walkway
        4: stop_line
        5: ped_crossing
        6: divider
    """

    DEFAULT_PRIORITY = (
        'drivable_area',
        'carpark_area',
        'walkway',
        'stop_line',
        'ped_crossing',
        'divider',
    )

    def __init__(
        self,
        dataset_root,
        map_grid_conf,
        classes=('drivable_area', 'ped_crossing', 'walkway',
                 'stop_line', 'carpark_area', 'divider'),
        priority=None,
    ):
        super().__init__()
        xbound = map_grid_conf['xbound']
        ybound = map_grid_conf['ybound']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = tuple(classes)

        order = tuple(priority) if priority is not None else self.DEFAULT_PRIORITY
        unknown = set(self.classes) - set(order)
        assert not unknown, f'priority is missing classes: {unknown}'
        # keep only entries that are in `classes`, in priority order (low -> high)
        self.priority_order = tuple(c for c in order if c in self.classes)
        # index assigned to each class in the flattened semantic map
        # (background = 0; classes get 1..N following priority order)
        self.class_to_index = {
            name: i + 1 for i, name in enumerate(self.priority_order)
        }

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def _layer_mappings(self):
        mappings = {}
        for name in self.classes:
            if name == 'drivable_area*':
                mappings[name] = ['road_segment', 'lane']
            elif name == 'divider':
                mappings[name] = ['road_divider', 'lane_divider']
            else:
                mappings[name] = [name]
        return mappings

    def __call__(self, results):
        location = results['location']
        ego2global_translation = results['ego2global_translation']
        ego2global_rotation = results['ego2global_rotation']
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)

        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        mappings = self._layer_mappings()
        layer_names = sorted({ln for v in mappings.values() for ln in v})

        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        ).astype(bool)

        # union nuScenes sub-layers into per-class binary masks
        num_classes = len(self.classes)
        class_masks = np.zeros((num_classes, *self.canvas_size), dtype=bool)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                idx = layer_names.index(layer_name)
                class_masks[k] |= masks[idx]

        # priority-based flatten: low priority first, high priority overwrites
        semantic_indices = np.zeros(self.canvas_size, dtype=np.int64)
        for name in self.priority_order:
            k = self.classes.index(name)
            semantic_indices[class_masks[k]] = self.class_to_index[name]

        results.update({
            'semantic_map': torch.from_numpy(class_masks.astype(np.uint8)),
            'semantic_indices': torch.from_numpy(semantic_indices).long(),
        })
        return results


@PIPELINES.register_module()
class BEVFusionSegmentationV3:
    """4-class BEV segmentation combining HD-map layers and GT 3D box rasterization.

    Semantic index layout (priority-based, higher overwrites lower):
        0: background
        1: road    — drivable_area (road_segment | lane polygon) from HD map
        2: lane    — road_divider + lane_divider (line layers) from HD map
        3: vehicle — truck, bus, motorcycle, bicycle, construction_vehicle, trailer (GT boxes)
        4: car     — vehicle.car only (GT boxes)

    During inference (no gt_bboxes_3d), vehicle/car indices remain 0.

    Canvas coordinate mapping from LiDAR frame (x=fwd, y=left, z=up):
        col = (-y_lidar - xbound[0]) / xbound[2]
        row = (ybound[1] - x_lidar) / ybound[2]
    """

    CAR_NAMES = frozenset({'car'})
    VEHICLE_NAMES = frozenset({
        'truck', 'bus', 'motorcycle', 'bicycle',
        'construction_vehicle', 'trailer',
    })
    CLASS_NAMES = ('road', 'lane', 'vehicle', 'car')
    CLASS_TO_INDEX = {'road': 1, 'lane': 2, 'vehicle': 3, 'car': 4}

    def __init__(self, dataset_root, map_grid_conf,
                 det_class_names=None, lane_width=3):
        xbound = map_grid_conf['xbound']
        ybound = map_grid_conf['ybound']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        self.canvas_size = (int(patch_h / ybound[2]), int(patch_w / xbound[2]))
        self.patch_size = (patch_h, patch_w)
        self.xbound = xbound
        self.ybound = ybound
        self.lane_width = lane_width
        self.det_class_names = det_class_names or [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
        ]
        self.maps = {loc: NuScenesMap(dataset_root, loc) for loc in LOCATIONS}

    def _bev_corners(self, boxes_tensor):
        """Compute 4 BEV corners per box in LiDAR (x=fwd, y=left).

        boxes_tensor: (N, 7+) ndarray  [cx, cy, cz, dx, dy, dz, yaw, ...]
        Returns: (N, 4, 2) corners in LiDAR (x, y) order.
        """
        cx  = boxes_tensor[:, 0]
        cy  = boxes_tensor[:, 1]
        dx  = boxes_tensor[:, 3]
        dy  = boxes_tensor[:, 4]
        yaw = boxes_tensor[:, 6]

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        # 4 corners in box-local frame: (N, 4, 2)
        hx, hy = dx / 2, dy / 2
        offsets = np.stack([
            np.stack([ hx,  hy], axis=1),
            np.stack([ hx, -hy], axis=1),
            np.stack([-hx, -hy], axis=1),
            np.stack([-hx,  hy], axis=1),
        ], axis=1)  # (N, 4, 2)

        # Per-box rotation matrix: (N, 2, 2)
        R = np.stack([
            np.stack([cos_y, -sin_y], axis=1),
            np.stack([sin_y,  cos_y], axis=1),
        ], axis=1)

        # Rotate and translate: (N, 4, 2)
        corners = offsets @ R.transpose(0, 2, 1)
        corners[:, :, 0] += cx[:, None]
        corners[:, :, 1] += cy[:, None]
        return corners

    def _to_cv2_pixels(self, corners_lidar):
        """Convert LiDAR corners (N, 4, 2) to canvas (col, row) int32 for cv2."""
        x = corners_lidar[..., 0]  # forward
        y = corners_lidar[..., 1]  # left
        col = (-y - self.xbound[0]) / self.xbound[2]
        row = (self.ybound[1] - x) / self.ybound[2]
        return np.round(np.stack([col, row], axis=-1)).astype(np.int32)

    def _rasterize_boxes(self, boxes_tensor, labels, target_names,
                         canvas, class_idx):
        """Fill canvas in-place for boxes whose class is in target_names."""
        import cv2
        target_ids = {
            i for i, n in enumerate(self.det_class_names) if n in target_names
        }
        if not target_ids:
            return
        corners = self._bev_corners(boxes_tensor)   # (N, 4, 2) LiDAR xy
        pixels  = self._to_cv2_pixels(corners)      # (N, 4, 2) col,row
        for i, label in enumerate(labels):
            if int(label) in target_ids:
                cv2.fillPoly(canvas, [pixels[i]], class_idx)

    def __call__(self, results):
        import cv2

        location  = results['location']
        map_pose  = np.array(results['ego2global_translation'])[:2]
        rotation  = Quaternion(results['ego2global_rotation'])
        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        # ── static map: drivable area (road_segment | lane polygon) ───
        road_raw = self.maps[location].get_map_mask(
            patch_box, patch_angle,
            ['road_segment', 'lane'], self.canvas_size,
        ).astype(bool)
        road_mask = road_raw[0] | road_raw[1]

        # ── static map: lane/road divider lines ───────────────────────
        lane_raw = self.maps[location].get_map_mask(
            patch_box, patch_angle,
            ['road_divider', 'lane_divider'], self.canvas_size,
        ).astype(bool)
        lane_mask = lane_raw[0] | lane_raw[1]
        if self.lane_width > 1:
            k = np.ones((self.lane_width, self.lane_width), np.uint8)
            lane_mask = cv2.dilate(
                lane_mask.astype(np.uint8), k).astype(bool)

        # ── priority flatten: road first, lane overwrites ──────────────
        semantic_indices = np.zeros(self.canvas_size, dtype=np.int32)
        semantic_indices[road_mask] = self.CLASS_TO_INDEX['road']
        semantic_indices[lane_mask] = self.CLASS_TO_INDEX['lane']

        # ── dynamic objects from GT boxes (train-time only) ────────────
        gt_bboxes_3d = results.get('gt_bboxes_3d', None)
        gt_labels_3d = results.get('gt_labels_3d', None)
        if gt_bboxes_3d is not None and gt_labels_3d is not None:
            # Unwrap DataContainer if already wrapped (shouldn't happen here,
            # but guard for safety).
            boxes  = gt_bboxes_3d.data if hasattr(gt_bboxes_3d, 'data') else gt_bboxes_3d
            labels = gt_labels_3d.data if hasattr(gt_labels_3d, 'data') else gt_labels_3d
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            if len(boxes) > 0:
                btensor = boxes.tensor.numpy()  # (N, 7)
                self._rasterize_boxes(
                    btensor, labels, self.VEHICLE_NAMES,
                    semantic_indices, self.CLASS_TO_INDEX['vehicle'])
                self._rasterize_boxes(
                    btensor, labels, self.CAR_NAMES,
                    semantic_indices, self.CLASS_TO_INDEX['car'])

        # ── multi-channel binary mask (road/lane/vehicle/car) ─────────
        n = len(self.CLASS_NAMES)
        class_masks = np.zeros((n, *self.canvas_size), dtype=np.uint8)
        for i, name in enumerate(self.CLASS_NAMES):
            class_masks[i] = (semantic_indices == self.CLASS_TO_INDEX[name])

        results.update({
            'semantic_map': torch.from_numpy(class_masks),
            'semantic_indices': torch.from_numpy(
                semantic_indices.astype(np.int64)),
        })
        return results