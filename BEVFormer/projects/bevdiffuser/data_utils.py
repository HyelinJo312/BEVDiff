# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import numpy as np
import os
from transformers import CLIPTokenizer, CLIPTextModel
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.nuscenes_dataset_v2 import CustomNuScenesDatasetV2
import cv2
from pyquaternion import Quaternion
from projects.mmdet3d_plugin.datasets.nuscenes_occ import NuSceneOcc


def get_content_shapes(img_metas):
    """Per-camera (H, W, C) of the real image content inside the padded frame.

    """
    img_shapes = img_metas['img_shape']
    pad_shapes = img_metas['pad_shape']
    ori_shapes = img_metas.get('ori_shape', None) if hasattr(img_metas, 'get') else None
    if ori_shapes is None or len(ori_shapes) != len(pad_shapes):
        return img_shapes
    out = []
    for i, (ori, pad) in enumerate(zip(ori_shapes, pad_shapes)):
        out.append(ori if (ori[0] <= pad[0] and ori[1] <= pad[1]) else img_shapes[i])
    return out


@DATASETS.register_module()
class CustomNuScenesDiffusionDataset_layout(CustomNuScenesDataset): 
    def __init__(self, pc_range, use_3d_bbox=True, num_classes=12, num_bboxes=300, use_layout=True, use_semantics=False, semantic_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox=use_3d_bbox
        self.num_classes=num_classes
        self.num_bboxes=num_bboxes
        self.object_names = list(self.CLASSES) + ['__image__', '__null__']
        # self.object_names = list(text_prompt) + ['__image__', '__null__']
        self.object_clips = self.embed_object_names()
        self.use_layout = use_layout
        self.use_semantics = use_semantics
        self.semantic_path = semantic_path

        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.use_layout:
            layout = self.get_layout_info(data)
            for info in layout.keys():
                data[info] = layout[info]

        # Segmentation mask
        if self.use_semantics and self.semantic_path is not None:
            img_metas_raw = data['img_metas']
            img_metas_list = img_metas_raw.data if isinstance(img_metas_raw, DC) else img_metas_raw
            if isinstance(img_metas_list, dict):
                current_metas = img_metas_list[max(img_metas_list.keys())]
            else:
                current_metas = img_metas_list[-1]
            if isinstance(current_metas, DC):
                current_metas = current_metas.data
            seg_maps = self.load_segmaps(
                current_metas['filename'],
                current_metas,
                self.semantic_path,
            )
            data['seg_maps'] = DC(seg_maps)

        return data
    
    def embed_object_names(self):
        pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2-1'
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder.requires_grad_(False)
        object_tokens = tokenizer(self.object_names, 
                                  max_length=tokenizer.model_max_length,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt").input_ids
        object_clip_embed = text_encoder(object_tokens)[1]
        return object_clip_embed # (N, 1024)
    
    def get_layout_info(self, data):
        # data['gt_labels_3d'] should be a DC of a tensor with size [N]
        # data['gt_bboxes_3d'] should be a DC of LiDARInstance3DBoxes of a tensor with size [N, 9]: (x, y, z, x_size, y_size, z_size, yaw, vx, vy) 
        class_ids = torch.tensor([])
        gt_bboxes = torch.tensor([])
        if 'gt_labels_3d' in data:
            while not isinstance(data['gt_labels_3d'], DC):
                data['gt_labels_3d'] = data['gt_labels_3d'][0]
            class_ids = data['gt_labels_3d'].data
            class_ids = (class_ids + len(self.CLASSES))%len(self.CLASSES)
        if 'gt_bboxes_3d' in data:
            while not isinstance(data['gt_bboxes_3d'], DC):
                data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0]
            gt_bboxes = data['gt_bboxes_3d'].data
            
        layout_obj_classes = torch.LongTensor(self.num_bboxes).fill_(self.num_classes-1)
        layout_is_valid = torch.zeros([self.num_bboxes])
        
        layout_obj_classes[0] = self.num_classes-2
        layout_is_valid[0] = 1.0
        # default_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
        
        num_valid = min(len(class_ids), self.num_bboxes-1)
        layout_obj_classes[1: 1+num_valid] = class_ids
        layout_is_valid[1: 1+num_valid] = 1.0
        
        layout_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
       
        
        if self.use_3d_bbox:
            layout_obj_bboxes = self.get_3d_layout_bboxes(gt_bboxes)
        else:
            layout_obj_bboxes = self.get_2d_layout_bboxes(gt_bboxes)
            
        layout = {
            'layout_obj_classes': DC(layout_obj_classes),
            'layout_obj_bboxes': DC(layout_obj_bboxes),
            'layout_obj_is_valid': DC(layout_is_valid),
            # 'layout_obj_names': DC(layout_obj_clip),
            # 'default_obj_names': DC(default_obj_clip),
        }

        return layout
    
    def normalize_bbox(self, bbox):
        # normalize bbox into [0,1], ego at [0.5, 0.5] 
        x, y = torch.tensor_split(bbox[..., :2], 2, dim=-1)
        x = (x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y = (y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        if bbox.shape[-1] > 2:
            z, x_size, y_size, z_size, yaw, vx, vy = torch.tensor_split(bbox[..., 2:], 7, dim=-1)
            z = (z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            x_size = x_size / (self.pc_range[3] - self.pc_range[0])
            y_size = y_size / (self.pc_range[4] - self.pc_range[1])
            z_size = z_size / (self.pc_range[5] - self.pc_range[2])
            return torch.cat((x, y, z, x_size, y_size, z_size, yaw, vx, vy), dim=-1)
        return torch.cat((x, y), dim=-1)
            
    
    def get_3d_layout_bboxes(self, gt_bboxes): 
        # 3d bbox: (xc, yc, zc, x_size, y_size, z_size, yaw, vx, vy) 
        # ego coordinate, origin at ego position, x towards right, y towards front 
        layout_bboxes = torch.zeros([self.num_bboxes, 9])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            # (x, y) -> (x-0.5, x-0.5)
            gt_bboxes = self.normalize_bbox(gt_bboxes.tensor)
            gt_bboxes[..., :2] = gt_bboxes[..., :2] - 0.5
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
    
    def get_2d_layout_bboxes(self, gt_bboxes):
        # 2d bbox: (x0, y0, x1, y1), 
        # image coordinate, orgin at upper left, x towards right, y towards down
        layout_bboxes = torch.zeros([self.num_bboxes, 4])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 1, 1])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            gt_bboxes = self.normalize_bbox(gt_bboxes.corners[..., :2]) # N x 8 x 2
            # (x, y) -> (x, 1-y)
            gt_bboxes[..., 1] = 1 - gt_bboxes[..., 1]
            gt_bboxes_min = gt_bboxes.min(dim=1).values # N x 2
            gt_bboxes_max = gt_bboxes.max(dim=1).values # N x 2
            gt_bboxes = torch.cat((gt_bboxes_min, gt_bboxes_max), dim=-1)
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
            
    def load_segmaps(self, filenames, img_metas, semantic_root):
        """
        Args:
            filenames:     list[str] — per-camera image paths, e.g.  '.../samples/CAM_FRONT/xxx.jpg'
            img_metas:     dict — contains
                             'ori_shape': list of (H, W, C) after scale, before pad
                             'pad_shape': list of (H, W, C) after pad
                             ('img_shape' also equals the padded size here)
            semantic_root: str — root directory for seg maps

        Returns:
            LongTensor of shape [V, H_pad, W_pad]
        """
        from PIL import Image

        ignore_label = -1
        H, W = 900, 1600

        seg_maps = []
        # BUGFIX: 'img_shape' holds the PADDED size (PadMultiViewImage overwrites
        # it), not the content size, so resizing the mask to it stretched the seg
        # map across the pad region and shifted labels upward relative to the
        # image (~24 px at the bottom row). See get_content_shapes().
        # img_shapes = img_metas['img_shape']  # list[(H, W, C)] per cam
        img_shapes = get_content_shapes(img_metas)  # pre-pad content (H, W, C) per cam
        pad_shapes = img_metas['pad_shape']  # list[(H, W, C)] per cam

        for i, fname in enumerate(filenames):
            # Derive seg map path: '.../samples/CAM_XXX/name.jpg' → 'samples/CAM_XXX/name.bin'
            rel = fname.split('samples/')[-1]        # 'CAM_FRONT/xxx.jpg'
            base = os.path.splitext(rel)[0]          # 'CAM_FRONT/xxx'
            seg_path = os.path.join(semantic_root, 'samples', base + '_mask.bin')

            h_scale, w_scale = img_shapes[i][:2]
            h_pad, w_pad = pad_shapes[i][:2]

            # Load binary seg map (int8 on disk, promote to int16 to hold -1 cleanly)
            try:
                x = np.fromfile(seg_path, dtype=np.int8)
                if x.size != H * W:
                    mask = np.full((H, W), ignore_label, dtype=np.int16)
                else:
                    mask = x.reshape(H, W).astype(np.int16)
            except FileNotFoundError:
                print(f'Warning: seg map not found at {seg_path}, filling with ignore_label')
                mask = np.full((H, W), ignore_label, dtype=np.int16)

            # Optional raw seg-id remap (e.g. SAM3 -> model taxonomy). Tested against
            # the ORIGINAL mask so overlapping src/dst (e.g. 16->2 and 19->16) do not
            # cascade. Ids not in the dict (incl. 0 background, -1 ignore) pass through.
            seg_id_remap = getattr(self, 'seg_id_remap', None)
            if seg_id_remap:
                remapped = mask.copy()
                for src, dst in seg_id_remap.items():
                    remapped[mask == src] = dst
                mask = remapped

            # Resize to img_shape using NEAREST to preserve class IDs.
            # PIL mode 'I' (int32) is used because mode 'I;16' treats values as
            # unsigned and would corrupt negative ignore_label (-1 → 65535).
            seg_pil = Image.fromarray(mask.astype(np.int32))   # mode 'I'
            seg_pil = seg_pil.resize((w_scale, h_scale), Image.NEAREST)
            seg_arr = np.array(seg_pil, dtype=np.int16)

            # Pad to pad_shape, fill with ignore_label (mirrors PadMultiViewImage)
            pad_h = h_pad - h_scale
            pad_w = w_pad - w_scale
            seg_arr = np.pad(seg_arr, ((0, pad_h), (0, pad_w)),
                             mode='constant', constant_values=ignore_label)

            seg_maps.append(torch.tensor(seg_arr, dtype=torch.long))

        return torch.stack(seg_maps, dim=0)  # [V, H_pad, W_pad]
            
            
@DATASETS.register_module()
class CustomNuScenesDiffusionDataset_layout_seg(CustomNuScenesDataset): 
    def __init__(self, pc_range, use_3d_bbox=True, num_classes=18, num_bboxes=300, use_layout=True, use_semantics=True, use_depth=False,
                 semantic_path=None, depth_path=None, seg_class=None, total_class=None, seg_id_remap=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox=use_3d_bbox
        self.num_classes=num_classes
        self.num_bboxes=num_bboxes
        self.total_class = total_class
        self.seg_class = seg_class
        self.seg_id_remap = seg_id_remap
        self.object_names = list(self.total_class) + ['__image__', '__null__']
        self.object_clips = self.embed_object_names()
        self.use_layout = use_layout
        self.use_semantics = use_semantics
        self.use_depth = use_depth
        self.semantic_path = semantic_path
        self.depth_path = depth_path

        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # Segmentation mask
        if self.use_semantics and self.semantic_path is not None:
            img_metas_raw = data['img_metas']
            img_metas_list = img_metas_raw.data if isinstance(img_metas_raw, DC) else img_metas_raw
            if isinstance(img_metas_list, dict):
                current_metas = img_metas_list[max(img_metas_list.keys())]
            else:
                current_metas = img_metas_list[-1]
            if isinstance(current_metas, DC):
                current_metas = current_metas.data
            seg_maps = self.load_segmaps(
                current_metas['filename'],
                current_metas,
                self.semantic_path,
            )
            data['seg_maps'] = DC(seg_maps)
            
        if self.use_layout:
            layout = self.get_layout_info(data, seg_maps.unique().tolist())
            for info in layout.keys():
                data[info] = layout[info]
        
        if self.use_depth and self.depth_path is not None:
            img_metas_raw = data['img_metas']
            img_metas_list = img_metas_raw.data if isinstance(img_metas_raw, DC) else img_metas_raw
            if isinstance(img_metas_list, dict):
                depth_metas = img_metas_list[max(img_metas_list.keys())]
            else:
                depth_metas = img_metas_list[-1]
            if isinstance(depth_metas, DC):
                depth_metas = depth_metas.data
            depth_maps = self.load_depth_from_filenames(depth_metas['filename'])
            data['depth_maps'] = DC(depth_maps)
        
        return data
    
    def embed_object_names(self):
        pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2-1'
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder.requires_grad_(False)
        object_tokens = tokenizer(self.object_names, 
                                  max_length=tokenizer.model_max_length,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt").input_ids
        object_clip_embed = text_encoder(object_tokens)[1]
        return object_clip_embed # (N, 1024)
    
    def get_layout_info(self, data, seg_class_valid):
        # data['gt_labels_3d'] should be a DC of a tensor with size [N]
        # data['gt_bboxes_3d'] should be a DC of LiDARInstance3DBoxes of a tensor with size [N, 9]: (x, y, z, x_size, y_size, z_size, yaw, vx, vy) 
        class_ids = torch.tensor([])
        gt_bboxes = torch.tensor([])
        if 'gt_labels_3d' in data:
            while not isinstance(data['gt_labels_3d'], DC):
                data['gt_labels_3d'] = data['gt_labels_3d'][0]
            class_ids = data['gt_labels_3d'].data
            class_ids = (class_ids + len(self.CLASSES))%len(self.CLASSES)
        if 'gt_bboxes_3d' in data:
            while not isinstance(data['gt_bboxes_3d'], DC):
                data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0]
            gt_bboxes = data['gt_bboxes_3d'].data
            
        layout_obj_classes = torch.LongTensor(self.num_bboxes).fill_(self.num_classes-1)
        layout_is_valid = torch.zeros([self.num_bboxes])
        
        layout_obj_classes[0] = self.num_classes-2
        layout_is_valid[0] = 1.0
        # default_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
        
        num_valid = min(len(class_ids), self.num_bboxes-1)
        layout_obj_classes[1: 1+num_valid] = class_ids
        
        n = 1+num_valid
        for i in self.seg_class.keys():
            if i in seg_class_valid:
                layout_obj_classes[n] = self.seg_class[i][-1]
                n += 1
        
        layout_is_valid[1: 1+num_valid] = 1.0
        
        layout_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
       
        
        if self.use_3d_bbox:
            layout_obj_bboxes = self.get_3d_layout_bboxes(gt_bboxes)
        else:
            layout_obj_bboxes = self.get_2d_layout_bboxes(gt_bboxes)
            
        layout = {
            'layout_obj_classes': DC(layout_obj_classes),
            'layout_obj_bboxes': DC(layout_obj_bboxes),
            'layout_obj_is_valid': DC(layout_is_valid),
            'layout_obj_names': DC(layout_obj_clip),
            # 'default_obj_names': DC(default_obj_clip),
        }

        return layout
    
    def normalize_bbox(self, bbox):
        # normalize bbox into [0,1], ego at [0.5, 0.5] 
        x, y = torch.tensor_split(bbox[..., :2], 2, dim=-1)
        x = (x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y = (y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        if bbox.shape[-1] > 2:
            z, x_size, y_size, z_size, yaw, vx, vy = torch.tensor_split(bbox[..., 2:], 7, dim=-1)
            z = (z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            x_size = x_size / (self.pc_range[3] - self.pc_range[0])
            y_size = y_size / (self.pc_range[4] - self.pc_range[1])
            z_size = z_size / (self.pc_range[5] - self.pc_range[2])
            return torch.cat((x, y, z, x_size, y_size, z_size, yaw, vx, vy), dim=-1)
        return torch.cat((x, y), dim=-1)
            
    
    def get_3d_layout_bboxes(self, gt_bboxes): 
        # 3d bbox: (xc, yc, zc, x_size, y_size, z_size, yaw, vx, vy) 
        # ego coordinate, origin at ego position, x towards right, y towards front 
        layout_bboxes = torch.zeros([self.num_bboxes, 9])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            # (x, y) -> (x-0.5, x-0.5)
            gt_bboxes = self.normalize_bbox(gt_bboxes.tensor)
            gt_bboxes[..., :2] = gt_bboxes[..., :2] - 0.5
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
    
    def get_2d_layout_bboxes(self, gt_bboxes):
        # 2d bbox: (x0, y0, x1, y1), 
        # image coordinate, orgin at upper left, x towards right, y towards down
        layout_bboxes = torch.zeros([self.num_bboxes, 4])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 1, 1])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            gt_bboxes = self.normalize_bbox(gt_bboxes.corners[..., :2]) # N x 8 x 2
            # (x, y) -> (x, 1-y)
            gt_bboxes[..., 1] = 1 - gt_bboxes[..., 1]
            gt_bboxes_min = gt_bboxes.min(dim=1).values # N x 2
            gt_bboxes_max = gt_bboxes.max(dim=1).values # N x 2
            gt_bboxes = torch.cat((gt_bboxes_min, gt_bboxes_max), dim=-1)
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
            
    def load_depth_from_filenames(self, filenames):
        """Load multi-view depth maps (.npy, float16) from nuscenes_depth_da3.

        Args:
            filenames: list[str] — per-camera image paths, e.g.
                       '.../samples/CAM_FRONT/xxx.jpg'  (length = V)

        Returns:
            FloatTensor of shape [V, H, W]  (H=448, W=798)
        """
        view_depths = []
        for fname in filenames:
            # '.../samples/CAM_FRONT/xxx.jpg' → 'CAM_FRONT/xxx'
            rel = fname.split('samples/')[-1]
            base = os.path.splitext(rel)[0]
            npy_path = os.path.join(self.depth_path, 'samples', base + '.npy')

            try:
                depth = np.load(npy_path, mmap_mode='r')
                depth_tensor = torch.from_numpy(depth.copy()).float()
            except FileNotFoundError:
                print(f'Warning: depth map not found at {npy_path}, filling with zeros')
                depth_tensor = torch.zeros(448, 798, dtype=torch.float32)

            view_depths.append(depth_tensor)

        return torch.stack(view_depths, dim=0)  # [V, H, W]

    def load_segmaps(self, filenames, img_metas, semantic_root):
        """Load multi-view segmentation maps (.bin) and apply the same spatial
        transforms as the image pipeline (nearest-neighbor resize then pad).

        Args:
            filenames:     list[str] — per-camera image paths, e.g.
                           '.../samples/CAM_FRONT/xxx.jpg'
            img_metas:     dict — contains
                             'ori_shape': list of (H, W, C) after scale, before pad
                             'pad_shape': list of (H, W, C) after pad
                             ('img_shape' also equals the padded size here)
            semantic_root: str — root directory for seg maps

        Returns:
            LongTensor of shape [V, H_pad, W_pad]
        """
        from PIL import Image

        ignore_label = -1
        H, W = 900, 1600

        seg_maps = []
        # BUGFIX: 'img_shape' holds the PADDED size (PadMultiViewImage overwrites
        # img_shapes = img_metas['img_shape']
        img_shapes = get_content_shapes(img_metas)  # pre-pad content (H, W, C) per cam
        pad_shapes = img_metas['pad_shape']  # list[(H, W, C)] per cam

        for i, fname in enumerate(filenames):
            # Derive seg map path: '.../samples/CAM_XXX/name.jpg' → 'samples/CAM_XXX/name.bin'
            rel = fname.split('samples/')[-1]        # 'CAM_FRONT/xxx.jpg'
            base = os.path.splitext(rel)[0]          # 'CAM_FRONT/xxx'
            seg_path = os.path.join(semantic_root, 'samples', base + '_mask.bin')

            h_scale, w_scale = img_shapes[i][:2]
            h_pad, w_pad = pad_shapes[i][:2]

            # Load binary seg map (int8 on disk, promote to int16 to hold -1 cleanly)
            try:
                x = np.fromfile(seg_path, dtype=np.int8)
                if x.size != H * W:
                    mask = np.full((H, W), ignore_label, dtype=np.int16)
                else:
                    mask = x.reshape(H, W).astype(np.int16)
            except FileNotFoundError:
                print(f'Warning: seg map not found at {seg_path}, filling with ignore_label')
                mask = np.full((H, W), ignore_label, dtype=np.int16)

            # Optional raw seg-id remap (e.g. SAM3 -> model taxonomy). 
            seg_id_remap = getattr(self, 'seg_id_remap', None)
            if seg_id_remap:
                remapped = mask.copy()
                for src, dst in seg_id_remap.items():
                    remapped[mask == src] = dst
                mask = remapped

            # Resize to img_shape using NEAREST to preserve class IDs.
            # PIL mode 'I' (int32) is used because mode 'I;16' treats values as
            # unsigned and would corrupt negative ignore_label (-1 → 65535).
            seg_pil = Image.fromarray(mask.astype(np.int32))   # mode 'I'
            seg_pil = seg_pil.resize((w_scale, h_scale), Image.NEAREST)
            seg_arr = np.array(seg_pil, dtype=np.int16)

            # Pad to pad_shape, fill with ignore_label (mirrors PadMultiViewImage)
            pad_h = h_pad - h_scale
            pad_w = w_pad - w_scale
            seg_arr = np.pad(seg_arr, ((0, pad_h), (0, pad_w)),
                             mode='constant', constant_values=ignore_label)

            seg_maps.append(torch.tensor(seg_arr, dtype=torch.long))

        return torch.stack(seg_maps, dim=0)  # [V, H_pad, W_pad]



@DATASETS.register_module()
class CustomNuScenesDiffusionDataset_seg_depth(CustomNuScenesDataset):
    """Multi-view segmentation map + DA3 depth map loader WITHOUT layout.

    Intended for the DINO+Seg diffusion UNet (dinoseg_diffusion_unet), which
    conditions on DINOv2 features + seg BEV + (optional) depth consistency and
    has no layout-encoder path. Compared to the *_layout / *_layout_seg
    datasets, this class skips all layout/object-CLIP processing (no
    embed_object_names / get_layout_info), so it neither needs seg_class /
    total_class nor loads the CLIP text encoder.

    Produces (in addition to the base pipeline outputs):
        - data['seg_maps']:   LongTensor [V, H_pad, W_pad]   (if use_semantics)
        - data['depth_maps']: FloatTensor [V, dH, dW]        (if use_depth)
    """
    def __init__(self, pc_range, use_3d_bbox=True, num_classes=12, num_bboxes=300, use_semantics=True, use_depth=False,
                 semantic_path=None, depth_path=None, seg_id_remap=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox=use_3d_bbox
        self.num_classes=num_classes
        self.num_bboxes=num_bboxes
        self.use_semantics = use_semantics
        self.use_depth = use_depth
        self.semantic_path = semantic_path
        self.depth_path = depth_path
        self.seg_id_remap = seg_id_remap

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Segmentation maps [V, H, W]
        if self.use_semantics and self.semantic_path is not None:
            img_metas_raw = data['img_metas']
            img_metas_list = img_metas_raw.data if isinstance(img_metas_raw, DC) else img_metas_raw
            if isinstance(img_metas_list, dict):
                current_metas = img_metas_list[max(img_metas_list.keys())]
            else:
                current_metas = img_metas_list[-1]
            if isinstance(current_metas, DC):
                current_metas = current_metas.data
            seg_maps = self.load_segmaps(
                current_metas['filename'],
                current_metas,
                self.semantic_path,
            )
            data['seg_maps'] = DC(seg_maps)

        # DA3 depth maps [V, dH, dW] for FB-BEV depth consistency
        if self.use_depth and self.depth_path is not None:
            img_metas_raw = data['img_metas']
            img_metas_list = img_metas_raw.data if isinstance(img_metas_raw, DC) else img_metas_raw
            if isinstance(img_metas_list, dict):
                depth_metas = img_metas_list[max(img_metas_list.keys())]
            else:
                depth_metas = img_metas_list[-1]
            if isinstance(depth_metas, DC):
                depth_metas = depth_metas.data
            depth_maps = self.load_depth_from_filenames(depth_metas['filename'])
            data['depth_maps'] = DC(depth_maps)

        return data

    def load_segmaps(self, filenames, img_metas, semantic_root):
        """Load multi-view seg maps (.bin) and apply image-pipeline transforms
        (nearest-neighbor resize then pad). Returns LongTensor [V, H_pad, W_pad].
        """
        from PIL import Image

        ignore_label = -1
        H, W = 900, 1600

        seg_maps = []
        # BUGFIX: 'img_shape' holds the PADDED size 
        # img_shapes = img_metas['img_shape']
        img_shapes = get_content_shapes(img_metas)  # pre-pad content (H, W, C) per cam
        pad_shapes = img_metas['pad_shape']  # list[(H, W, C)] per cam

        for i, fname in enumerate(filenames):
            rel = fname.split('samples/')[-1]        # 'CAM_FRONT/xxx.jpg'
            base = os.path.splitext(rel)[0]          # 'CAM_FRONT/xxx'
            seg_path = os.path.join(semantic_root, 'samples', base + '_mask.bin')

            h_scale, w_scale = img_shapes[i][:2]
            h_pad, w_pad = pad_shapes[i][:2]

            try:
                x = np.fromfile(seg_path, dtype=np.int8)
                if x.size != H * W:
                    mask = np.full((H, W), ignore_label, dtype=np.int16)
                else:
                    mask = x.reshape(H, W).astype(np.int16)
            except FileNotFoundError:
                print(f'Warning: seg map not found at {seg_path}, filling with ignore_label')
                mask = np.full((H, W), ignore_label, dtype=np.int16)

            # Optional raw seg-id remap (e.g. SAM3 -> model taxonomy).
            seg_id_remap = getattr(self, 'seg_id_remap', None)
            if seg_id_remap:
                remapped = mask.copy()
                for src, dst in seg_id_remap.items():
                    remapped[mask == src] = dst
                mask = remapped

            # NEAREST resize preserves class IDs; PIL mode 'I' keeps -1 intact.
            seg_pil = Image.fromarray(mask.astype(np.int32))   # mode 'I'
            seg_pil = seg_pil.resize((w_scale, h_scale), Image.NEAREST)
            seg_arr = np.array(seg_pil, dtype=np.int16)

            pad_h = h_pad - h_scale
            pad_w = w_pad - w_scale
            seg_arr = np.pad(seg_arr, ((0, pad_h), (0, pad_w)),
                             mode='constant', constant_values=ignore_label)

            seg_maps.append(torch.tensor(seg_arr, dtype=torch.long))

        return torch.stack(seg_maps, dim=0)  # [V, H_pad, W_pad]

    def load_depth_from_filenames(self, filenames):
        """Load multi-view DA3 depth maps (.npy) from `self.depth_path`.
        Returns FloatTensor [V, H, W] (e.g. H=448, W=798).
        """
        view_depths = []
        for fname in filenames:
            rel = fname.split('samples/')[-1]
            base = os.path.splitext(rel)[0]
            npy_path = os.path.join(self.depth_path, 'samples', base + '.npy')
            try:
                depth = np.load(npy_path, mmap_mode='r')
                depth_tensor = torch.from_numpy(depth.copy()).float()
            except FileNotFoundError:
                print(f'Warning: depth map not found at {npy_path}, filling with zeros')
                depth_tensor = torch.zeros(448, 798, dtype=torch.float32)
            view_depths.append(depth_tensor)
        return torch.stack(view_depths, dim=0)  # [V, H, W]

           
def _retain_meta_key(transform, key):
    if type(transform).__name__ == 'CustomCollect3D':
        if key not in transform.meta_keys:
            transform.meta_keys = tuple(transform.meta_keys) + (key,)
        return 1
    children = getattr(transform, 'transforms', ())
    if isinstance(children, (list, tuple)):
        return sum(_retain_meta_key(child, key) for child in children)
    return _retain_meta_key(children, key)


def _unwrap_singleton(value):
    while isinstance(value, DC) or (isinstance(value, (list, tuple)) and len(value) == 1):
        value = value.data if isinstance(value, DC) else value[0]
    return value


def _last_occurrence_mask(tokens):
    seen = set()
    valid = [False] * len(tokens)
    for index in range(len(tokens) - 1, -1, -1):
        valid[index] = tokens[index] not in seen
        seen.add(tokens[index])
    return valid


@DATASETS.register_module()
class CustomNuScenesDiffusionDataset_seg_depth_v2(CustomNuScenesDataset):
    """NuScenes images with aligned SAM3/Metric3D maps or cached semantic BEV."""

    def __init__(self, pc_range, use_3d_bbox=True, num_classes=12, num_bboxes=300,
                 use_semantics=True, use_depth=False, semantic_path=None,
                 depth_path=None, seg_id_remap=None, *args,
                 depth_raw_shape=(900, 1600), depth_range=(2.5, 170.0),
                 opencv_num_threads=1, camera_types=None,
                 use_semantic_bev_cache=False, semantic_bev_cache_root=None,
                 semantic_bev_cache_shape=None, semantic_bev_cache_projection=None,
                 load_depth_in_cache=False, load_temporal_depth=False,
                 use_semantic_temporal=False, semantic_type='share',
                 semantic_logcount_scale=1e5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox = use_3d_bbox
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        self.use_semantics = use_semantics
        self.use_depth = use_depth
        self.semantic_path = semantic_path
        self.depth_path = depth_path
        self.seg_id_remap = seg_id_remap
        self.depth_raw_shape = tuple(depth_raw_shape)
        self.depth_range = tuple(depth_range)
        self.opencv_num_threads = int(opencv_num_threads)
        self.use_semantic_bev_cache = use_semantic_bev_cache
        self.semantic_bev_cache_root = semantic_bev_cache_root
        self.load_depth_in_cache = bool(load_depth_in_cache)
        self.load_temporal_depth = bool(load_temporal_depth)

        # Ego-aligned temporal semantic BEV. Off by default: when False nothing
        # below runs and every emitted key is byte-identical to before.
        self.use_semantic_temporal = bool(use_semantic_temporal)
        self._info_by_token = None
        if self.use_semantic_temporal:
            # Queue frames are picked at random by prepare_train_data(), so the
            # rigid transform has to be looked up per token rather than by index.
            self._info_by_token = {info['token']: info for info in self.data_infos}

        # Change only the in-memory annotation order, before calibration is built.
        if camera_types is not None:
            for info in self.data_infos:
                info['cams'] = {camera: info['cams'][camera] for camera in camera_types}

        # Accept older configs that specify a projector instead of a cache shape.
        if semantic_bev_cache_shape is None and semantic_bev_cache_projection is not None:
            projection = semantic_bev_cache_projection
            semantic_bev_cache_shape = (projection['num_classes'] + 1,
                                       projection['bev_h'], projection['bev_w'])
        self.semantic_bev_cache_shape = (tuple(semantic_bev_cache_shape)
                                        if semantic_bev_cache_shape is not None else None)
        configure_semantic_type(self, semantic_type, semantic_logcount_scale,
                                self.use_semantic_bev_cache, semantic_bev_cache_root)

        needs_raw_maps = self.load_depth_in_cache or self.load_temporal_depth or (
            not self.use_semantic_bev_cache and (self.use_semantics or self.use_depth))
        
        if needs_raw_maps and not self._retain_raw_projection(self.pipeline):
            raise ValueError('Metric3D dataset requires CustomCollect3D to retain raw calibration')
        if self.load_temporal_depth:
            self._retain_temporal_pose(self.pipeline)

    def load_semantic_bev(self, token):
        """Read a cached sample and return the model's semantic input.

        semantic_bev_cache_shape is the share shape [C, H, W]. The stored array is
        [C, H, W] for the legacy 'share' layout or [2C+1, H, W] for
        'share+counts+support'; the returned input is [C, H, W] for
        semantic_type='share' and [2C, H, W] for 'share+logcount'.
        """
        from .semantic_bev_cache import split_cache_array
        from .layout_diffusion.seg_bev_aligner_one_hot_v5 import semantic_input

        path = os.path.join(self.semantic_bev_cache_root, f'{token}.npy')
        value = np.load(path, allow_pickle=False)
        layout = getattr(self, 'semantic_bev_cache_layout', 'share')
        expected_shape = self.semantic_bev_cache_shape
        if value.ndim != 3 or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f'Expected floating semantic BEV [C, H, W] at {path}, '
                             f'got {value.shape} {value.dtype}')
        if expected_shape is not None and getattr(self, 'semantic_type', 'share') == 'share':
            if value.shape == (2 * expected_shape[0] + 1, *expected_shape[1:]):
                layout = 'share+counts+support'
        if expected_shape is not None:
            classes = expected_shape[0]
        elif layout == 'share':
            classes = value.shape[0]
        else:
            classes = (value.shape[0] - 1) // 2
        stored_shape = (classes if layout == 'share' else 2 * classes + 1,
                        *(expected_shape[1:] if expected_shape is not None else value.shape[1:]))
        if value.shape != stored_shape:
            raise ValueError(f'Expected floating semantic BEV {stored_shape} at {path}, '
                             f'got {value.shape} {value.dtype}')
        share, counts, _ = split_cache_array(value, classes, layout)
        semantic = semantic_input(share, counts, getattr(self, 'semantic_type', 'share'),
                                  getattr(self, 'semantic_logcount_scale', 1e5))
        return np.ascontiguousarray(semantic, dtype=np.float32)

    @staticmethod
    def lidar2global(info):
        """4x4 lidar -> global transform for one sample.

        Semantic BEV is cached in each frame's own lidar frame, so fusing frame
        t-1 into frame t needs
            T_rel = inv(lidar2global(t)) @ lidar2global(t-1)
        lidar2ego is NOT constant across the dataset (three sensor calibrations
        appear across train+val), so it must come from that sample's own info.
        """
        ego = np.eye(4, dtype=np.float64)
        ego[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego[:3, 3] = np.asarray(info['ego2global_translation'], dtype=np.float64)
        lidar = np.eye(4, dtype=np.float64)
        lidar[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar[:3, 3] = np.asarray(info['lidar2ego_translation'], dtype=np.float64)
        return ego @ lidar

    def _temporal_semantic_bev(self, temporal_metas):
        """Queue-frame semantic BEV plus what is needed to ego-align it.

        Returns (probabilities [K, C, H, W], lidar2global [K, 4, 4],
                 timestamp [K], valid [K]).
        """
        tokens = [frame['sample_idx'] for frame in temporal_metas]
        probabilities = np.stack([self.load_semantic_bev(t) for t in tokens])
        transforms = np.stack([self.lidar2global(self._info_by_token[t]) for t in tokens])
        timestamps = np.asarray([self._info_by_token[t]['timestamp'] / 1e6 for t in tokens],
                                dtype=np.float64)

        return probabilities, transforms, timestamps, np.asarray(_last_occurrence_mask(tokens), dtype=bool)

    @classmethod
    def _retain_raw_projection(cls, transform):
        return _retain_meta_key(transform, 'metric3d_lidar2img_raw')

    @classmethod
    def _retain_temporal_pose(cls, transform):
        return _retain_meta_key(transform, 'lidar2global')

    def get_data_info(self, index):
        info = super().get_data_info(index)
        if info is not None and 'lidar2img' in info:
            info['metric3d_lidar2img_raw'] = [matrix.copy() for matrix in info['lidar2img']]
        if info is not None and self.load_temporal_depth:
            info['lidar2global'] = self.lidar2global(self.data_infos[index])
        return info

    @staticmethod
    def _current_metas(raw):
        raw = _unwrap_singleton(raw)
        if isinstance(raw, dict) and 'filename' in raw:
            return raw
        if isinstance(raw, dict):
            return CustomNuScenesDiffusionDataset_seg_depth_v2._current_metas(raw[max(raw)])
        raise ValueError('Expected temporal training metadata or one test augmentation')

    @staticmethod
    def _temporal_metas(raw):
        raw = _unwrap_singleton(raw)
        if isinstance(raw, dict) and 'filename' in raw:
            return [raw]
        if isinstance(raw, dict):
            return [CustomNuScenesDiffusionDataset_seg_depth_v2._current_metas(
                raw[index]) for index in sorted(raw)]
        raise ValueError('Expected temporal training metadata or one test augmentation')

    def __getitem__(self, idx):
        # Configure OpenCV in each worker; some builds ignore OPENCV_NUM_THREADS.
        if cv2.getNumThreads() != self.opencv_num_threads:
            cv2.setNumThreads(self.opencv_num_threads)
        data = super().__getitem__(idx)
        temporal_metas = self._temporal_metas(data['img_metas'])
        metas = temporal_metas[-1]
        if self.use_semantic_bev_cache:
            if self.use_semantic_temporal:
                probs, mats, stamps, valid = self._temporal_semantic_bev(temporal_metas)
                data['temporal_semantic_bev'] = DC(torch.from_numpy(probs))
                data['temporal_lidar2global'] = DC(torch.from_numpy(mats).float())
                data['temporal_timestamp'] = DC(torch.from_numpy(stamps))
                data['temporal_semantic_valid'] = DC(torch.from_numpy(valid))
                # Current frame stays under its original key so the existing
                # SBAM path is untouched and the two can be ablated separately.
                probabilities = probs[-1]
            else:
                probabilities = self.load_semantic_bev(metas['sample_idx'])
                
            data['semantic_bev_probabilities'] = DC(torch.from_numpy(probabilities))
            
        elif self.use_semantics:
            data['seg_maps'] = DC(self.load_segmaps(metas['filename'], metas, self.semantic_path))
        load_depth = self.load_depth_in_cache if self.use_semantic_bev_cache else self.use_depth
        if self.load_temporal_depth:
            temporal_depth = torch.stack([
                self.load_depth_from_filenames(frame['filename'], frame)
                for frame in temporal_metas])
            data['temporal_depth_maps'] = DC(temporal_depth)
            tokens = [frame['sample_idx'] for frame in temporal_metas]
            data['temporal_valid_mask'] = DC(torch.tensor(_last_occurrence_mask(tokens), dtype=torch.bool))
            data['depth_maps'] = DC(temporal_depth[-1])
        elif load_depth:
            data['depth_maps'] = DC(self.load_depth_from_filenames(metas['filename'], metas))
        return data

    @staticmethod
    def _map_path(root, filename, suffix):
        rel = filename.split('samples/')[-1]
        return os.path.join(root, 'samples', os.path.splitext(rel)[0] + suffix)

    def _align_map(self, array, img_metas, view, fill_value):
        original = np.asarray(img_metas['metric3d_lidar2img_raw'][view], dtype=np.float64)
        augmented = np.asarray(img_metas['lidar2img'][view], dtype=np.float64)
        transform = augmented @ np.linalg.inv(original)
        # Image augmentation must preserve camera-Z and homogeneous scale.
        if (not np.isfinite(transform).all()
                or not np.allclose(transform[2:], np.eye(4)[2:], atol=1e-5)
                or not np.allclose(transform[:2, 3], 0, atol=1e-5)):
            raise ValueError('Expected image-plane augmentation, not a 3D frame change')
        # PadMultiViewImage stores the pre-padding size in ori_shape.
        content_h, content_w = img_metas['ori_shape'][view][:2]
        pad_h, pad_w = img_metas['pad_shape'][view][:2]
        aligned = cv2.warpPerspective(
            np.asarray(array, dtype=np.float32), transform[:3, :3],
            (int(content_w), int(content_h)), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=float(fill_value))
        return np.pad(aligned, ((0, pad_h - content_h), (0, pad_w - content_w)),
                      mode='constant', constant_values=fill_value)

    def load_segmaps(self, filenames, img_metas, semantic_root):
        maps = []
        for view, filename in enumerate(filenames):
            path = self._map_path(semantic_root, filename, '_mask.bin')
            raw = np.fromfile(path, dtype=np.int8).reshape(self.depth_raw_shape).astype(np.int16)
            remapped = raw.copy()
            for source, target in (self.seg_id_remap or {}).items():
                remapped[raw == source] = target
            aligned = self._align_map(remapped, img_metas, view, -1)
            maps.append(torch.from_numpy(aligned.astype(np.int64)))
        return torch.stack(maps)

    def load_depth_from_filenames(self, filenames, img_metas=None):
        """Load depth, zero invalid values, and optionally align to the padded image."""
        maps = []
        for view, filename in enumerate(filenames):
            path = self._map_path(self.depth_path, filename, '.npy')
            if not os.path.isfile(path):
                # Metric3D exports may omit the DA3-style 'samples' directory.
                relative = os.path.splitext(filename.split('samples/')[-1])[0] + '.npy'
                path = os.path.join(self.depth_path, relative)
            raw = np.load(path, allow_pickle=False)
            if raw.shape != self.depth_raw_shape or not np.issubdtype(raw.dtype, np.floating):
                raise ValueError(f'Expected floating {self.depth_raw_shape} depth at {path}, '
                                 f'got {raw.shape} {raw.dtype}')
            valid = np.isfinite(raw) & (raw >= self.depth_range[0]) & (raw <= self.depth_range[1])
            depth = np.where(valid, raw, 0).astype(np.float32)
            if img_metas is not None:
                depth = self._align_map(depth, img_metas, view, 0)
            maps.append(torch.from_numpy(depth))
        return torch.stack(maps)


def configure_semantic_type(dataset, semantic_type, logcount_scale, use_cache, cache_root,
                            expected_frame=None):
    """Set the semantic input type and check it against the cache manifest.

    Mirrors the output_frame check: 'share+logcount' needs a cache built with
    layout 'share+counts+support'; a legacy share cache is rejected at startup.
    """
    from .layout_diffusion.seg_bev_aligner_one_hot_v5 import SEMANTIC_TYPES
    from .semantic_bev_cache import read_cache_manifest

    if semantic_type not in SEMANTIC_TYPES:
        raise ValueError(f'semantic_type must be one of {SEMANTIC_TYPES}, got {semantic_type!r}')
    if not np.isfinite(logcount_scale) or logcount_scale <= 0:
        raise ValueError('semantic_logcount_scale must be positive and finite')
    dataset.semantic_type = semantic_type
    dataset.semantic_logcount_scale = float(logcount_scale)
    dataset.semantic_bev_cache_layout = 'share'
    needs_manifest = use_cache and (semantic_type == 'share+logcount' or expected_frame is not None
                                    or getattr(dataset, 'semantic_bev_cache_shape', None) is None)
    if not needs_manifest:
        return
    contract = read_cache_manifest(cache_root)
    if contract is not None:
        dataset.semantic_bev_cache_layout = contract.get('layout', 'share')
    if expected_frame is not None:
        if contract is None:
            raise FileNotFoundError(os.path.join(cache_root, 'manifest.json'))
        frame = contract['projection'].get('output_frame', 'lidar')
        if frame != expected_frame:
            raise ValueError(f'Occupancy semantic cache output frame {frame} differs from '
                             f'expected {expected_frame}: {os.path.join(cache_root, "manifest.json")}')
    if semantic_type == 'share+logcount' and dataset.semantic_bev_cache_layout != 'share+counts+support':
        source = 'no manifest' if contract is None else f'layout {dataset.semantic_bev_cache_layout!r}'
        raise ValueError(f"semantic_type='share+logcount' needs a cache built with "
                         f"semantic_bev_cache_layout='share+counts+support', but {cache_root} has "
                         f'{source}; rebuild it with build_semantic_bev_cache.py')


@DATASETS.register_module()
class ViDARSemanticCacheDataset(CustomNuScenesDiffusionDataset_seg_depth_v2):
    """Keep the original queue/GT filter, then emit only the current token/cache.

    CustomNuScenesDataset.prepare_train_data still samples the three-frame
    queue and rejects it if any frame has no valid GT. The ViDAR worker
    uses the returned current-frame token to read the images with its own
    native test pipeline.
    """

    def __init__(self, *args, **kwargs):
        if not kwargs.get('use_semantic_bev_cache'):
            raise ValueError('ViDARSemanticCacheDataset requires semantic BEV cache')
        if kwargs.get('load_depth_in_cache') or kwargs.get('load_temporal_depth'):
            raise ValueError('Token-only ViDAR dataset cannot load depth maps')
        super().__init__(*args, **kwargs)

    def union2one(self, queue):
        # The queue's annotation-based acceptance has already been checked.
        return queue[-1]

    def __getitem__(self, idx):
        # Bypass the v2 image/segmentation wrapper while retaining the base
        # dataset's empty-GT retry and the current frame's cache validation.
        current = CustomNuScenesDataset.__getitem__(self, idx)
        token = current['img_metas'].data['sample_idx']
        probabilities = self.load_semantic_bev(token)
        return {
            'sample_token': DC(token, cpu_only=True),
            'semantic_bev_probabilities': DC(torch.from_numpy(probabilities)),
        }


@DATASETS.register_module()
class Occ3DDiffusionNuScenesDataset(NuSceneOcc):
    """Baseline Occ3D queues/GT/metrics plus current-frame Metric3D conditions.

    Reuse the Metric3D readers, but retain NuSceneOcc's data-info, queue and
    evaluation implementation. Detection-layout inheritance is not involved.
    """
    _current_metas = staticmethod(CustomNuScenesDiffusionDataset_seg_depth_v2._current_metas)
    _map_path = staticmethod(CustomNuScenesDiffusionDataset_seg_depth_v2._map_path)
    _retain_raw_projection = classmethod(CustomNuScenesDiffusionDataset_seg_depth_v2._retain_raw_projection.__func__)
    _align_map = CustomNuScenesDiffusionDataset_seg_depth_v2._align_map
    load_segmaps = CustomNuScenesDiffusionDataset_seg_depth_v2.load_segmaps
    load_depth_from_filenames = CustomNuScenesDiffusionDataset_seg_depth_v2.load_depth_from_filenames
    load_semantic_bev = CustomNuScenesDiffusionDataset_seg_depth_v2.load_semantic_bev

    def __init__(self, *args, pc_range=(-40, -40, -1, 40, 40, 5.4),
                 use_semantics=True, use_depth=True, semantic_path=None, depth_path=None,
                 seg_id_remap=None, depth_raw_shape=(900, 1600), depth_range=(2.5, 170),
                 opencv_num_threads=1, use_semantic_bev_cache=False,
                 semantic_bev_cache_root=None, semantic_bev_cache_shape=(17, 50, 50),
                 semantic_bev_cache_output_frame=None, semantic_type='share',
                 semantic_logcount_scale=1e5,
                 use_3d_bbox=False, num_classes=18, num_bboxes=0, load_annos=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pc_range = tuple(pc_range)
        self.use_semantics, self.use_depth = use_semantics, use_depth
        self.semantic_path, self.depth_path = semantic_path, depth_path
        self.seg_id_remap = seg_id_remap
        self.depth_raw_shape, self.depth_range = tuple(depth_raw_shape), tuple(depth_range)
        self.opencv_num_threads = int(opencv_num_threads)
        if not 0 < self.depth_range[0] < self.depth_range[1]:
            raise ValueError('depth_range must contain positive increasing meter bounds')
        self.use_semantic_bev_cache = use_semantic_bev_cache
        self.semantic_bev_cache_root = semantic_bev_cache_root
        self.semantic_bev_cache_shape = (tuple(semantic_bev_cache_shape)
                                         if semantic_bev_cache_shape is not None else None)
        if semantic_bev_cache_output_frame not in (None, 'lidar', 'ego'):
            raise ValueError('semantic_bev_cache_output_frame must be lidar, ego, or None')
        if not self.use_semantic_bev_cache and (self.use_semantics or self.use_depth):
            if not self._retain_raw_projection(self.pipeline):
                raise ValueError('Metric3D occupancy dataset requires CustomCollect3D to retain raw calibration')
        if self.use_semantic_bev_cache and (not use_semantics or semantic_bev_cache_root is None):
            raise ValueError('Cache mode requires use_semantics=True and semantic_bev_cache_root')
        configure_semantic_type(self, semantic_type, semantic_logcount_scale,
                                self.use_semantic_bev_cache, semantic_bev_cache_root,
                                semantic_bev_cache_output_frame)

    def get_data_info(self, index):
        info = super().get_data_info(index)
        if info is not None:
            if 'occ_gt_path' not in info:
                raise KeyError('Occ3D temporal annotations must provide occ_gt_path')
            if 'lidar2img' in info:
                info['metric3d_lidar2img_raw'] = [matrix.copy() for matrix in info['lidar2img']]
        return info

    def get_unconditioned_item(self, index):
        """Used by the offline builder without loading semantic/depth sources."""
        return super().__getitem__(index)

    def __getitem__(self, index):
        if cv2.getNumThreads() != self.opencv_num_threads:
            cv2.setNumThreads(self.opencv_num_threads)
        data = self.get_unconditioned_item(index)
        metas = self._current_metas(data['img_metas'])
        if self.use_semantic_bev_cache:
            probability = self.load_semantic_bev(metas['sample_idx'])
            data['semantic_bev_probabilities'] = DC(torch.from_numpy(probability))
        else:
            if self.use_semantics:
                data['seg_maps'] = DC(self.load_segmaps(metas['filename'], metas, self.semantic_path))
            if self.use_depth:
                data['depth_maps'] = DC(self.load_depth_from_filenames(metas['filename'], metas))
        return data


@DATASETS.register_module()
class Occ3DLayoutDiffusionNuScenesDataset(NuSceneOcc):
    """Occ3D temporal samples augmented with BEVDiffuser GT-layout tokens."""

    def __init__(self, *args, pc_range=(-40, -40, -1.0, 40, 40, 5.4),
                 use_3d_bbox=True, num_classes=12, num_bboxes=300,
                 use_layout=True, load_annos=False, layout_pc_range=None, **kwargs):
        self.pc_range = tuple(pc_range)
        self.layout_pc_range = tuple(
            self.pc_range if layout_pc_range is None else layout_pc_range)
        self.use_3d_bbox = bool(use_3d_bbox)
        self.num_layout_classes = int(num_classes)
        self.num_bboxes = int(num_bboxes)
        self.use_layout = bool(use_layout)
        self.load_annos = bool(load_annos)
        if len(self.pc_range) != 6:
            raise ValueError('pc_range must contain six values')
        if (len(self.layout_pc_range) != 6 or
                not np.isfinite(self.layout_pc_range).all() or
                any(self.layout_pc_range[axis + 3] <= self.layout_pc_range[axis]
                    for axis in range(3))):
            raise ValueError('layout_pc_range must have finite increasing XYZ bounds')
        if self.num_layout_classes < 3:
            raise ValueError('num_classes must include object, image and null classes')
        if self.num_bboxes < 1:
            raise ValueError('num_bboxes must be positive')
        super().__init__(*args, **kwargs)

    _unwrap_layout_value = staticmethod(_unwrap_singleton)

    def get_data_info(self, index):
        info = super().get_data_info(index)
        if info is not None and self.load_annos and 'ann_info' not in info:
            info['ann_info'] = self.get_ann_info(index)
        return info

    def normalize_layout_bbox(self, bbox):
        """Normalize native LiDAR boxes as in the original layout dataset."""
        bbox = bbox.clone()
        bounds = self.layout_pc_range
        bbox[..., 0] = ((bbox[..., 0] - bounds[0])
                        / (bounds[3] - bounds[0])) - 0.5
        bbox[..., 1] = ((bbox[..., 1] - bounds[1])
                        / (bounds[4] - bounds[1])) - 0.5
        bbox[..., 2] = ((bbox[..., 2] - bounds[2])
                        / (bounds[5] - bounds[2]))
        bbox[..., 3] = bbox[..., 3] / (bounds[3] - bounds[0])
        bbox[..., 4] = bbox[..., 4] / (bounds[4] - bounds[1])
        bbox[..., 5] = bbox[..., 5] / (bounds[5] - bounds[2])
        return bbox

    def get_layout_info(self, data):
        class_ids = self._unwrap_layout_value(data['gt_labels_3d'])
        gt_bboxes = self._unwrap_layout_value(data['gt_bboxes_3d'])

        layout_classes = torch.full(
            (self.num_bboxes,), self.num_layout_classes - 1, dtype=torch.long)
        layout_bboxes = torch.zeros((self.num_bboxes, 9), dtype=torch.float32)
        layout_valid = torch.zeros((self.num_bboxes,), dtype=torch.float32)
        layout_classes[0] = self.num_layout_classes - 2
        layout_bboxes[0] = torch.tensor(
            [0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.float32)
        layout_valid[0] = 1.0

        num_valid = min(len(class_ids), len(gt_bboxes), self.num_bboxes - 1)
        if num_valid:
            selected_classes = class_ids[:num_valid].long()
            if (selected_classes.min() < 0
                    or selected_classes.max() >= self.num_layout_classes - 2):
                raise ValueError('GT layout class IDs exceed the configured object taxonomy')
            boxes = self.normalize_layout_bbox(
                gt_bboxes.tensor[:num_valid, :9].to(dtype=torch.float32))
            layout_classes[1:1 + num_valid] = selected_classes
            layout_bboxes[1:1 + num_valid] = boxes
            layout_valid[1:1 + num_valid] = 1.0

        return dict(
            layout_obj_classes=DC(layout_classes),
            layout_obj_bboxes=DC(layout_bboxes),
            layout_obj_is_valid=DC(layout_valid),
        )

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.use_layout:
            data.update(self.get_layout_info(data))
        return data
class CustomNuScenesDiffusionDataset_layout_seg_v1(CustomNuScenesDataset): 
    def __init__(self, pc_range, use_3d_bbox=True, num_classes=18, num_bboxes=300, use_layout=True, use_semantics=True, use_depth=False,
                 semantic_path=None, depth_path=None, seg_class=None, total_class=None, seg_id_remap=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox=use_3d_bbox
        self.num_classes=num_classes
        self.num_bboxes=num_bboxes
        self.total_class = total_class
        self.seg_class = seg_class
        self.seg_id_remap = seg_id_remap
        self.object_names = list(self.total_class) + ['__image__', '__null__']
        self.object_clips = self.embed_object_names()
        self.use_layout = use_layout
        self.use_semantics = use_semantics
        self.use_depth = use_depth
        self.semantic_path = semantic_path
        self.depth_path = depth_path

        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # Segmentation mask
        if self.use_semantics and self.semantic_path is not None:
            img_metas_raw = data['img_metas']
            img_metas_list = img_metas_raw.data if isinstance(img_metas_raw, DC) else img_metas_raw
            if isinstance(img_metas_list, dict):
                current_metas = img_metas_list[max(img_metas_list.keys())]
            else:
                current_metas = img_metas_list[-1]
            if isinstance(current_metas, DC):
                current_metas = current_metas.data
            seg_maps = self.load_segmaps(
                current_metas['filename'],
                current_metas,
                self.semantic_path,
            )
            data['seg_maps'] = DC(seg_maps)
            
        if self.use_layout:
            layout = self.get_layout_info(data, seg_maps.unique().tolist())
            for info in layout.keys():
                data[info] = layout[info]
        
        if self.use_depth and self.depth_path is not None:
            img_metas_raw = data['img_metas']
            img_metas_list = img_metas_raw.data if isinstance(img_metas_raw, DC) else img_metas_raw
            if isinstance(img_metas_list, dict):
                depth_metas = img_metas_list[max(img_metas_list.keys())]
            else:
                depth_metas = img_metas_list[-1]
            if isinstance(depth_metas, DC):
                depth_metas = depth_metas.data
            depth_maps = self.load_depth_from_filenames(depth_metas['filename'])
            data['depth_maps'] = DC(depth_maps)
        
        return data
    
    def embed_object_names(self):
        pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2-1'
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder.requires_grad_(False)
        object_tokens = tokenizer(self.object_names, 
                                  max_length=tokenizer.model_max_length,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt").input_ids
        object_clip_embed = text_encoder(object_tokens)[1]
        return object_clip_embed # (N, 1024)
    
    def get_layout_info(self, data, seg_class_valid):
        # data['gt_labels_3d'] should be a DC of a tensor with size [N]
        # data['gt_bboxes_3d'] should be a DC of LiDARInstance3DBoxes of a tensor with size [N, 9]: (x, y, z, x_size, y_size, z_size, yaw, vx, vy) 
        class_ids = torch.tensor([])
        gt_bboxes = torch.tensor([])
        if 'gt_labels_3d' in data:
            while not isinstance(data['gt_labels_3d'], DC):
                data['gt_labels_3d'] = data['gt_labels_3d'][0]
            class_ids = data['gt_labels_3d'].data
            class_ids = (class_ids + len(self.CLASSES))%len(self.CLASSES)
        if 'gt_bboxes_3d' in data:
            while not isinstance(data['gt_bboxes_3d'], DC):
                data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0]
            gt_bboxes = data['gt_bboxes_3d'].data
            
        layout_obj_classes = torch.LongTensor(self.num_bboxes).fill_(self.num_classes-1)
        layout_is_valid = torch.zeros([self.num_bboxes])
        
        layout_obj_classes[0] = self.num_classes-2
        layout_is_valid[0] = 1.0
        # default_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
        
        num_valid = min(len(class_ids), self.num_bboxes-1)
        layout_obj_classes[1: 1+num_valid] = class_ids
        
        n = 1+num_valid
        for i in self.seg_class.keys():
            if i in seg_class_valid:
                layout_obj_classes[n] = self.seg_class[i][-1]
                n += 1
        
        layout_is_valid[1: 1+num_valid] = 1.0
        
        layout_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
       
        
        if self.use_3d_bbox:
            layout_obj_bboxes = self.get_3d_layout_bboxes(gt_bboxes)
        else:
            layout_obj_bboxes = self.get_2d_layout_bboxes(gt_bboxes)
            
        layout = {
            'layout_obj_classes': DC(layout_obj_classes),
            'layout_obj_bboxes': DC(layout_obj_bboxes),
            'layout_obj_is_valid': DC(layout_is_valid),
            'layout_obj_names': DC(layout_obj_clip),
            # 'default_obj_names': DC(default_obj_clip),
        }

        return layout
    
    def normalize_bbox(self, bbox):
        # normalize bbox into [0,1], ego at [0.5, 0.5] 
        x, y = torch.tensor_split(bbox[..., :2], 2, dim=-1)
        x = (x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y = (y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        if bbox.shape[-1] > 2:
            z, x_size, y_size, z_size, yaw, vx, vy = torch.tensor_split(bbox[..., 2:], 7, dim=-1)
            z = (z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            x_size = x_size / (self.pc_range[3] - self.pc_range[0])
            y_size = y_size / (self.pc_range[4] - self.pc_range[1])
            z_size = z_size / (self.pc_range[5] - self.pc_range[2])
            return torch.cat((x, y, z, x_size, y_size, z_size, yaw, vx, vy), dim=-1)
        return torch.cat((x, y), dim=-1)
            
    
    def get_3d_layout_bboxes(self, gt_bboxes): 
        # 3d bbox: (xc, yc, zc, x_size, y_size, z_size, yaw, vx, vy) 
        # ego coordinate, origin at ego position, x towards right, y towards front 
        layout_bboxes = torch.zeros([self.num_bboxes, 9])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            # (x, y) -> (x-0.5, x-0.5)
            gt_bboxes = self.normalize_bbox(gt_bboxes.tensor)
            gt_bboxes[..., :2] = gt_bboxes[..., :2] - 0.5
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
    
    def get_2d_layout_bboxes(self, gt_bboxes):
        # 2d bbox: (x0, y0, x1, y1), 
        # image coordinate, orgin at upper left, x towards right, y towards down
        layout_bboxes = torch.zeros([self.num_bboxes, 4])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 1, 1])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            gt_bboxes = self.normalize_bbox(gt_bboxes.corners[..., :2]) # N x 8 x 2
            # (x, y) -> (x, 1-y)
            gt_bboxes[..., 1] = 1 - gt_bboxes[..., 1]
            gt_bboxes_min = gt_bboxes.min(dim=1).values # N x 2
            gt_bboxes_max = gt_bboxes.max(dim=1).values # N x 2
            gt_bboxes = torch.cat((gt_bboxes_min, gt_bboxes_max), dim=-1)
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
            
    def load_depth_from_filenames(self, filenames):
        """Load multi-view depth maps (.npy, float16) from nuscenes_depth_da3.

        Args:
            filenames: list[str] — per-camera image paths, e.g.
                       '.../samples/CAM_FRONT/xxx.jpg'  (length = V)

        Returns:
            FloatTensor of shape [V, H, W]  (H=448, W=798)
        """
        view_depths = []
        for fname in filenames:
            # '.../samples/CAM_FRONT/xxx.jpg' → 'CAM_FRONT/xxx'
            rel = fname.split('samples/')[-1]
            base = os.path.splitext(rel)[0]
            npy_path = os.path.join(self.depth_path, 'samples', base + '.npy')

            try:
                depth = np.load(npy_path, mmap_mode='r')
                depth_tensor = torch.from_numpy(depth.copy()).float()
            except FileNotFoundError:
                print(f'Warning: depth map not found at {npy_path}, filling with zeros')
                depth_tensor = torch.zeros(448, 798, dtype=torch.float32)

            view_depths.append(depth_tensor)

        return torch.stack(view_depths, dim=0)  # [V, H, W]

    def load_segmaps(self, filenames, img_metas, semantic_root):
        """Load multi-view segmentation maps (.bin) and apply the same spatial
        transforms as the image pipeline (nearest-neighbor resize then pad).
        Returns:
            LongTensor of shape [V, H_pad, W_pad]
        """
        from PIL import Image

        ignore_label = -1
        H, W = 900, 1600

        seg_maps = []
        # BUGFIX: 'img_shape' holds the PADDED size (PadMultiViewImage overwrites
        img_shapes = img_metas['img_shape']
        # img_shapes = get_content_shapes(img_metas)  # pre-pad content (H, W, C) per cam
        pad_shapes = img_metas['pad_shape']  # list[(H, W, C)] per cam

        for i, fname in enumerate(filenames):
            # Derive seg map path: '.../samples/CAM_XXX/name.jpg' → 'samples/CAM_XXX/name.bin'
            rel = fname.split('samples/')[-1]        # 'CAM_FRONT/xxx.jpg'
            base = os.path.splitext(rel)[0]          # 'CAM_FRONT/xxx'
            seg_path = os.path.join(semantic_root, 'samples', base + '_mask.bin')

            h_scale, w_scale = img_shapes[i][:2]
            h_pad, w_pad = pad_shapes[i][:2]

            # Load binary seg map (int8 on disk, promote to int16 to hold -1 cleanly)
            try:
                x = np.fromfile(seg_path, dtype=np.int8)
                if x.size != H * W:
                    mask = np.full((H, W), ignore_label, dtype=np.int16)
                else:
                    mask = x.reshape(H, W).astype(np.int16)
            except FileNotFoundError:
                print(f'Warning: seg map not found at {seg_path}, filling with ignore_label')
                mask = np.full((H, W), ignore_label, dtype=np.int16)

            # Optional raw seg-id remap (e.g. SAM3 -> model taxonomy). 
            seg_id_remap = getattr(self, 'seg_id_remap', None)
            if seg_id_remap:
                remapped = mask.copy()
                for src, dst in seg_id_remap.items():
                    remapped[mask == src] = dst
                mask = remapped

            # Resize to img_shape using NEAREST to preserve class IDs.
            # PIL mode 'I' (int32) is used because mode 'I;16' treats values as
            # unsigned and would corrupt negative ignore_label (-1 → 65535).
            seg_pil = Image.fromarray(mask.astype(np.int32))   # mode 'I'
            seg_pil = seg_pil.resize((w_scale, h_scale), Image.NEAREST)
            seg_arr = np.array(seg_pil, dtype=np.int16)

            # Pad to pad_shape, fill with ignore_label (mirrors PadMultiViewImage)
            pad_h = h_pad - h_scale
            pad_w = w_pad - w_scale
            seg_arr = np.pad(seg_arr, ((0, pad_h), (0, pad_w)),
                             mode='constant', constant_values=ignore_label)

            seg_maps.append(torch.tensor(seg_arr, dtype=torch.long))

        return torch.stack(seg_maps, dim=0)  # [V, H_pad, W_pad]
