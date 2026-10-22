# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import numpy as np
import os
import json
import hashlib
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
# from projects.mmdet3d_plugin.datasets.nuscenes_dataset_v2 import CustomNuScenesDatasetV2


def get_content_shapes(img_metas):
    """Per-camera pre-pad (post-scale) image content shape, as [(H, W, C), ...].

    `img_shape` cannot be used for this. PadMultiViewImage._pad_img overwrites it
    with the PADDED size and stashes the pre-pad size in `ori_shape`:

        scale 0.8 -> ori_shape (720, 1280) | img_shape == pad_shape (736, 1280)
    """
    return img_metas.get('ori_shape', img_metas['img_shape'])


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
            semantic_root: str — root directory for seg maps

        Returns:
            LongTensor of shape [V, H_pad, W_pad]
        """
        from PIL import Image

        ignore_label = -1
        H, W = 900, 1600

        seg_maps = []
        # BUGFIX: 'img_shape' holds the PADDED size (PadMultiViewImage overwrites
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
                 semantic_path=None, depth_path=None, seg_class=None, total_class=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox=use_3d_bbox
        self.num_classes=num_classes
        self.num_bboxes=num_bboxes
        self.total_class = total_class
        self.seg_class = seg_class
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
            semantic_root: str — root directory for seg maps

        Returns:
            LongTensor of shape [V, H_pad, W_pad]
        """
        from PIL import Image

        ignore_label = -1
        H, W = 900, 1600

        seg_maps = []
        # BUGFIX: 'img_shape' holds the PADDED size (PadMultiViewImage overwrites
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
    def __init__(self, pc_range, use_3d_bbox=True, num_classes=12,
                 num_bboxes=300, use_semantics=True, use_depth=False,
                 semantic_path=None, depth_path=None, seg_id_remap=None,
                 load_frame_interval=None, *args, **kwargs):
        super().__init__(
            *args, load_frame_interval=load_frame_interval, **kwargs)
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

# Semantic input types, matching BEVFormer's seg_bev_aligner_one_hot_v5.SEMANTIC_TYPES.
# 'share': per-class pixel share (legacy prior).
# 'share+logcount': share | log1p(raw per-class pixel count) / log1p(logcount_scale).
SEMANTIC_TYPES = ('share', 'share+logcount')
# Cache layouts written by BEVFormer's build_semantic_bev_cache.py.
# 'share': [C, H, W]; 'share+counts+support': [2C+1, H, W] = share | counts | support.
CACHE_LAYOUTS = ('share', 'share+counts+support')


def semantic_input(share, counts=None, semantic_type='share', logcount_scale=1e5):
    """Model input from share [C, H, W] and raw counts; same formula as BEVFormer."""
    if semantic_type == 'share':
        return share
    if semantic_type != 'share+logcount':
        raise ValueError(f'semantic_type must be one of {SEMANTIC_TYPES}, got {semantic_type!r}')
    if counts is None or counts.shape != share.shape:
        raise ValueError("semantic_type='share+logcount' requires raw counts shaped like the share")
    logcount = np.log1p(np.clip(counts, 0, None)) / np.log1p(logcount_scale)
    return np.concatenate([share, logcount], axis=-3).astype(np.float32)


def stored_cache_shape(share_shape, layout):
    """Stored [channels, H, W] of a cache sample for a share shape [C, H, W]."""
    classes, height, width = share_shape
    if layout == 'share':
        return (classes, height, width)
    if layout == 'share+counts+support':
        return (2 * classes + 1, height, width)
    raise ValueError(f'cache layout must be one of {CACHE_LAYOUTS}, got {layout!r}')


@DATASETS.register_module()
class CustomNuScenesDiffusionDataset_seg_depth_v2(CustomNuScenesDiffusionDataset_seg_depth):
    """Segmentation dataset using precomputed SAM3/Metric3D BEV probabilities.

    The cache contains input conditioning only. The map segmentation labels and
    BEV target are still provided by the original segmentation dataset/model.

    semantic_bev_cache_shape is the share shape [C, H, W]. With
    semantic_type='share+logcount' the cache must have been built with
    layout 'share+counts+support' and the returned input is [2C, H, W].
    """

    def __init__(self, semantic_bev_cache_root, semantic_bev_cache_shape,
                 semantic_bev_cache_projection, seg_id_remap=None,
                 use_semantic_bev_cache=True, semantic_type='share',
                 semantic_logcount_scale=1e5, **kwargs):
        if semantic_type not in SEMANTIC_TYPES:
            raise ValueError(f'semantic_type must be one of {SEMANTIC_TYPES}, got {semantic_type!r}')
        if not np.isfinite(semantic_logcount_scale) or semantic_logcount_scale <= 0:
            raise ValueError('semantic_logcount_scale must be positive and finite')
        if not use_semantic_bev_cache:
            raise ValueError('This dataset variant requires a semantic BEV cache')
        # The cache already includes semantic projection and metric depth.
        kwargs.pop('use_semantics', None)
        kwargs.pop('use_depth', None)
        super().__init__(use_semantics=False, use_depth=False,
                         seg_id_remap=seg_id_remap, **kwargs)
        self.use_semantic_bev_cache = True
        self.semantic_bev_cache_root = Path(semantic_bev_cache_root).resolve()
        self.semantic_bev_cache_shape = tuple(semantic_bev_cache_shape)
        manifest_path = self.semantic_bev_cache_root / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Missing semantic BEV cache manifest: {manifest_path}')
        self.semantic_bev_manifest = json.loads(manifest_path.read_text())
        contract = self.semantic_bev_manifest.get('contract', {})
        projection = contract.get('projection', {})
        # The source cache was built in BEVFormer, so its code hash and image
        # augmentation metadata do not describe this segmentation repository.
        # Physical BEV grid and semantic taxonomy must nevertheless match.
        expected = dict(semantic_bev_cache_projection)
        for key, value in expected.items():
            if projection.get(key) != value:
                raise ValueError(f'Semantic BEV cache {key} mismatch: '
                                 f'{projection.get(key)!r} != {value!r}')
        if contract.get('remap') != {str(k): v for k, v in (seg_id_remap or {}).items()}:
            raise ValueError('Semantic BEV cache SAM3 label remap mismatch')
        shape = (projection.get('num_classes', -1) + 1,
                 projection.get('bev_h'), projection.get('bev_w'))
        if shape != self.semantic_bev_cache_shape:
            raise ValueError(f'Semantic BEV cache shape mismatch: {shape}')
        # Layout check, like the output_frame check: the input type must be
        # derivable from what the cache stores.
        self.semantic_bev_cache_layout = contract.get('layout', 'share')
        if (semantic_type == 'share+logcount'
                and self.semantic_bev_cache_layout != 'share+counts+support'):
            raise ValueError(
                f"semantic_type='share+logcount' needs a cache built with layout "
                f"'share+counts+support', but {manifest_path} has layout "
                f'{self.semantic_bev_cache_layout!r}; rebuild it with '
                'BEVFormer/projects/bevdiffuser/build_semantic_bev_cache.py')
        self.semantic_bev_stored_shape = stored_cache_shape(
            self.semantic_bev_cache_shape, self.semantic_bev_cache_layout)
        self.semantic_type = semantic_type
        self.semantic_logcount_scale = float(semantic_logcount_scale)

    def load_semantic_bev(self, token):
        """Verified cache read -> [C, H, W] share or [2C, H, W] share+logcount."""
        entry = self.semantic_bev_manifest.get('entries', {}).get(token)
        if entry is None:
            raise KeyError(f'Sample {token} is missing from semantic BEV cache')
        path = self.semantic_bev_cache_root / f'{token}.npy'
        value = np.load(path, allow_pickle=False)
        if (value.shape != self.semantic_bev_stored_shape or value.dtype != np.float32
                or not np.isfinite(value).all() or (value < 0).any()):
            raise ValueError(f'Invalid semantic BEV cache sample at {path}: expected float32 '
                             f'{self.semantic_bev_stored_shape} ({self.semantic_bev_cache_layout}), '
                             f'got {value.dtype} {value.shape}')
        if hashlib.sha256(value.tobytes()).hexdigest() != entry.get('sha256'):
            raise ValueError(f'Corrupt semantic BEV probabilities at {path}')
        classes = self.semantic_bev_cache_shape[0]
        share = value[:classes]
        counts = value[classes:2 * classes] if self.semantic_bev_cache_layout != 'share' else None
        return np.ascontiguousarray(
            semantic_input(share, counts, self.semantic_type, self.semantic_logcount_scale),
            dtype=np.float32)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        metadata = data['img_metas']
        while isinstance(metadata, DC):
            metadata = metadata.data
        if isinstance(metadata, dict) and 'sample_idx' not in metadata:
            metadata = metadata[max(metadata)]
        if isinstance(metadata, (list, tuple)):
            metadata = metadata[-1]
        while isinstance(metadata, DC):
            metadata = metadata.data
        data['semantic_bev_probabilities'] = DC(
            torch.from_numpy(self.load_semantic_bev(metadata['sample_idx'])))
        return data
