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


def get_content_shapes(img_metas):
    """Per-camera (H, W, C) of the real image content inside the padded frame.

    `PadMultiViewImage._pad_img` overwrites BOTH 'img_shape' and 'pad_shape'
    with the *padded* size, so 'img_shape' no longer records the pre-pad content
    size -- that survives only in 'ori_shape'. Resizing a seg map to 'img_shape'
    therefore stretches it across the pad region: with
    RandomScaleImageMultiViewImage(0.5) + PadMultiViewImage(32) the content is
    450x800 inside a 480x800 frame, so 900x1600 -> 480x800 is a 480/450
    vertical stretch that drags labels ~0.0625*row upward relative to the image
    (measured: -7/-13/-19/-24 px at rows 90-450). `fm_feature.py` already uses
    the ori_shape -> img_shape convention for DA3 depth.

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



           
@DATASETS.register_module()
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
