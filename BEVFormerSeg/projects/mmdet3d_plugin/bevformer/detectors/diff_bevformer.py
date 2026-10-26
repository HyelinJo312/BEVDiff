import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS

from .diff_bevformer_onlyseg import DiffBEVFormerMapSeg


@DETECTORS.register_module()
class DiffBEVFormer(DiffBEVFormerMapSeg):
    """Original BEVDiffuser teacher (GT layout condition) for the BEV seg student.

    Port of BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer.py.
    Task loss, MGD and loss mixing are inherited from DiffBEVFormerMapSeg so the
    baseline and the semantic-prior teacher share the same Stage 2 code; only the
    teacher call differs (tools/bevdiffuser.py: forward(x, condition, grad_fn)).
    """

    def run_teacher(self, bev_diffuser, noisy_bev, img_metas, grad_fn, **kwargs):
        cond = {}
        if 'layout_obj_classes' in kwargs:
            cond['obj_class'] = torch.stack(kwargs['layout_obj_classes'])
        if 'layout_obj_bboxes' in kwargs:
            cond['obj_bbox'] = torch.stack(kwargs['layout_obj_bboxes'])
        if 'layout_obj_is_valid' in kwargs:
            cond['is_valid_obj'] = torch.stack(kwargs['layout_obj_is_valid'])
        if 'layout_obj_names' in kwargs:
            cond['obj_name'] = torch.stack(kwargs['layout_obj_names'])
        if 'default_obj_names' in kwargs:
            cond['default_obj_names'] = torch.stack(kwargs['default_obj_names'])
        return bev_diffuser(noisy_bev, cond, grad_fn=grad_fn)


class ImageRenormTarget(torch.nn.Module):
    """Feed model_target images in the input format its checkpoint was trained with.

    Normalization: the student pipeline normalizes with `src` (e.g. caffe BGR,
    std 1) while the target checkpoint may expect `dst` (e.g. torchvision RGB).
    Images are mapped back to raw pixels and re-normalized; zero padding stays
    zero when the two means are the same colors in different channel order.

    Scale: with `rescale` != 1 the unpadded content is resized (as
    RandomScaleImageMultiViewImage would), re-padded to `size_divisor`, and
    lidar2img / img_shape / pad_shape / ori_shape are updated to match.
    """

    def __init__(self, model, src, dst, rescale=1.0, size_divisor=32):
        super().__init__()
        self.model = model
        self.flip = bool(src['to_rgb']) != bool(dst['to_rgb'])
        self.rescale = float(rescale)
        self.size_divisor = size_divisor
        for name, cfg in (('src', src), ('dst', dst)):
            self.register_buffer(f'{name}_mean', torch.tensor(cfg['mean'], dtype=torch.float32).view(3, 1, 1), persistent=False)
            self.register_buffer(f'{name}_std', torch.tensor(cfg['std'], dtype=torch.float32).view(3, 1, 1), persistent=False)

    def renorm(self, img):
        raw = img * self.src_std + self.src_mean
        if self.flip:
            raw = raw.flip(-3)
        return (raw - self.dst_mean) / self.dst_std

    def _rescale_meta(self, meta):
        meta = dict(meta)
        scale = np.eye(4)
        scale[0, 0] = scale[1, 1] = self.rescale
        meta['lidar2img'] = [scale @ l2i for l2i in meta['lidar2img']]
        content = [(int(h * self.rescale), int(w * self.rescale), c) for h, w, c in meta['ori_shape']]
        padded = [(int(np.ceil(h / self.size_divisor)) * self.size_divisor,
                   int(np.ceil(w / self.size_divisor)) * self.size_divisor, c) for h, w, c in content]
        meta.update(ori_shape=content, img_shape=padded, pad_shape=padded)
        return meta

    def _rescale(self, img, img_metas):
        # img: (B, Q, N, 3, H, W); img_metas: list over B of {queue index: meta}.
        h, w, _ = img_metas[0][0]['ori_shape'][0]
        for queue in img_metas:
            for meta in queue.values():
                if any(tuple(s[:2]) != (h, w) for s in meta['ori_shape']):
                    raise ValueError('ImageRenormTarget expects one image content size per batch')
        new_metas = [{q: self._rescale_meta(meta) for q, meta in queue.items()} for queue in img_metas]
        nh, nw, _ = new_metas[0][0]['ori_shape'][0]
        ph, pw, _ = new_metas[0][0]['img_shape'][0]
        lead = img.shape[:-3]
        content = img[..., :h, :w].reshape(-1, 3, h, w)
        content = F.interpolate(content, size=(nh, nw), mode='bilinear', align_corners=False)
        content = F.pad(content, (0, pw - nw, 0, ph - nh))
        return content.reshape(*lead, 3, ph, pw), new_metas

    def forward(self, *args, img=None, img_metas=None, **kwargs):
        if img is not None and not isinstance(img, list):
            img = self.renorm(img)
            if self.rescale != 1.0:
                img, img_metas = self._rescale(img, img_metas)
        elif isinstance(img, list):
            if self.rescale != 1.0:
                raise NotImplementedError('rescale is only supported for the training queue input')
            img = [self.renorm(each) for each in img]
        return self.model(*args, img=img, img_metas=img_metas, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('model'), name)
