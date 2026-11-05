# Stage 2 of the semantic-only BEV diffusion teacher on BEVDet4D.
#
# Port of projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer_onlyseg.py
# (DiffBEVFormerSegV3, the flagship DiffBEVFormer_tiny_onlyseg_metric3d_t100_alpha100_lambda_0.55)
# with the same structure and flags:
#   forward(only_bev=True)          -> BEV feature only (used on the frozen model_target)
#   forward_pts_train(given_bev=..) -> task head on a given BEV
#   train_step(model_target, bev_diffuser) -> mixed loss (1-w)*task + w*bev, w = 0.5
# Difference to BEVFormer: BEVDet builds the BEV in the detector (extract_img_feat ->
# img_bev_encoder_neck output, [B, 256, 128, 128]) and CenterHead consumes it directly, so the
# only_bev / given_bev flags live at the detector level instead of inside the head. The BEV is
# already [B, C, H, W]; no (B, H*W, C) reshapes are needed.
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mmdet.models import DETECTORS

from mmdet3d.models.detectors.bevdet import BEVDet4D


def sample_spatialchannel_keep_mask(x, cell_ratio, channel_ratio):
    """Keep channels independently within selected BEV cells."""
    if x.ndim != 4:
        raise ValueError('MGD input must have shape [B, C, H, W]')
    for name, ratio in (('cell_ratio', cell_ratio), ('channel_ratio', channel_ratio)):
        if not 0 <= ratio <= 1:
            raise ValueError(f'{name} must be between 0 and 1')
    B, C, H, W = x.shape
    selected_cells = torch.rand((B, 1, H, W), device=x.device) < cell_ratio
    selected_channels = torch.rand((B, C, H, W), device=x.device) < channel_ratio
    return (~(selected_cells & selected_channels)).to(dtype=x.dtype)


def sample_mgd_keep_mask(x, mgd_lambda):
    """MGD mask of DiffBEVFormerSegV3: drop a fraction mgd_lambda of BEV cells (all channels)."""
    B, _, H, W = x.shape
    mat = torch.rand((B, 1, H, W), device=x.device)
    return torch.where(mat > 1 - mgd_lambda, torch.zeros_like(mat), torch.ones_like(mat))


class BEVDistillProjector(nn.Module):
    def __init__(self, in_ch=256, hidden_ch=256, out_ch=256, groups=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.proj(x)


@DETECTORS.register_module()
class DiffBEVDet4D(BEVDet4D):
    """BEVDet4D student distilled from the semantic-only BEV diffusion teacher.

    Without ``bev_diffuser`` (e.g. evaluation, or train_step with no teacher) every path is
    numerically the BEVDet4D one. Extra modules (mgd_generation / bev_distill_proj) exist
    only in the student and are not used at test time.

    Args (as in DiffBEVFormerSegV3):
        use_mgd (bool): masked generative distillation on the BEV feature.
        mgd_alpha (float): weight of the MGD loss ('loss_bev').
        mgd_lambda (float): fraction of BEV cells masked before the generator.
        use_spatialchannel (bool): cell x channel mask instead of the cell mask.
        use_proj (bool): 1x1 projector + MSE (x100) instead of MGD.
        loss_mix_weight (float): w in (1 - w) * task + w * bev (hard-coded 0.5 in the flagship).
    """

    def __init__(self, *args,
                 use_proj=False,
                 use_mgd=False,
                 mgd_alpha=0.00002,
                 mgd_lambda=0.65,
                 use_spatialchannel=False,
                 spatialchannel_cell_ratio=0.8,
                 spatialchannel_channel_ratio=0.6875,
                 loss_mix_weight=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_proj = use_proj
        self.use_mgd = use_mgd
        self.mgd_alpha = mgd_alpha
        self.mgd_lambda = mgd_lambda
        self.use_spatialchannel = use_spatialchannel
        self.spatialchannel_cell_ratio = spatialchannel_cell_ratio
        self.spatialchannel_channel_ratio = spatialchannel_channel_ratio
        self.loss_mix_weight = loss_mix_weight
        for name, ratio in (('spatialchannel_cell_ratio', spatialchannel_cell_ratio),
                            ('spatialchannel_channel_ratio', spatialchannel_channel_ratio),
                            ('mgd_lambda', mgd_lambda),
                            ('loss_mix_weight', loss_mix_weight)):
            if not 0 <= ratio <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
        embed_dim = self.pts_bbox_head.in_channels  # BEV channels fed to CenterHead (256)
        if not isinstance(embed_dim, int):
            raise ValueError(f'expected an int pts_bbox_head.in_channels, got {embed_dim!r}')

        if use_proj:
            self.bev_distill_proj = BEVDistillProjector(embed_dim, embed_dim, embed_dim)

        if use_mgd:
            self.mgd_align = None
            self.mgd_generation = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )

    # ------------------------------------------------------------------ entry points
    def forward(self, return_loss=True, only_bev=False, **kwargs):
        """forward_train / forward_test as in BEVDet, plus ``only_bev`` (BEV feature only).

        ``only_bev`` follows DiffBEVFormer: it is used as ``return_loss=False, only_bev=True``
        on the frozen model_target with training-format inputs.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            if only_bev:
                return self.obtain_bev(**kwargs)
            else:
                return self.forward_test(**kwargs)

    def obtain_bev(self, img_inputs=None, img_metas=None, points=None, **kwargs):
        """BEV feature [B, C, H, W] = extract_img_feat(...)[0][0] (input of CenterHead).

        Accepts training-format inputs (list of tensors, list of metas) and test-format ones
        (one more list level for test-time augmentation; only the first is used).
        """
        if isinstance(img_inputs[0], (list, tuple)):
            img_inputs, img_metas = img_inputs[0], img_metas[0]
        img_feats, _, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        return img_feats[0]

    def train_step(self, data, optimizer, model_target=None, bev_diffuser=None, progress=None):
        """One training iteration (called by DiffEpochBasedRunner).

        Returns dict(loss, log_vars, num_samples) like BaseDetector.train_step; the loss is
        (1 - w) * task + w * bev with w = loss_mix_weight. As in DiffBEVFormerSegV3 the task
        term keeps the (1 - w) factor even without a teacher (bev = 0).
        """
        losses = self(**data, model_target=model_target, bev_diffuser=bev_diffuser)

        weight = self.loss_mix_weight
        loss, log_vars = self._parse_losses_mix(losses, weight=weight)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      given_bev=None,
                      model_target=None,
                      bev_diffuser=None,
                      **kwargs):
        """BEVDet.forward_train plus the frozen target BEV for the teacher.

        Args (in addition to BEVDet.forward_train):
            given_bev (Tensor, optional): [B, C, H, W] BEV to feed the head instead of the
                student's own BEV (no image branch is run then).
            model_target (nn.Module, optional): frozen BEVDet4D giving the clean BEV x0.
            bev_diffuser (nn.Module, optional): frozen teacher (BEVDiffuser).
        """
        bev_target = None
        if bev_diffuser:
            assert model_target is not None
            with torch.no_grad():
                bev_target = model_target(return_loss=False, only_bev=True,
                                          img_inputs=img_inputs, img_metas=img_metas).detach()

        img_feats = None
        if given_bev is None:
            img_feats, _, _ = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, given_bev, model_target,
                                            bev_diffuser, bev_target, **kwargs)

        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          given_bev=None,
                          model_target=None,
                          bev_diffuser=None,
                          bev_target=None,
                          **kwargs):
        """Task loss on the (given) BEV and, with a teacher, the distillation loss 'loss_bev'.

        Args:
            pts_feats (list[Tensor] | None): [img_feats[0]] from extract_feat; None when
                given_bev is used.
            bev_target (Tensor, optional): clean BEV x0 of the frozen model_target.
        """
        # original BEV feature (the student's only_bev output) unless a BEV is given
        bev = pts_feats[0] if given_bev is None else given_bev

        # task loss
        outs = self.pts_bbox_head([bev])
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        if bev_diffuser:
            assert bev_target is not None

            def get_classifier_gradient(x):
                x_in = x.detach().requires_grad_(True)
                outs = model_target.pts_bbox_head([x_in])
                losses = model_target.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
                loss, _ = self._parse_losses(losses)
                gradient = torch.autograd.grad(loss, x_in)[0]
                return gradient

            # Cached semantic BEV probabilities, already warped into this BEV frame (bda incl.)
            # by LoadSemanticBEVPrior; no seg_maps / depth_maps in the BEVDet path.
            if 'semantic_bev_probabilities' not in kwargs:
                raise KeyError('Stage 2 requires semantic_bev_probabilities (LoadSemanticBEVPrior)')
            semantic_prob = torch.stack(kwargs['semantic_bev_probabilities'], dim=0)
            segmaps = None
            depth_maps = None

            bev_target = bev_target.detach()
            # The teacher is frozen; build no graph unless classifier guidance needs one.
            grad_ctx = torch.enable_grad() if bev_diffuser.use_classifier_guidence else torch.no_grad()
            with grad_ctx:
                teacher_bev = bev_diffuser(bev_target, img_metas, segmaps, depth_maps,
                                           grad_fn=get_classifier_gradient, semantic_prob=semantic_prob)

            B, C, H, W = bev.shape
            student_bev = bev  # (B, C, H, W)
            if teacher_bev.shape != student_bev.shape:
                raise ValueError(f'teacher BEV {tuple(teacher_bev.shape)} != student {tuple(student_bev.shape)}')

            if self.use_mgd:
                if self.mgd_align is not None:
                    preds_S = self.mgd_align(student_bev)
                else:
                    preds_S = student_bev
                if self.use_spatialchannel:
                    mat = sample_spatialchannel_keep_mask(
                        preds_S, self.spatialchannel_cell_ratio,
                        self.spatialchannel_channel_ratio)
                else:
                    mat = sample_mgd_keep_mask(preds_S, self.mgd_lambda)
                new_bev = self.mgd_generation(torch.mul(preds_S, mat))
                loss_bev = F.mse_loss(new_bev, teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * self.mgd_alpha

            elif self.use_proj:
                bev_s_2d = self.bev_distill_proj(student_bev.float())
                loss_bev = F.mse_loss(bev_s_2d, teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * 100

            else:
                loss_bev = F.mse_loss(student_bev.float(), teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * 100

        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, given_bev=None, **kwargs):
        """BEVDet.simple_test; with ``given_bev`` the head runs on that BEV (no image branch)."""
        if given_bev is None:
            return super().simple_test(points, img_metas, img=img, rescale=rescale, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts([given_bev], img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def _parse_losses_mix(self, losses, weight=0.5):
        """Parse the raw outputs (losses) of the network.

        Same as DiffBEVFormerSegV3: loss = (1 - weight) * task + weight * bev, where 'bev'
        sums the loss terms whose name contains 'bev' and 'task' all other loss terms.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars).
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        total_loss = sum(_value for _key, _value in log_vars.items()
                         if 'loss' in _key)
        bev_loss = sum(_value for _key, _value in log_vars.items()
                       if 'loss' in _key and 'bev' in _key)
        task_loss = total_loss - bev_loss

        loss = (1 - weight) * task_loss + weight * bev_loss

        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if not isinstance(loss_value, torch.Tensor):
                loss_value = torch.tensor(float(loss_value), device=loss.device)
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
