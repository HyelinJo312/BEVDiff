"""Planar ego/LiDAR BEV transport for a frozen layout diffusion teacher."""

import torch
import torch.nn.functional as F


class TeacherBEVAlignment:
    def __init__(self, source_pc_range, teacher_pc_range, teacher_bev_size,
                 source_frame='ego', teacher_frame='lidar', ego_plane_z=0.0,
                 mask_invalid=True, min_valid_fraction=0.01):
        if (source_frame, teacher_frame) != ('ego', 'lidar'):
            raise ValueError('Only ego-to-LiDAR teacher BEV alignment is supported')
        self.source_pc_range = self._range(source_pc_range, 'source_pc_range')
        self.teacher_pc_range = self._range(teacher_pc_range, 'teacher_pc_range')
        if (len(teacher_bev_size) != 2 or
                any(int(size) != size or size < 1 for size in teacher_bev_size)):
            raise ValueError('teacher_bev_size must be two positive integers (H, W)')
        self.teacher_bev_size = tuple(int(size) for size in teacher_bev_size)
        self.ego_plane_z = float(ego_plane_z)
        self.mask_invalid = bool(mask_invalid)
        self.min_valid_fraction = float(min_valid_fraction)
        if not torch.isfinite(torch.tensor(self.ego_plane_z)):
            raise ValueError('ego_plane_z must be finite')
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError('min_valid_fraction must be in (0, 1]')

    @staticmethod
    def _range(value, name):
        if len(value) != 6:
            raise ValueError(f'{name} must contain six values')
        result = tuple(float(item) for item in value)
        bounds = torch.tensor(result)
        if not torch.isfinite(bounds).all() or not all(
                result[index + 3] > result[index] for index in range(3)):
            raise ValueError(f'{name} must have finite increasing XYZ bounds')
        return result

    def _planar_transform(self, img_metas, batch_size, device):
        if len(img_metas) != batch_size:
            raise ValueError('img_metas batch size does not match BEV batch size')
        matrices = []
        for meta in img_metas:
            if 'ego2lidar' not in meta:
                raise KeyError('ego2lidar is required for teacher BEV alignment')
            matrix = torch.as_tensor(meta['ego2lidar'], dtype=torch.float64, device=device)
            if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
                raise ValueError('ego2lidar must be a finite 4x4 matrix')
            if not torch.allclose(matrix[3], matrix.new_tensor([0, 0, 0, 1]),
                                  rtol=0, atol=1e-5):
                raise ValueError('ego2lidar must have a homogeneous last row')
            rotation = matrix[:3, :3]
            if (not torch.allclose(rotation.T @ rotation, torch.eye(3, device=device, dtype=torch.float64),
                                   rtol=0, atol=1e-4) or
                    not torch.isclose(torch.linalg.det(rotation), rotation.new_tensor(1),
                                      rtol=0, atol=1e-4)):
                raise ValueError('ego2lidar rotation must be rigid and right-handed')
            matrices.append(matrix)
        transforms = torch.stack(matrices)
        planar = transforms[:, :2, :2]
        condition = torch.linalg.cond(planar)
        if not torch.isfinite(condition).all() or (condition > 1e6).any():
            raise ValueError('ego2lidar planar transform is singular or ill-conditioned')
        offset = (transforms[:, :2, 3] +
                  transforms[:, :2, 2] * self.ego_plane_z)
        height_extent = max(abs(self.source_pc_range[2] - self.ego_plane_z),
                            abs(self.source_pc_range[5] - self.ego_plane_z))
        height_error_bound = torch.linalg.vector_norm(
            transforms[:, :2, 2], dim=1) * height_extent
        return planar, offset, height_error_bound

    @staticmethod
    def _grid(height, width, destination_range, source_range, source_points,
              device, dtype):
        y = (torch.arange(height, device=device, dtype=torch.float64) + 0.5)
        x = (torch.arange(width, device=device, dtype=torch.float64) + 0.5)
        y = destination_range[1] + y * (destination_range[4] - destination_range[1]) / height
        x = destination_range[0] + x * (destination_range[3] - destination_range[0]) / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        destination = torch.stack((grid_x, grid_y), dim=-1)
        source = source_points(destination)
        normalized_x = 2 * (source[..., 0] - source_range[0]) / (source_range[3] - source_range[0]) - 1
        normalized_y = 2 * (source[..., 1] - source_range[1]) / (source_range[4] - source_range[1]) - 1
        return torch.stack((normalized_x, normalized_y), dim=-1).to(dtype=dtype)

    @staticmethod
    def _sample(features, grid):
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=False)

    def to_teacher(self, source_bev, img_metas):
        if source_bev.ndim != 4 or not source_bev.is_floating_point():
            raise ValueError('source_bev must be floating BCHW')
        batch_size, _, source_height, source_width = source_bev.shape
        planar, offset, height_error_bound = self._planar_transform(
            img_metas, batch_size, source_bev.device)
        teacher_height, teacher_width = self.teacher_bev_size
        inverse = torch.linalg.inv(planar)
        grid = self._grid(
            teacher_height, teacher_width, self.teacher_pc_range,
            self.source_pc_range,
            lambda points: torch.einsum('bij,hwj->bhwi', inverse, points) -
            torch.einsum('bij,bj->bi', inverse, offset)[:, None, None, :],
            source_bev.device, source_bev.dtype)
        teacher_input = self._sample(source_bev.detach(), grid)
        footprint = self._sample(
            source_bev.new_ones((batch_size, 1, source_height, source_width)), grid)
        teacher_support = footprint >= 1 - 1e-6
        return teacher_input, teacher_support, (planar, offset, height_error_bound)

    def to_student(self, teacher_bev, teacher_support, img_metas,
                   source_size, transform=None):
        if teacher_bev.ndim != 4 or tuple(teacher_bev.shape[-2:]) != self.teacher_bev_size:
            raise ValueError('teacher_bev spatial shape does not match teacher_bev_size')
        if teacher_support.shape != (teacher_bev.shape[0], 1, *self.teacher_bev_size):
            raise ValueError('teacher_support must be B1HteacherWteacher')
        if len(source_size) != 2 or min(source_size) < 1:
            raise ValueError('source_size must be positive (H, W)')
        if transform is None:
            planar, offset, _ = self._planar_transform(
                img_metas, teacher_bev.shape[0], teacher_bev.device)
        else:
            planar, offset, _ = transform
        source_height, source_width = source_size
        grid = self._grid(
            source_height, source_width, self.source_pc_range,
            self.teacher_pc_range,
            lambda points: torch.einsum('bij,hwj->bhwi', planar, points) +
            offset[:, None, None, :],
            teacher_bev.device, teacher_bev.dtype)
        aligned = self._sample(teacher_bev.detach(), grid)
        support = self._sample(teacher_support.to(teacher_bev.dtype), grid)
        reverse_footprint = self._sample(
            teacher_bev.new_ones((teacher_bev.shape[0], 1, *self.teacher_bev_size)), grid)
        valid = (support >= 1 - 1e-6) & (reverse_footprint >= 1 - 1e-6)
        fractions = valid.flatten(1).float().mean(dim=1)
        if (fractions < self.min_valid_fraction).any():
            raise ValueError('Teacher/student BEV overlap below min_valid_fraction')
        return aligned, valid


def masked_bev_mse(student_bev, teacher_bev, valid_mask):
    if student_bev.shape != teacher_bev.shape or student_bev.ndim != 4:
        raise ValueError('Student and teacher BEV must have identical BCHW shape')
    if valid_mask.shape != (student_bev.shape[0], 1, *student_bev.shape[-2:]):
        raise ValueError('valid_mask must have shape B1HW')
    if valid_mask.dtype != torch.bool:
        raise TypeError('valid_mask must be boolean')
    if not valid_mask.any():
        raise ValueError('No valid BEV cells for distillation')
    difference = torch.where(
        valid_mask, student_bev.float() - teacher_bev.detach().float(), 0.)
    return difference.square().sum() / (valid_mask.sum() * student_bev.shape[1])
