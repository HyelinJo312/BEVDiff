"""BEVDet-side plugin for the semantic-only BEV diffusion teacher (Stage 2 distillation).

Kept separate from ``projects/mmdet3d_plugin`` (the BEVFormer plugin), whose package import
fails under BEVDet's mmdet3d 1.0 (duplicate registry names such as HungarianAssigner3D), and
from ``mmdet3d/models`` so the BEVDet4D baseline code stays untouched. Registered through
``custom_imports = dict(imports=['projects.bevdet_plugin'])`` in the Stage 2 config.
"""
from .detectors import *  # noqa: F401,F403
from .runner import *  # noqa: F401,F403
