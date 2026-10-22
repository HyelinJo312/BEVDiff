"""Stage 1 Metric3D semantic teacher with decoder SBAM timestep scale/shift.

Inherits the data, BEV target, and all other UNet settings from the reference
config. Select a distinct run/output name when launching this experiment.
"""

_base_ = ['./bev_tiny_onlyseg_sam_v2_metric3d.py']

unet = dict(parameters=dict(sbam_use_scale_shift_norm=True))
