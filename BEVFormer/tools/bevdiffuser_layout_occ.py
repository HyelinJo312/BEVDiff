"""Layout BEVDiffuser wrapper isolated for the Occ3D Stage-2 entrypoint."""

from bevdiffuser import BEVDiffuser as _LayoutBEVDiffuser
from projects.bevdiffuser.occ_diffusion_utils import load_diffusion_weights


class BEVDiffuser(_LayoutBEVDiffuser):
    """The original GT-layout diffuser; spatial size comes from its config."""

    def __init__(self, *args, unet_checkpoint_dir=None, **kwargs):
        super().__init__(
            *args, unet_checkpoint_dir=unet_checkpoint_dir, **kwargs)
        # The upstream wrapper ignores the model returned by the Diffusers
        # classmethod. Load into this already-configured 200x200 UNet directly.
        load_diffusion_weights(self.unet, unet_checkpoint_dir)
        self.unet.requires_grad_(False)
