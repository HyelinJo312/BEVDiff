from .nuscenes_dataset import CustomNuScenesDataset
# Import the module, not names from it: spawned DataLoader workers unpickle the
# dataset by importing projects.bevdiffuser.data_utils first, which imports this
# package while data_utils is still partially initialized. A module import is
# safe in that cycle; data_utils registers its datasets once it finishes loading.
import projects.bevdiffuser.data_utils  # noqa: F401
from .builder import custom_build_dataset

__all__ = [
    'CustomNuScenesDataset',
]
