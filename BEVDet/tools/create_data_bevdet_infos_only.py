# Info-only variant of tools/create_data_bevdet.py for camera-only BEVDet:
# builds bevdetv3-nuscenes_infos_{train,val}.pkl and skips the LiDAR GT database.
# Run from the BEVDet root: python tools/create_data_bevdet_infos_only.py
from tools.create_data_bevdet import add_ann_adj_info, nuscenes_data_prep

if __name__ == '__main__':
    extra_tag = 'bevdetv3-nuscenes'
    nuscenes_data_prep(
        root_path='./data/nuscenes',
        info_prefix=extra_tag,
        version='v1.0-trainval',
        max_sweeps=10)
    add_ann_adj_info(extra_tag)
