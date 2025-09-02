# projects/mmdet3d_plugin/hooks/fix_unet_lr_hook.py
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FixUnetLrHook(Hook):
    """UNet 파라미터가 들어있는 optimizer param_group의 lr을 항상 고정."""
    def __init__(self, unet_lr=1e-3, unet_attr_path='module.pts_bbox_head.unet', print_every=100):
        self.unet_lr = unet_lr
        self.unet_attr_path = unet_attr_path
        self._unet_param_ids = None
        self._unet_group_indices = None
        self.print_every = print_every

    def before_run(self, runner):
        # 모델에서 UNet 모듈을 찾아 파라미터 id 집합을 만든다.
        module = runner.model
        # DDP/FP16 래핑을 고려한 안전 접근 (module.module 형태)
        obj = module
        for name in self.unet_attr_path.split('.'):
            obj = getattr(obj, name)
        unet_params = list(obj.parameters())
        self._unet_param_ids = set(id(p) for p in unet_params)

        # 어떤 param_group이 UNet 파라미터를 포함하는지 미리 찾아둔다.
        self._unet_group_indices = []
        for gi, g in enumerate(runner.optimizer.param_groups):
            params = g['params']
            if any(id(p) in self._unet_param_ids for p in params):
                self._unet_group_indices.append(gi)

        if len(self._unet_group_indices) == 0:
            runner.logger.warning(
                '[FixUnetLrHook] No optimizer param_group matched UNet params. '
                'Check unet_attr_path or your model structure.'
            )

    def before_train_iter(self, runner):
        if not self._unet_group_indices:
            return
        # 매 iter마다 해당 그룹들의 lr을 원하는 값으로 강제로 고정
        for gi in self._unet_group_indices:
            runner.optimizer.param_groups[gi]['lr'] = self.unet_lr

        # 주기적으로 그룹별 LR 출력
        # if runner.iter % self.print_every == 0:
        #     lrs = [g['lr'] for g in runner.optimizer.param_groups]
        #     runner.logger.info(f'[DEBUG] iter={runner.iter} group LRs={lrs}')
        #     if self._unet_group_indices:
        #         unet_lrs = [runner.optimizer.param_groups[gi]['lr'] for gi in self._unet_group_indices]
        #         runner.logger.info(f'[DEBUG] UNet LRs={unet_lrs}')