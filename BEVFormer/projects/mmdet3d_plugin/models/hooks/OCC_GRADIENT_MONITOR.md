# Occupancy BEV gradient monitor

Edit `GRADIENT_MONITOR_ENABLED` and `GRADIENT_MONITOR_INTERVAL` in
`BEVFormer/tools/dist_train_onlyseg_cache_occ.sh`, then use the existing launcher
command. The active occupancy config defaults to `enabled=False`; the launcher
sets `True` and interval `200`. User `--cfg-options` or legacy `--options` are
forwarded unchanged; the trainer applies the two dedicated launcher flags after
config options, so the shell variables take precedence for those fields only.
The Python parser rejects using both option styles or an empty option list.

On the first training iteration and every interval thereafter, the hook probes
the original shared BEV tokens before optimizer backward. It measures gradients
of `0.5 × loss_occ` and `0.5 × loss_bev`, where `loss_bev` already includes MGD
alpha. It does not measure encoder parameter gradients. Global L2, RMS,
KD/task L2 ratio, pooled cosine, valid-cell conflict fraction, and realized
global MGD keep fraction are written under `work_dir/gradient_monitor/` as
unique-session JSONL and TensorBoard events. Metrics are pre-clipping;
undefined ratios and cosines are JSON `null` and omitted from TensorBoard.
The metadata record includes weights, checkpoint paths, sampler settings, seed,
resume path, and world size. Each probe records one-based iteration and epoch.
These are diagnostics only, not task losses or evidence of downstream gain.

## GT-layout direct-MSE occupancy

Edit `GRADIENT_MONITOR_ENABLED` and `GRADIENT_MONITOR_INTERVAL` in
`BEVFormer/tools/dist_train_layout_occ.sh`. The layout occupancy config also
defaults to `enabled=False`; the launcher enables probes every 200 iterations.
Dedicated flags are applied after `--cfg-options` or `--options`, so the shell
variables take precedence for these two settings. The launcher retains its
existing checkpoint safeguards and training command.

The layout head first extracts BEV tokens with `only_bev=True`, then decodes
them with `given_bev=bev`. The encoder and decoder each run once. The probe
uses the original shared `[B,H*W,C]` tokens returned by the first call, before
either branch's map conversion. It measures weighted task and direct-MSE KD
gradients at those tokens before optimizer backward, then maps gradients to
BCHW for the existing statistics. It does not probe a KD-only map copy.
The loss stays `0.5 × task + 0.5 × distill_loss_weight × MSE`; the active config
sets `distill_loss_weight=1.0`. The metadata records layout/MSE, target and
student initialization checkpoints separately, teacher path, sampling 5/5/5,
and the sampler's hardcoded CFG scale 2.0. `gradmon/mgd_keep_fraction` is
undefined (`null`) because direct MSE has no mask. Compare gradient metrics
with semantic MGD runs only after accounting for different losses, weights,
teachers, and sampling settings; they are not matched-condition outcomes.
