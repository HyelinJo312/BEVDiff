# Local copy of the Stable Diffusion 2.1 scheduler config

The Stage 1/2 scripts call `DDPMScheduler.from_pretrained(<path>, subfolder="scheduler")` (and
`DDIMGuidedScheduler`) with `stabilityai/stable-diffusion-2-1`, the setting of the flagship run
(tensorboard hparams of `BEVDiffuser_tiny_onlyseg_sam3_metric3d`). On 2026-11-04 that HF repo
returned 401 and this node had no HF cache, so the config is kept here and passed as a local path.
Only the scheduler is used (no SD weights); `prediction_type` is overridden to `sample` by the script.

Values are identical to `scheduler/scheduler_config.json` of two public mirrors fetched on
2026-11-04: `philschmid/stable-diffusion-2-1` (DDIMScheduler, diffusers 0.8.0, byte-identical keys)
and `flax/stable-diffusion-2-1` (FlaxDDIMScheduler, same values).
