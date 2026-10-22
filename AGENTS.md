# BEVDiff Agent Guide

## Scope

These instructions apply to the entire repository. All agents must read this file before acting and must also read `methodology_stage1_stage2_summary.md` when a task affects the research method, experiments, results, or paper claims.

## Research Objective

The primary contribution is a task-agnostic conditioning interface for BEV diffusion. The main method replaces BEVDiffuser's detection-specific GT layout condition with a dense semantic BEV prior.

- Stage 1 trains a task-head-free semantic-only diffusion teacher in BEV feature space.
- Stage 2 distills the frozen teacher into a downstream BEV student with task loss and feature distillation.
- 3D detection is the first validation task, not a SOTA claim.
- BEV segmentation is the key extension for testing whether the same conditioning interface transfers across tasks.
- Prefer the claims `task-agnostic conditioning interface` and `task-transferable BEV diffusion teacher` over `fully task-agnostic representation learning`.

## Sources of Truth

Use evidence in this order when reconstructing an experiment:

1. The run's dumped config and logs in its work directory.
2. The exact checkpoint and its state-dict keys.
3. The launch script and recorded command.
4. The Git commit plus documented working-tree state.
5. The current config and source files.

Do not infer a historical run solely from the current config. The flagship semantic-only run used gated SBAM even though the active source may not contain that implementation, and historical cache settings may differ from the current working tree.

## Collaboration Model

The main Codex agent owns task decomposition, user intent, synthesis, and the final response.

For research or implementation tasks, use the following project agents:

- `methodology_reviewer`: validates hypotheses, controlled comparisons, ablations, and claim strength.
- `repro_auditor`: reconstructs actual experiment conditions and finds reproducibility gaps.
- `experiment_analyst`: validates matched comparisons, metrics, statistics, and feature-alignment evidence.
- `bev_engineer`: implements approved changes and performs targeted verification.

The first three agents are read-only and may investigate independent questions in parallel. The main agent must collect and reconcile their findings before delegating implementation. During a delegated multi-agent workflow, `bev_engineer` is the only subagent allowed to modify repository files. Avoid concurrent write-heavy tasks.

Use agents only when their specialization materially helps. Small, well-scoped tasks may be handled directly by the main agent. Do not delegate merely to increase the number of active agents.

## Required Workflow

### Research and design

1. Define the exact hypothesis or decision.
2. Ask `methodology_reviewer` to identify confounds, necessary controls, and acceptance criteria.
3. Ask `repro_auditor` to establish the actual state of relevant runs, configs, checkpoints, and code.
4. Ask `experiment_analyst` to assess whether the proposed metrics and comparisons can support the claim.
5. Reconcile disagreements and produce one bounded implementation brief.
6. Give that brief to `bev_engineer` if code changes are required.

### Implementation

1. Inspect `git status` and preserve all user changes.
2. Trace the active execution path before editing.
3. Make the smallest coherent change that implements one experimental axis.
4. Expose important ablations as explicit config flags rather than undocumented source swaps.
5. Add or update focused tests for tensor shapes, gradient flow, frozen modules, loss weights, and config wiring where applicable.
6. Run the narrowest useful verification first and report the exact commands and outcomes.

### Experiment analysis

- Compare only matched epochs, datasets, seeds, teacher checkpoints, and Stage 2 settings unless differences are explicitly labeled.
- Report absolute metrics, deltas, run counts, seeds, and variance when available.
- Never invent missing metrics or silently combine values from different runs.
- Treat MSE, cosine similarity, and CKA as complementary measurements, not interchangeable evidence.
- For feature alignment, distinguish global, foreground, semantic-boundary, static/background, and class-wise behavior where possible.
- A lower feature MSE does not by itself prove better downstream transfer.

## Safety and Resource Rules

- Never discard, overwrite, or revert existing user changes unless explicitly requested.
- Never delete or overwrite datasets, semantic BEV caches, checkpoints, work directories, or result artifacts.
- Do not run full training, multi-GPU jobs, large cache generation, or expensive evaluation without explicit user authorization.
- Prefer shape checks, unit tests, synthetic inputs, one-batch smoke tests, and config validation for routine verification.
- Do not change dataset paths, checkpoint paths, CUDA settings, GPU counts, or distributed launch settings without reporting the change.
- Treat shell scripts as executable experiment specifications; keep run names and output directories unambiguous.
- Do not claim reproducibility without identifying the config, code state, checkpoint, data/cache source, seed, and command.

## Important Research Risks

Always check for these known confounds:

- Semantic-only and original teachers may differ in both conditioning and `task_loss_scale`.
- Gated versus ungated SBAM may be encoded only in checkpoint keys or historical source state.
- Cache and live projection paths may not use identical runtime inputs.
- A config filename may be shared by runs that used different uncommitted code.
- The clean BEV target can remain detection-trained even when the conditioning interface is task-agnostic.
- Small mAP/NDS improvements require repeated seeds or another variance argument.
- Detection-only results are insufficient for the strongest cross-task claim; BEV segmentation is the decisive extension.

## Reporting Standard

Every agent report should separate:

- verified facts with file paths, symbols, log lines, or checkpoint evidence;
- inferences and their assumptions;
- unresolved questions;
- recommended next actions, ordered by impact and cost.

Code-change reports must also list modified files, verification commands, observed results, and any verification that could not be completed.
