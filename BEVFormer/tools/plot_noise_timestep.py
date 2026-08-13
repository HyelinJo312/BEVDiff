import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

LABEL_OUTLINE = [pe.withStroke(linewidth=2.5, foreground="white")]

noise_timesteps = [5, 50, 100, 150, 200, 300, 500, 700, 900, 999]
x_pos = list(range(len(noise_timesteps)))

baseline_nds = [48.61, 47.60, 47.08, 46.70, 46.50, 46.13, 45.48, 43.91, 40.93, 38.40]
baseline_map = [34.56, 33.18, 32.66, 32.34, 32.11, 31.57, 30.14, 27.25, 21.51, 17.13]

ours_nds = [49.79, 53.41, 53.69, 53.61, 53.64, 53.55, 53.08, 51.57, 49.08, 47.75]
ours_map = [37.81, 40.89, 41.09, 40.90, 40.92, 40.66, 39.97, 38.02, 34.17, 31.60]

# baseline_nds = [48.6, 47.6, 47.1, 46.7, 46.5, 45.5, 38.4]
# baseline_map = [34.6, 33.2, 32.7, 32.3, 32.1, 30.1, 17.1]

# ours_nds = [49.8, 53.4, 53.7, 53.6, 53.6, 53.1, 47.8]
# ours_map = [37.8, 40.9, 41.1, 40.9, 40.9, 40.0, 31.6]


COLOR_BASELINE = "#E07B54"
COLOR_OURS     = "#4C9BE8"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Detection Performance with Different Noise Timesteps", fontsize=15, y=0.98)

for ax, baseline_vals, ours_vals, metric in zip(
    axes,
    [baseline_nds, baseline_map],
    [ours_nds,     ours_map],
    ["NDS",    "mAP"],
):
    # Forward-diffusion regime: connected line.
    ax.plot(
        x_pos, baseline_vals,
        marker="o", linewidth=2, markersize=7,
        color=COLOR_BASELINE, label="BEVDiffuser",
    )
    ax.plot(
        x_pos, ours_vals,
        marker="s", linewidth=2, markersize=7,
        color=COLOR_OURS, label="Ours",
    )

    for xi, yb, yo in zip(x_pos, baseline_vals, ours_vals):
        ax.annotate(
            f"{yb:.1f}", (xi, yb), textcoords="offset points",
            xytext=(0, -16), ha="center", va="top",
            fontsize=11, fontweight="bold", color=COLOR_BASELINE,
            path_effects=LABEL_OUTLINE,
        )
        ax.annotate(
            f"{yo:.1f}", (xi, yo), textcoords="offset points",
            xytext=(0, 16), ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=COLOR_OURS,
            path_effects=LABEL_OUTLINE,
        )

    ax.set_xlabel("Noise Timestep", fontsize=14, labelpad=12)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(noise_timesteps)

    all_vals = baseline_vals + ours_vals
    margin = (max(all_vals) - min(all_vals)) * 0.32
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

axes[1].legend(
    loc="lower left",
    fontsize=13,
    frameon=True,
)

plt.tight_layout()
output_path = "noise_timestep_ablation_v4.pdf"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
plt.show()
