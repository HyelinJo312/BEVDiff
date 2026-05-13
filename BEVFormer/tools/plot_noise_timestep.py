import matplotlib.pyplot as plt

noise_timesteps = [5, 50, 100, 150, 200, 500, 999]
x_pos = list(range(len(noise_timesteps)))

baseline_nds = [48.61, 47.60, 47.08, 46.70, 46.50, 45.48, 38.40]
baseline_map = [34.56, 33.18, 32.66, 32.34, 32.11, 30.14, 17.13]

ours_nds = [49.79, 53.41, 53.69, 53.61, 53.64, 53.08, 47.75]
ours_map = [37.81, 40.89, 41.09, 40.90, 40.92, 39.97, 31.60]

COLOR_BASELINE = "#E07B54"
COLOR_OURS     = "#4C9BE8"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Detection Performance with Different Noise Timesteps", fontsize=17, y=0.98)

for ax, baseline_vals, ours_vals, metric in zip(
    axes,
    [baseline_nds, baseline_map],
    [ours_nds,     ours_map],
    ["NDS (%)",    "mAP (%)"],
):
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
        ax.annotate(f"{yb:.2f}", (xi, yb), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=COLOR_BASELINE)
        ax.annotate(f"{yo:.2f}", (xi, yo), textcoords="offset points",
                    xytext=(0,  6),  ha="center", fontsize=8.5, color=COLOR_OURS)

    ax.set_xlabel("Noise Timestep T", fontsize=14, labelpad=12)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(noise_timesteps)

    all_vals = baseline_vals + ours_vals
    margin = (max(all_vals) - min(all_vals)) * 0.25
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    fontsize=13,
    frameon=True,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
output_path = "noise_timestep_ablation.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
plt.show()
