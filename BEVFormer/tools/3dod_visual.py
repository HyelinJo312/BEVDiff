# Clean, log-scale visualization of NDS and mAP
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FixedLocator, MultipleLocator

# Data
T = [0, 1, 10, 50, 100, 200, 300, 500, 700, 900, 1000]
NDS = [43.87, 43.88, 43.88, 43.89, 43.73, 43.73, 43.63, 42.98, 41.93, 41.44, 41.49]
mAP = [28.50, 28.51, 28.50, 28.49, 28.34, 28.34, 28.26, 27.80, 26.99, 26.64, 26.68]

baseline_nds = 43.06
baseline_map = 27.01

# Helper
def drop_none(x_list, y_list):
    xs, ys = [], []
    for x, y in zip(x_list, y_list):
        if y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys

T_pos = [t for t in T if t > 0]
NDS_pos = [y for t, y in zip(T, NDS) if t > 0]
mAP_pos = [y for t, y in zip(T, mAP) if t > 0]

# === Plot style ===
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
# plt.subplots_adjust(bottom=0.28, wspace=0.35)
plt.subplots_adjust(bottom=0.22, wspace=0.35)


# === Subplot 1: NDS ===
ax = axes[0]
ax.plot(T_pos, NDS_pos, marker='o', markersize=5, linewidth=2, color='royalblue',
        label='Time-conditioned sampling')
ax.plot(1, baseline_nds, marker='*', markersize=10, markeredgewidth=1.0,
        linestyle='None', label='5-step sampling (BEVDiffuser)', color='dodgerblue')

ax.set_title('NDS')
ax.set_xlabel('Timestep')
ax.set_xscale('log')
ax.set_xlim(min(T_pos) * 0.6, max(T_pos) * 1.5)  # 여유 있는 시작
ax.margins(y=0.1)
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          frameon=True, facecolor='white', edgecolor='gray', ncol=1)
legend.get_frame().set_linewidth(0.6)
# x축 라벨을 간결하게 (로그 눈금 중 필요한 것만 표시)
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.tick_params(axis='x', rotation=0)
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# === Subplot 2: mAP ===
ax = axes[1]
ax.plot(T_pos, mAP_pos, marker='o', markersize=5, linewidth=2, color='tomato',
        label='Time-conditioned sampling')
ax.plot(1, baseline_map, marker='*', markersize=10, markeredgewidth=1.0,
        linestyle='None', label='5-step sampling (BEVDiffuser)', color='orange')

ax.set_title('mAP')
ax.set_xlabel('Timestep')
ax.set_xscale('log')
ax.set_xlim(min(T_pos) * 0.6, max(T_pos) * 1.5)
ax.margins(y=0.1)
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          frameon=True, facecolor='white', edgecolor='gray', ncol=1)
legend.get_frame().set_linewidth(0.6)

ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.tick_params(axis='x', rotation=0)
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# === 테두리 유지 ===
for ax in axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)

# Save
os.makedirs('../../figures', exist_ok=True)
out_path = '../../figures/time_conditioned_sampling_result_clean_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved figure to: {out_path}")

