import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

T = [0, 1, 10, 100, 1000]
NDS = [33.26, 33.84, 34.07, 34.07, 32.74]
mAP = [23.48, 23.9, 24.10, 24.10, 22.80]

nds = 32.59
map = 22.54


# None 제거
def drop_none(x_list, y_list):
    xs, ys = [], []
    for x, y in zip(x_list, y_list):
        if y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys

T_nds, NDS_v = drop_none(T, NDS)
T_map, mAP_v = drop_none(T, mAP)

# 동일 간격 x축
x_pos = list(range(len(T_nds)))
xticklabels = [str(t) for t in T_nds]

# Figure 생성
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Subplot 1: NDS
axes[0].plot(x_pos, NDS_v, marker='o', linewidth=2, color='royalblue')
axes[0].set_title('NDS')
axes[0].set_xlabel('Timestep')
# axes[0].set_ylabel('Score')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(xticklabels)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].plot(x_pos[0], nds, marker='x', markersize=7, markeredgewidth=2, linestyle='None', label='K steps sampling', color='royalblue')
axes[0].legend(fontsize=8)
axes[0].set_ylim(min(NDS_v) - 0.5, max(NDS_v) + 0.5)  # 자동으로 약간의 패딩 추가

# Subplot 2: mAP
axes[1].plot(x_pos, mAP_v, marker='s', linewidth=2, color='darkorange')
axes[1].set_title('mAP')
axes[1].set_xlabel('Timestep')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(xticklabels)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].plot(x_pos[0], map, marker='x', markersize=7, markeredgewidth=2, linestyle='None', label='K steps sampling', color='darkorange')
axes[1].legend(fontsize=8)
axes[1].set_ylim(min(mAP_v) - 0.5, max(mAP_v) + 0.5)  # 자동으로 약간의 패딩 추가

plt.tight_layout()

# 저장
os.makedirs('figures', exist_ok=True)
out_path = 'figures/performance_vs_T_results_v2.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved figure to: {out_path}")


# def drop_none(x_list, y_list):
#     x_new, y_new = [], []
#     for x, y in zip(x_list, y_list):
#         if y is not None:
#             x_new.append(x)
#             y_new.append(y)
#     return x_new, y_new

# T_nds, NDS_valid = drop_none(T, NDS)
# T_map, mAP_valid = drop_none(T, mAP)

# # x축을 동일 간격으로 하기 위해 인덱스 기반으로 표현
# x_positions = range(len(T_nds))

# # 그래프
# plt.figure(figsize=(7, 5))
# plt.plot(x_positions, NDS_valid, marker='o', color='royalblue', linewidth=2, label='NDS')
# plt.plot(x_positions, mAP_valid, marker='s', color='orange', linewidth=2, label='mAP')

# # 축 설정
# plt.xticks(ticks=x_positions, labels=[str(t) for t in T_nds])  # 실제 T값을 눈금으로 표시
# plt.xlabel('Timestep', fontsize=12)
# # plt.ylabel('Score', fontsize=12)
# plt.title('Performance vs. Timestep', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=12)
# plt.tight_layout()

# # 저장
# save_dir = "figures"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "Performance_vs_T_v2.png")

# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"✅ Figure saved to: {save_path}")


if __name__ == "__main__":
    pass