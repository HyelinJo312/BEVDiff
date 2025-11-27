import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# T = [0, 1, 10, 100, 1000]
# NDS = [33.26, 33.84, 34.07, 34.07, 32.74]
# mAP = [23.48, 23.9, 24.10, 24.10, 22.80]

# NDS = [33.26, 33.84, 34.07, 34.03, 34.07, 34.03, 34.00, 33.80, 33.47, 32.75]
# mAP = [23.48, 23.9, 24.10, 24.10, 24.10, 24.09, 24.03, 23.87, 23.54, 22.78]

# NDS = [41.16, 41.17, 41.21, 41.20, 41.20, 41.09, 41.09, 40.42, 40.42, 40.01, 39.90]
# mAP = [25.95, 25.95, 25.94, 25.94, 25.94, 25.96, 25.87, 25.17, 25.17, 24.80, 24.69]

T = [0, 1, 10, 50, 100, 200, 300, 500, 700, 900, 1000]
NDS = [43.87, 43.88, 43.88, 43.89, 43.73, 43.73, 43.63, 42.98, 41.93, 41.44, 41.49]
mAP = [28.50, 28.51, 28.50, 28.49, 28.34, 28.34, 28.26, 27.80, 26.99, 26.64, 26.68]

nds = 43.06
map = 27.01

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

T_pos = [t for t in T if t > 0]
NDS_pos = [y for t, y in zip(T, NDS) if t > 0]
mAP_pos = [y for t, y in zip(T, mAP) if t > 0]

# 동일 간격 x축
# x_pos = list(range(len(T_nds)))
# xticklabels = [str(t) for t in T_nds]

# Figure 생성
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Subplot 1: NDS
axes[0].plot(T_pos, NDS_pos, marker='o', markersize=5, linewidth=2,
             color='royalblue', label='time-conditioned sampling')
axes[0].set_title('NDS')
axes[0].set_xlabel('Timestep')
axes[0].set_xscale('log')
axes[0].set_xlim(min(T_pos)*0.7, max(T)*1.3)   # 첫 포인트(1) 바로 앞에서 시작
axes[0].grid(True, linestyle='--', alpha=0.5, which='both')
axes[0].plot(1, nds, marker='x', markersize=8, markeredgewidth=3,
             linestyle='None', label='5-steps sampling (BEVDiffuser)', color='magenta')
axes[0].legend(fontsize=8)
axes[0].set_ylim(min(NDS) - 0.1, max(NDS) + 0.6)

# Subplot 2: mAP
axes[1].plot(T_pos, mAP_pos, marker='s', markersize=5, linewidth=2,
             color='royalblue', label='time-conditioned sampling')
axes[1].set_title('mAP')
axes[1].set_xlabel('Timestep')
axes[1].set_xscale('log')
axes[1].set_xlim(min(T_pos)*0.7, max(T)*1.3)   # 첫 포인트(1) 바로 앞에서 시작
axes[1].grid(True, linestyle='--', alpha=0.5, which='both')
axes[1].plot(1, map, marker='x', markersize=8, markeredgewidth=3,
             linestyle='None', label='5-steps sampling (BEVDiffuser)', color='magenta')
axes[1].legend(fontsize=8)
axes[1].set_ylim(min(mAP) - 0.1, max(mAP) + 0.6)

# === 테두리(spines) 제거 ===
# for ax in axes:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(1.2)  # X축 강조 (선택사항)
#     ax.spines['left'].set_linewidth(1.2)    # Y축 강조 (선택사항)

plt.tight_layout()

# 저장
os.makedirs('../../figures', exist_ok=True)
out_path = '../../figures/time-condition_samping_result.png'
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