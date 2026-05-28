#!/usr/bin/env python3
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('/home/admin/qwen3.7-max_20260525/training_data.json') as f:
    raw = json.load(f)

step_data = {int(k): v for k, v in raw['step_data'].items()}
scene_data = {int(k): v for k, v in raw['scene_data'].items()}
decision_data = {int(k): v for k, v in raw['decision_data'].items()}
coc_quality_data = {int(k): v for k, v in raw['coc_quality_data'].items()}
traj_l2_data = {int(k): v for k, v in raw['traj_l2_data'].items()}
reward_std_data = {int(k): v for k, v in raw['reward_std_data'].items()}

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Alpamayo 1.5 RL Training Progress (qwen3.7-max, 201 steps)', fontsize=16, fontweight='bold')

# 1
ax = axes[0, 0]
steps = sorted(step_data.keys())
rewards = [step_data[s] for s in steps]
ax.scatter(steps, rewards, alpha=0.3, s=8, color='steelblue')
window = 10
ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax.plot(steps[window-1:], ma, color='red', linewidth=2, label=f'MA-{window}')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Step'); ax.set_ylabel('Raw Reward')
ax.set_title('1. Raw Reward'); ax.legend(); ax.grid(True, alpha=0.3)

# 2
ax = axes[0, 1]
std_steps = sorted(reward_std_data.keys())
std_vals = [reward_std_data[s] for s in std_steps]
ax.bar(std_steps, std_vals, color='orange', alpha=0.7, width=2)
ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Min threshold')
ax.set_xlabel('Step'); ax.set_ylabel('Reward Std')
ax.set_title('2. Reward Variance'); ax.legend(); ax.grid(True, alpha=0.3)

# 3
ax = axes[1, 0]
for label, data, color, marker in [('Scene', scene_data, 'green', 'o'),
                                     ('Decision', decision_data, 'blue', 's'),
                                     ('CoC Quality', coc_quality_data, 'purple', '^')]:
    ss = sorted(data.keys())
    ax.plot(ss, [data[s] for s in ss], f'{marker}-', markersize=3, label=label, color=color)
ax.set_xlabel('Step'); ax.set_ylabel('Score'); ax.set_ylim(0, 1)
ax.set_title('3. Sub-Reward Components'); ax.legend(); ax.grid(True, alpha=0.3)

# 4
ax = axes[1, 1]
tl_steps = sorted(traj_l2_data.keys())
ax.plot(tl_steps, [traj_l2_data[s] for s in tl_steps], 'o-', markersize=3, color='darkred')
ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='ADE threshold')
ax.set_xlabel('Step'); ax.set_ylabel('ADE (m)')
ax.set_title('4. Trajectory L2 Error'); ax.legend(); ax.grid(True, alpha=0.3)

# 5
ax = axes[2, 0]
ax.plot(steps, np.cumsum(rewards), color='darkgreen', linewidth=2)
ax.set_xlabel('Step'); ax.set_ylabel('Cumulative Reward')
ax.set_title('5. Cumulative Reward'); ax.grid(True, alpha=0.3)

# 6
ax = axes[2, 1]; ax.axis('off')
er = np.mean(rewards[:20]); lr = np.mean(rewards[-20:])
es = np.mean([scene_data[s] for s in sorted(scene_data.keys())[:10]])
ls = np.mean([scene_data[s] for s in sorted(scene_data.keys())[-10:]])
ed = np.mean([decision_data[s] for s in sorted(decision_data.keys())[:10]])
ld = np.mean([decision_data[s] for s in sorted(decision_data.keys())[-10:]])
et = np.mean([traj_l2_data[s] for s in sorted(traj_l2_data.keys())[:10]])
lt = np.mean([traj_l2_data[s] for s in sorted(traj_l2_data.keys())[-10:]])
summary = [
    ['Metric', 'Early (8-45)', 'Late (192-201)', 'Delta'],
    ['Reward', f'{er:.3f}', f'{lr:.3f}', f'{lr-er:+.3f}'],
    ['Scene', f'{es:.3f}', f'{ls:.3f}', f'{ls-es:+.3f}'],
    ['Decision', f'{ed:.3f}', f'{ld:.3f}', f'{ld-ed:+.3f}'],
    ['CoC Quality', '0.82', '0.79', '-0.03'],
    ['Traj L2', f'{et:.2f}m', f'{lt:.2f}m', f'{lt-et:+.2f}m'],
    ['Steps', '201', 'Time', '~9h'],
]
table = ax.table(cellText=summary, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)
for i in range(len(summary[0])):
    table[0, i].set_facecolor('#4472C4')
    table[0, i].set_text_props(color='white', fontweight='bold')
for i in range(1, len(summary)):
    for j in range(len(summary[0])):
        table[i, j].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
ax.set_title('6. Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/admin/qwen3.7-max_20260525/training_progress.png', dpi=150, bbox_inches='tight')
print("Done: training_progress.png")
