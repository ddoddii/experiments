"""
vllm-ppd 2P2D: TOOL_DELAY 0s vs 3s predicted throughput comparison

Estimation model:
  C=1  (sequential): new_wall = wall_0s + N_items * tc_turns * D
  C>=2 (concurrent): new_wall = wall_0s + n_success * tc_turns * D / C
  tput_3s = total_tokens / new_wall

Assumptions:
  - avg tool-call turns per item = 3.0  (p_tool=0.8 * 3.7 avg turns)
  - failed items (C=12: 68) abort early -> no added delay
  - output token count unchanged by delay
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Measured data (TOOL_DELAY=0s) ───────────────────────────────────────────
#  C: (tput tok/s, n_success, wall_time_s, total_tokens)
measured = {
    1:  (11.0, 200, 732,  8075),
    8:  (63.7, 200, 148,  9424),
    12: (22.1, 132, 197,  4352),
}

N_ITEMS   = 200
TC_TURNS  = 3.0   # avg tool-call turns per item
DELAY     = 3.0   # seconds

# ─── Estimate: TOOL_DELAY=3s ─────────────────────────────────────────────────
estimated_3s = {}
for C, (tput_0, n_ok, wall_0, tokens) in measured.items():
    if C == 1:
        added = N_ITEMS * TC_TURNS * DELAY          # all 200 items complete
    else:
        added = n_ok * TC_TURNS * DELAY / C         # only successful items add delay
    new_wall = wall_0 + added
    estimated_3s[C] = (tokens / new_wall, n_ok, new_wall, tokens)

# Print estimates for verification
print("Throughput estimates:")
for C in [1, 8, 12]:
    t0 = measured[C][0]
    t3, _, w3, _ = estimated_3s[C]
    print(f"  C={C:2d}:  0s={t0:.1f} tok/s  ->  3s={t3:.1f} tok/s  "
          f"(drop {(1-t3/t0)*100:.0f}%,  new_wall={w3:.0f}s)")

# ─── Layout ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor('#f7f7f7')

BLUE   = '#2C72B5'
ORANGE = '#E05C2E'
GRAY   = '#888888'

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Bar chart: direct comparison by concurrency level
# ═══════════════════════════════════════════════════════════════════════════════
ax1 = axes[0]
ax1.set_facecolor('#fdfdfd')

C_list = [1, 8, 12]
t0_vals = [measured[c][0]     for c in C_list]
t3_vals = [estimated_3s[c][0] for c in C_list]

x = np.arange(len(C_list))
w = 0.36

bars0 = ax1.bar(x - w/2, t0_vals, w,
                label='TOOL_DELAY = 0s  (measured)',
                color=BLUE, alpha=0.88, edgecolor='white', linewidth=1.2)
bars3 = ax1.bar(x + w/2, t3_vals, w,
                label='TOOL_DELAY = 3s  (estimated)',
                color=ORANGE, alpha=0.88, edgecolor='white', linewidth=1.2,
                hatch='///')

# Value labels on bars
for bar, val in zip(bars0, t0_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10,
             fontweight='bold', color=BLUE)
for bar, val in zip(bars3, t3_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10,
             fontweight='bold', color=ORANGE)

# Drop percentage arrows
for i, (v0, v3) in enumerate(zip(t0_vals, t3_vals)):
    drop_pct = (1 - v3/v0) * 100
    ax1.annotate('', xy=(x[i]+w/2, v3+0.5), xytext=(x[i]-w/2, v0-0.5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.8,
                                 connectionstyle='arc3,rad=0.30'))
    ax1.text(x[i] + 0.07, (v0+v3)/2 + 5,
             f'-{drop_pct:.0f}%', ha='center', fontsize=9.5,
             color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='red', alpha=0.85))

# Success rate note below x-axis
success_labels = {1: '200/200 ok', 8: '200/200 ok', 12: '132/200 ok'}
for i, c in enumerate(C_list):
    ax1.text(x[i], -5.5, success_labels[c], ha='center', fontsize=7.5,
             color='#555', style='italic')

ax1.set_xticks(x)
ax1.set_xticklabels([f'C={c}' for c in C_list], fontsize=12)
ax1.set_ylabel('Overall Throughput  (tok/s)', fontsize=11)
ax1.set_title('vllm-ppd 2P2D — TOOL_DELAY 0s vs 3s\nOverall Throughput (Llama-3.1-8B)', fontsize=12.5)
ax1.legend(fontsize=9.5, loc='upper center', framealpha=0.9)
ax1.set_ylim(-9, 82)
ax1.axhline(0, color='#ccc', linewidth=0.8)
ax1.grid(axis='y', alpha=0.3, linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.text(0.02, 0.97,
         'Estimate: wall_new = wall_0s + n_ok x 3turns x 3s / C',
         transform=ax1.transAxes, fontsize=7.5, color='#777',
         va='top', style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Line chart: throughput curve + insight annotation
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]
ax2.set_facecolor('#fdfdfd')

c_pts  = [1, 8, 12]
t0_pts = [measured[c][0]     for c in c_pts]
t3_pts = [estimated_3s[c][0] for c in c_pts]

ax2.plot(c_pts, t0_pts, 'o-', color=BLUE, linewidth=2.5, markersize=10,
         label='TOOL_DELAY = 0s  (measured)', zorder=5)
ax2.plot(c_pts, t3_pts, 's--', color=ORANGE, linewidth=2.5, markersize=10,
         label='TOOL_DELAY = 3s  (estimated)', zorder=5,
         markerfacecolor='white', markeredgewidth=2.5)

# Fill between: throughput lost
ax2.fill_between(c_pts, t3_pts, t0_pts,
                 alpha=0.13, color=GRAY, label='Throughput lost to tool delay',
                 interpolate=True)

# Annotate C=8 key values
ax2.annotate('63.7 tok/s', xy=(8, 63.7), xytext=(9.1, 68),
             fontsize=9.5, color=BLUE, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
ax2.annotate('25.3 tok/s\n(est.)', xy=(8, 25.3), xytext=(9.1, 18),
             fontsize=9.5, color=ORANGE, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))

# Vertical drop arrow at C=8
ax2.annotate('', xy=(8, 25.3+1), xytext=(8, 63.7-1),
             arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax2.text(8.25, 44, '-60%', fontsize=10, color='red', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                   edgecolor='red', alpha=0.85))

# C=1 baseline reference
ax2.axhline(y=11.0, color=BLUE, linestyle=':', alpha=0.4, linewidth=1.2)
ax2.text(12.5, 11.5, 'C=1\nbaseline\n(0s)', fontsize=7.5,
         color=BLUE, va='center', ha='left')

# ─── Insight text box ─────────────────────────────────────────────────────────
insight = (
    "Why throughput drops with 3s delay\n"
    "─────────────────────────────────────\n"
    "Per-turn timeline (C=1):\n"
    "  Compute   ~1.0s  [GPU busy  ]\n"
    "  Tool wait  3.0s  [GPU IDLE  ]\n\n"
    "GPU idle fraction = 3.0/(1.0+3.0) = 75%\n\n"
    "Effective concurrent load (C=8):\n"
    "  8 x 1.0/(1.0+3.0) = 2.0 workers\n"
    "  => server utilization drops ~75%\n\n"
    "To recover C=8 throughput with 3s delay\n"
    "  => need C ~ 8 x 4 = 32 workers\n\n"
    "Research opportunity:\n"
    "  · KV tiering frees GPU during idle\n"
    "  · P-node prefetches next request\n"
    "  · Placement policy predicts delay"
)
ax2.text(0.015, 0.985, insight,
         transform=ax2.transAxes, fontsize=8.0,
         va='top', ha='left', family='monospace',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#fffde7',
                   edgecolor='#d4a017', alpha=0.94, linewidth=1.5))

# ─── Per-turn timeline mini-diagram ──────────────────────────────────────────
tl_y = -12
tl_h = 4.5
n_turns = 3

for i in range(n_turns):
    t_start = i * (1.0 + 3.0)
    # Compute segment (blue)
    ax2.barh(tl_y, 1.0, left=t_start, height=tl_h,
             color=BLUE, alpha=0.75, align='center')
    ax2.text(t_start + 0.5, tl_y, 'T', ha='center', va='center',
             fontsize=7, color='white', fontweight='bold')
    # Delay segment (gray)
    if i < n_turns:
        ax2.barh(tl_y, 3.0, left=t_start + 1.0, height=tl_h,
                 color='#cccccc', alpha=0.55, align='center',
                 hatch='....', edgecolor='white')
        ax2.text(t_start + 2.5, tl_y, 'idle', ha='center', va='center',
                 fontsize=7, color='#888')

ax2.text(-0.4, tl_y, 'GPU\n(C=1)', ha='right', va='center',
         fontsize=7.5, color='#444')
ax2.text(0.5, tl_y - tl_h/2 - 1.5, 'compute', ha='center',
         fontsize=6.5, color=BLUE)
ax2.text(2.5, tl_y - tl_h/2 - 1.5, '--- tool call delay (idle) ---',
         ha='center', fontsize=6.5, color='#999')

ax2.set_xlim(-0.6, 13.8)
ax2.set_ylim(-22, 82)
ax2.set_xticks([1, 4, 8, 12])
ax2.set_xlabel('Concurrency (C)', fontsize=11)
ax2.set_ylabel('Overall Throughput  (tok/s)', fontsize=11)
ax2.set_title('Throughput vs Concurrency — Impact of 3s Tool Delay\nvllm-ppd 2P2D, Llama-3.1-8B', fontsize=12.5)
ax2.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
ax2.grid(alpha=0.3, linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for c in [1, 4, 8, 12]:
    ax2.axvline(c, color='#e0e0e0', linewidth=0.8, zorder=0)

# ─── Overall title ────────────────────────────────────────────────────────────
fig.suptitle(
    'Effect of Mock Tool Call Duration (3s) on vllm-ppd 2P2D Throughput',
    fontsize=14, fontweight='bold', y=1.01
)

plt.tight_layout()
out = '/home/user/experiments/tool_delay_throughput_comparison.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='#f7f7f7')
print(f'\nSaved: {out}')
