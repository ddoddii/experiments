"""
TP=4 vs 2P2D — Tool Delay 3s impact by model size (8B vs 27B)

Scaling model  (8B → 27B):
  TP=4  (BF16, no quant):  per-turn compute ×3.375 (model size ratio)
                           wall_time_27B = wall_time_8B × 3.375
  2P2D  (fp8  quant):      compute ×3.375, fp8 speedup ~1.5×
                           → net per-turn ×2.25, plus ~15% penalty
                           for larger KV spilling 1GB P-buffer
                           → effective wall_time × 2.6

3s delay estimation (same as before):
  new_wall = wall_0s + n_success × tc_turns × D / C
  tput_3s  = total_tokens / new_wall

Measured data (Llama-3.1-8B, A6000×4):
  TP=4 C=1: 34.1 tok/s,  wall=280s,  tok=9557
  TP=4 C=8: 132.9 tok/s, wall=198s,  tok=26351
  2P2D C=1: 11.0 tok/s,  wall=732s,  tok=8075,  success=200/200
  2P2D C=8: 63.7 tok/s,  wall=148s,  tok=9424,  success=200/200
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Parameters ──────────────────────────────────────────────────────────────
N       = 200     # benchmark items
TC_TURNS = 3.0   # avg tool-call turns per item
D       = 3.0    # tool delay (seconds)

# ─── Measured (8B, 0s delay) ──────────────────────────────────────────────────
#                  tput   wall    tokens  n_ok
tp4_8b_c1_raw  = (34.1,  280,   9557,   200)
tp4_8b_c8_raw  = (132.9, 198,  26351,   200)
ppd_8b_c1_raw  = (11.0,  732,   8075,   200)
ppd_8b_c8_raw  = (63.7,  148,   9424,   200)

def est_3s(wall_0, tokens, n_ok, C):
    """Estimate throughput with 3s tool delay."""
    added   = n_ok * TC_TURNS * D / C if C > 1 else N * TC_TURNS * D
    new_wall = wall_0 + added
    return tokens / new_wall

# ─── 8B estimates (3s delay) ──────────────────────────────────────────────────
tp4_8b_c1_3s = est_3s(280,  9557,  200, 1)   # 4.6
tp4_8b_c8_3s = est_3s(198, 26351,  200, 8)   # 62.3
ppd_8b_c1_3s = est_3s(732,  8075,  200, 1)   # 3.2
ppd_8b_c8_3s = est_3s(148,  9424,  200, 8)   # 25.3

# ─── 27B estimates ─────────────────────────────────────────────────────────────
# Scaling factors:
#   TP=4 BF16:  per-turn ×3.375  → wall ×3.375 → tput / 3.375
#   2P2D fp8:   per-turn ×2.25 + buffer constraint ~15% extra failures
#               conservative: wall ×2.6, ~80% success rate (160/200)

SCALE_TP4 = 3.375
SCALE_PPD = 2.6          # includes fp8 speedup + buffer constraint penalty

# Tokens: assume same benchmark → similar output token counts
# (BFCL structured JSON output is model-size-agnostic to first order)
TP4_27B_TOKENS_C8 = 26351   # same benchmark
PPD_27B_TOKENS_C8 = int(9424 * 0.80)  # ~80% success rate → fewer tokens
PPD_27B_TOKENS_C1 = 8075    # C=1 sequential: all succeed (no concurrent buffer pressure)

tp4_27b_c1 = {'wall': 280*SCALE_TP4,  'tok': 9557,               'n_ok': 200, 'C': 1}
tp4_27b_c8 = {'wall': 198*SCALE_TP4,  'tok': TP4_27B_TOKENS_C8,  'n_ok': 200, 'C': 8}
ppd_27b_c1 = {'wall': 732*SCALE_PPD,  'tok': PPD_27B_TOKENS_C1,  'n_ok': 200, 'C': 1}
ppd_27b_c8 = {'wall': 148*SCALE_PPD,  'tok': PPD_27B_TOKENS_C8,  'n_ok': 160, 'C': 8}

def tput(d):   return d['tok'] / d['wall']
def tput3(d):  return est_3s(d['wall'], d['tok'], d['n_ok'], d['C'])

# ─── Final numbers ─────────────────────────────────────────────────────────────
results = {
    # label:  (tput_0s,            tput_3s,             measured?)
    'TP=4\n8B\n(measured)':  (tp4_8b_c8_raw[0], tp4_8b_c8_3s, True),
    'TP=4\n27B\n(est.)':     (tput(tp4_27b_c8),  tput3(tp4_27b_c8), False),
    '2P2D\n8B\n(measured)':  (ppd_8b_c8_raw[0], ppd_8b_c8_3s, True),
    '2P2D\n27B\n(est.)':     (tput(ppd_27b_c8),  tput3(ppd_27b_c8), False),
}

# Retention rate
retention = {k: v[1]/v[0] for k, v in results.items()}

# C=1 sequential (for insight panel)
c1_results = {
    'TP=4 8B':   (tp4_8b_c1_raw[0], tp4_8b_c1_3s),
    'TP=4 27B':  (tput(tp4_27b_c1),  tput3(tp4_27b_c1)),
    '2P2D 8B':   (ppd_8b_c1_raw[0], ppd_8b_c1_3s),
    '2P2D 27B':  (tput(ppd_27b_c1),  tput3(ppd_27b_c1)),
}

print("=== C=8 Throughput estimates ===")
for label, (t0, t3, meas) in results.items():
    name = label.replace('\n', ' ')
    print(f"  {name:25s}  0s={t0:6.1f}  3s={t3:6.1f}  retention={t3/t0*100:.1f}%  {'(measured)' if meas else '(estimated)'}")

print("\n=== C=1 Throughput estimates ===")
for name, (t0, t3) in c1_results.items():
    print(f"  {name:15s}  0s={t0:6.1f}  3s={t3:6.1f}  retention={t3/t0*100:.1f}%")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6.5),
                         gridspec_kw={'width_ratios': [2.8, 2.8, 1.8]})
fig.patch.set_facecolor('#f5f5f5')

labels  = list(results.keys())
t0_vals = [v[0] for v in results.values()]
t3_vals = [v[1] for v in results.values()]
measured_flags = [v[2] for v in results.values()]

x = np.arange(len(labels))
w = 0.38

# ── Color scheme ──────────────────────────────────────────────────────────────
# TP=4: blue family;  2P2D: teal/green family
# solid = 0s delay;  hatched = 3s delay
BLUE_DARK  = '#1a5fa8'   # TP=4 8B
BLUE_LIGHT = '#6aaed6'   # TP=4 27B
TEAL_DARK  = '#1d7a5f'   # 2P2D 8B
TEAL_LIGHT = '#72c5a4'   # 2P2D 27B
colors_0s = [BLUE_DARK, BLUE_LIGHT, TEAL_DARK, TEAL_LIGHT]

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Absolute throughput (C=8)
# ═══════════════════════════════════════════════════════════════════════════════
ax1 = axes[0]
ax1.set_facecolor('#fefefe')

for i, (t0, t3, col) in enumerate(zip(t0_vals, t3_vals, colors_0s)):
    # 0s bar (solid)
    b0 = ax1.bar(x[i] - w/2, t0, w, color=col, alpha=0.90,
                 edgecolor='white', linewidth=1.0)
    # 3s bar (hatched, same color family, lighter)
    b3 = ax1.bar(x[i] + w/2, t3, w, color=col, alpha=0.45,
                 edgecolor=col, linewidth=1.5, hatch='///')

    # Value labels
    ax1.text(x[i]-w/2, t0+1.5, f'{t0:.1f}', ha='center', fontsize=8.5,
             fontweight='bold', color=col)
    ax1.text(x[i]+w/2, t3+1.5, f'{t3:.1f}', ha='center', fontsize=8.5,
             fontweight='bold', color=col, alpha=0.8)

    # Drop arrow + percentage
    drop = (1 - t3/t0)*100
    ax1.annotate('', xy=(x[i]+w/2, t3+1), xytext=(x[i]-w/2, t0-1),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.4,
                                 connectionstyle='arc3,rad=0.35'))
    ax1.text(x[i]+0.06, (t0+t3)/2 + 5,
             f'-{drop:.0f}%', fontsize=8.5, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                       edgecolor='red', alpha=0.82))

# Legend patches
p_solid  = mpatches.Patch(facecolor='gray', alpha=0.85, label='TOOL_DELAY = 0s')
p_hatch  = mpatches.Patch(facecolor='gray', alpha=0.40, hatch='///',
                           edgecolor='gray', label='TOOL_DELAY = 3s (est.)')
p_meas   = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=1.5,
                           label='* = measured baseline')
ax1.legend(handles=[p_solid, p_hatch, p_meas], fontsize=8.5, loc='upper right',
           framealpha=0.9)

# Model size separator
ax1.axvline(1.5, color='#bbb', linestyle='--', linewidth=1.0)
ax1.text(0.5, 138, 'TP=4 (4-GPU single instance)',
         ha='center', fontsize=8, color='#1a5fa8', style='italic')
ax1.text(2.5, 138, '2P2D (disaggregated)',
         ha='center', fontsize=8, color='#1d7a5f', style='italic')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel('Overall Throughput  (tok/s)', fontsize=11)
ax1.set_title('C=8 Concurrent — Absolute Throughput\n(0s vs 3s tool delay)', fontsize=11.5)
ax1.set_ylim(0, 150)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Throughput retained under 3s delay (%)
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]
ax2.set_facecolor('#fefefe')

ret_vals = [t3/t0*100 for t0, t3 in zip(t0_vals, t3_vals)]

bars = ax2.bar(x, ret_vals, 0.55, color=colors_0s, alpha=0.88,
               edgecolor='white', linewidth=1.0)

for bar, ret in zip(bars, ret_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.0,
             f'{ret:.1f}%', ha='center', fontsize=10, fontweight='bold',
             color=bar.get_facecolor())

# Reference lines
ax2.axhline(50, color='#999', linestyle=':', linewidth=1.2)
ax2.text(3.65, 51, '50%', fontsize=8, color='#888')

# Bracket showing 8B vs 27B improvement
for i, (name, (j_8b, j_27b)) in enumerate(
        [('TP=4', (0, 1)), ('2P2D', (2, 3))]):
    delta = ret_vals[j_27b] - ret_vals[j_8b]
    y_max = max(ret_vals[j_8b], ret_vals[j_27b]) + 4
    ax2.annotate(f'+{delta:.0f}pp\nwith 27B',
                 xy=(j_27b, ret_vals[j_27b]+2), xytext=(j_27b-0.5, y_max+8),
                 fontsize=8.5, color='#333', ha='center',
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('Throughput retained  (%)', fontsize=11)
ax2.set_title('Throughput Retention Under 3s Delay\n(larger model = less sensitive to delay)', fontsize=11.5)
ax2.set_ylim(0, 105)
ax2.axvline(1.5, color='#bbb', linestyle='--', linewidth=1.0)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Insight: why larger model is less sensitive
# ═══════════════════════════════════════════════════════════════════════════════
ax3 = axes[2]
ax3.set_facecolor('#fefefe')
ax3.axis('off')

# Per-item compute time diagram
items = [
    ('TP=4\n8B',   198*8/200,  BLUE_DARK),
    ('TP=4\n27B',  198*SCALE_TP4*8/200, BLUE_LIGHT),
    ('2P2D\n8B',   148*8/200,  TEAL_DARK),
    ('2P2D\n27B',  148*SCALE_PPD*8/200, TEAL_LIGHT),
]
delay_bar = 9.0  # 3 turns × 3s

ax3.set_xlim(0, 1)
ax3.set_ylim(-0.5, len(items)+1.5)
ax3.set_title('Per-item time at C=8\nvs 3s delay budget (9s)', fontsize=10.5)

for i, (name, t_compute, col) in enumerate(items):
    y = len(items) - i - 0.5
    total = t_compute + delay_bar
    w_compute = t_compute / (total + 5)
    w_delay   = delay_bar / (total + 5)

    # Compute bar
    ax3.barh(y, w_compute, left=0.02, height=0.5,
             color=col, alpha=0.88)
    # Delay bar
    ax3.barh(y, w_delay, left=0.02+w_compute, height=0.5,
             color='#ddd', alpha=0.8, hatch='....',
             edgecolor='#aaa')

    # Labels
    ax3.text(0.0, y, name, ha='right', va='center', fontsize=8, color=col)
    ax3.text(0.02+w_compute/2, y, f'{t_compute:.1f}s',
             ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
    retention_val = t_compute / total * 100
    ax3.text(0.02+w_compute+w_delay+0.01, y,
             f'{retention_val:.0f}%\nretained',
             ha='left', va='center', fontsize=7.5, color=col)

# Legend for horizontal bars
ax3.barh(-0.1, 0.12, left=0.02, height=0.3, color='gray', alpha=0.7)
ax3.text(0.15, -0.1, 'compute', va='center', fontsize=7.5, color='#444')
ax3.barh(-0.1, 0.12, left=0.35, height=0.3, color='#ddd', hatch='....',
         edgecolor='#aaa', alpha=0.8)
ax3.text(0.48, -0.1, 'tool delay', va='center', fontsize=7.5, color='#666')

# Key formula box
insight = (
    "Retention =  compute_time\n"
    "           ────────────────\n"
    "           compute + delay\n\n"
    "Larger model → longer compute\n"
    "→ delay is proportionally smaller\n"
    "→ less throughput loss\n\n"
    "27B compute ~2-3× longer than 8B\n"
    "→ delay (9s) becomes relatively\n"
    "  less dominant"
)
ax3.text(0.5, -0.45, insight, transform=ax3.transAxes,
         ha='center', va='bottom', fontsize=8.0,
         family='monospace',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffde7',
                   edgecolor='#d4a017', alpha=0.92, linewidth=1.5))

# ─── Overall figure ──────────────────────────────────────────────────────────
fig.suptitle(
    'Throughput: TP=4 vs 2P2D (8B Measured / 27B Estimated)  —  TOOL_DELAY = 0s vs 3s',
    fontsize=13.5, fontweight='bold', y=1.01
)

# Bottom note
fig.text(0.5, -0.03,
    'Estimates assume: TP=4 27B scales as 8B × 3.375 (model size ratio);  '
    '2P2D 27B scales as 8B × 2.6 (fp8 speedup 1.5×, + buffer constraint penalty).  '
    'Delay model: new_wall = wall_0s + n_ok × 3turns × 3s / C',
    ha='center', fontsize=7.5, color='#666', style='italic')

plt.tight_layout()
out = '/home/user/experiments/model_delay_comparison.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='#f5f5f5')
print(f'\nSaved: {out}')
