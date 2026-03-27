import matplotlib.pyplot as plt

# -------- Global style (publication-like) --------
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.facecolor": "white",
    }
)

# -------- Data --------
methods = ["Zero-shot", "RAG", "Ours-init", "Ours-v5", "Sel-LR", "Sel-RF", "Oracle"]
acc = [0.918, 0.926, 0.906, 0.912, 0.914, 0.926, 0.950]

deploy_labels = ["Zero-shot", "RAG", "Ours-init", "Ours-v5"]
deploy_acc = [0.918, 0.926, 0.906, 0.912]
deploy_tokens = [393.188, 809.168, 2305.046, 1508.504]

selector_names = ["LogReg Selector", "RF Selector", "Oracle UB"]
selector_acc = [0.914, 0.926, 0.950]

# Scientific palette
C_BLUE = "#4C78A8"
C_GREEN = "#59A14F"
C_ORANGE = "#F28E2B"
C_RED = "#E15759"
C_PURPLE = "#B07AA1"
C_GOLD = "#EDC948"
C_GRAY = "#9D9D9D"

# -------- Figure 1: Main accuracy bars --------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
bar_colors = [C_BLUE, C_GREEN, C_RED, C_ORANGE, C_PURPLE, "#7FBC41", C_GOLD]
bars = ax.bar(methods, acc, color=bar_colors, edgecolor="black", linewidth=0.4)
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0.89, 0.955)
ax.set_ylabel("Accuracy")
ax.set_title("Main Results on 500 GSM8K Samples")
ax.grid(axis="y")
ax.axhline(0.926, color=C_GREEN, linewidth=1.2, linestyle=":", label="RAG baseline = 0.926")
ax.legend(frameon=False, loc="upper left")
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig("paper_assets/fig1_accuracy_bar.png", dpi=300)
plt.close()

# -------- Figure 2: Cost-performance scatter --------
fig, ax = plt.subplots(figsize=(6.8, 5.3))
colors = [C_BLUE, C_GREEN, C_RED, C_ORANGE]
for xi, yi, lab, c in zip(deploy_tokens, deploy_acc, deploy_labels, colors):
    ax.scatter(xi, yi, s=110, color=c, edgecolor="black", linewidth=0.5, zorder=3)
    ax.text(xi + 30, yi + 0.0008, lab, fontsize=9)
ax.set_xlabel("Average Tokens per Problem")
ax.set_ylabel("Accuracy")
ax.set_title("Cost-Performance Trade-off")
ax.grid(True)
ax.set_xlim(250, 2450)
ax.set_ylim(0.902, 0.9295)
plt.tight_layout()
plt.savefig("paper_assets/fig2_cost_performance.png", dpi=300)
plt.close()

# -------- Figure 3: Selector comparison --------
fig, ax = plt.subplots(figsize=(7.1, 4.7))
bars = ax.bar(
    selector_names,
    selector_acc,
    color=[C_PURPLE, "#7FBC41", C_GOLD],
    edgecolor="black",
    linewidth=0.4,
)
for b, v in zip(bars, selector_acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.905, 0.955)
ax.set_ylabel("Accuracy")
ax.set_title("Selector Performance (5-fold CV)")
ax.grid(axis="y")
plt.xticks(rotation=8)
plt.tight_layout()
plt.savefig("paper_assets/fig3_selector_summary.png", dpi=300)
plt.close()

# -------- Figure 4: Delta from RAG --------
fig, ax = plt.subplots(figsize=(7.8, 4.8))
compare_labels = ["Zero-shot", "Ours-init", "Ours-v5", "Sel-LR", "Sel-RF", "Oracle UB"]
compare_acc = [0.918, 0.906, 0.912, 0.914, 0.926, 0.950]
delta = [v - 0.926 for v in compare_acc]
colors = [C_BLUE if d >= 0 else C_RED for d in delta]
bars = ax.barh(compare_labels, delta, color=colors, edgecolor="black", linewidth=0.35)
for b, d in zip(bars, delta):
    ax.text(d + (0.0008 if d >= 0 else -0.0008), b.get_y() + b.get_height() / 2, f"{d:+.3f}",
            va="center", ha=("left" if d >= 0 else "right"), fontsize=9)
ax.axvline(0.0, color=C_GRAY, linewidth=1.0)
ax.set_xlabel("Accuracy Delta vs RAG")
ax.set_title("Relative Gain/Loss Compared with RAG")
ax.grid(axis="x")
plt.tight_layout()
plt.savefig("paper_assets/fig4_delta_vs_rag.png", dpi=300)
plt.close()

# -------- Figure 5: Token budget comparison --------
fig, ax = plt.subplots(figsize=(8.4, 4.9))
token_labels = ["Zero-shot", "RAG", "Ours-init", "Ours-v5"]
token_vals = [393.188, 809.168, 2305.046, 1508.504]
bars = ax.bar(token_labels, token_vals, color=[C_BLUE, C_GREEN, C_RED, C_ORANGE], edgecolor="black", linewidth=0.4)
for b, v in zip(bars, token_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 35, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("Average Tokens per Problem")
ax.set_title("Inference Token Budget by Method")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("paper_assets/fig5_token_budget.png", dpi=300)
plt.close()

print("Saved figures to paper_assets/: fig1~fig5")

# -------- Figure 6: Pareto-style line (Accuracy vs Tokens) --------
fig, ax = plt.subplots(figsize=(7.2, 5.0))
pt_labels = ["Zero-shot", "RAG", "Ours-v5", "Ours-init"]
pt_tokens = [393.188, 809.168, 1508.504, 2305.046]
pt_acc = [0.918, 0.926, 0.912, 0.906]
pt_colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]

# sort by tokens for a proper line trend
order = sorted(range(len(pt_tokens)), key=lambda i: pt_tokens[i])
x = [pt_tokens[i] for i in order]
y = [pt_acc[i] for i in order]
labs = [pt_labels[i] for i in order]
cols = [pt_colors[i] for i in order]

ax.plot(x, y, color=C_GRAY, linewidth=1.4, linestyle="-", alpha=0.8, label="trend")
for xi, yi, lab, c in zip(x, y, labs, cols):
    ax.scatter(xi, yi, s=110, color=c, edgecolor="black", linewidth=0.5, zorder=3)
    ax.text(xi + 28, yi + 0.0006, f"{lab}\n({yi:.3f})", fontsize=8.6)
ax.set_xlabel("Average Tokens per Problem")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy-Token Pareto Trend (Deployable Methods)")
ax.grid(True)
ax.set_xlim(250, 2450)
ax.set_ylim(0.902, 0.9295)
plt.tight_layout()
plt.savefig("paper_assets/fig6_pareto_line.png", dpi=300)
plt.close()

# -------- Figure 7: Relative comparison line (vs RAG baseline) --------
fig, ax = plt.subplots(figsize=(8.0, 4.8))
cmp_labels = ["Zero-shot", "Ours-init", "Ours-v5", "Sel-LR", "Sel-RF", "Oracle UB"]
cmp_vals = [0.918, 0.906, 0.912, 0.914, 0.926, 0.950]
delta = [v - 0.926 for v in cmp_vals]

ax.plot(range(len(cmp_labels)), delta, color=C_PURPLE, linewidth=2.0, marker="o", markersize=6)
for i, d in enumerate(delta):
    ax.text(i, d + (0.0012 if d >= 0 else -0.0018), f"{d:+.3f}", ha="center", fontsize=8.8)
ax.axhline(0.0, color=C_GRAY, linewidth=1.1, linestyle="--", label="RAG baseline")
ax.set_xticks(range(len(cmp_labels)))
ax.set_xticklabels(cmp_labels, rotation=12)
ax.set_ylabel("Accuracy Delta vs RAG")
ax.set_title("Relative Performance Curve Against RAG")
ax.grid(axis="y")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("paper_assets/fig7_relative_line.png", dpi=300)
plt.close()

print("Saved figures to paper_assets/: fig1~fig7")
