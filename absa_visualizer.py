"""
absa_visualizer.py
──────────────────
Generates a rich multi-panel analytics dashboard for ABSA results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from collections import Counter
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# ── Dark theme palette ────────────────────────────────────────
BG       = "#0a0e1a"
CARD     = "#111827"
CARD2    = "#1f2937"
BORDER   = "#374151"
TXT_PRI  = "#f9fafb"
TXT_SEC  = "#9ca3af"

SENT_COLORS = {
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral":  "#6b7280",
    "mixed":    "#f59e0b",
}

ASPECT_COLORS = [
    "#3b82f6","#8b5cf6","#ec4899","#f59e0b",
    "#10b981","#06b6d4","#f97316","#84cc16","#e11d48","#0ea5e9",
]


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT_SEC, labelsize=8)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color(BORDER)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, color=TXT_PRI, fontsize=9.5, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=TXT_SEC, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TXT_SEC, fontsize=8)


def build_dashboard(insights: List, output_path: str, all_analyses: List):
    """
    insights      : list of ProductInsight
    all_analyses  : flat list of ReviewAnalysis (all products combined)
    """
    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        "Product Review Insights Dashboard  ·  Aspect-Based Sentiment Analysis (ABSA)",
        color=TXT_PRI, fontsize=15, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.05, right=0.97, top=0.94, bottom=0.05
    )

    product_names = [ins.product_name for ins in insights]
    colors_p = ["#3b82f6", "#8b5cf6", "#f59e0b"]

    # ── Panel 1: Aspect Sentiment Heatmap (spans 2 cols, 2 rows) ──
    ax_heat = fig.add_subplot(gs[0:2, 0:2])
    _style(ax_heat, "Aspect Sentiment Scores by Product (−3 to +3)")

    all_aspects = sorted(set(
        a for ins in insights for a in ins.aspect_scores
    ))
    matrix = np.array([
        [ins.aspect_scores.get(a, np.nan) for a in all_aspects]
        for ins in insights
    ])

    cmap = LinearSegmentedColormap.from_list(
        "absa", ["#ef4444", "#1f2937", "#10b981"], N=256
    )
    masked = np.ma.masked_invalid(matrix)
    im = ax_heat.imshow(masked, cmap=cmap, vmin=-3, vmax=3,
                         aspect="auto", interpolation="nearest")

    ax_heat.set_yticks(range(len(insights)))
    ax_heat.set_yticklabels([ins.product_name for ins in insights],
                              color=TXT_PRI, fontsize=8.5)
    ax_heat.set_xticks(range(len(all_aspects)))
    ax_heat.set_xticklabels([a.replace("_", "\n") for a in all_aspects],
                              rotation=0, ha="center", fontsize=7.5, color=TXT_SEC)

    # Annotate cells
    for i in range(len(insights)):
        for j, aspect in enumerate(all_aspects):
            val = insights[i].aspect_scores.get(aspect, None)
            if val is not None:
                color = "white" if abs(val) > 1.0 else TXT_PRI
                ax_heat.text(j, i, f"{val:+.1f}", ha="center", va="center",
                              fontsize=8, color=color, fontweight="bold")
            else:
                ax_heat.text(j, i, "—", ha="center", va="center",
                              fontsize=9, color=BORDER)

    plt.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02,
                 label="Sentiment Score").ax.tick_params(colors=TXT_SEC, labelsize=7)

    # ── Panel 2: Overall Sentiment Bar (products) ──────────────
    ax_ov = fig.add_subplot(gs[0, 2])
    _style(ax_ov, "Overall NLP Score", ylabel="Score (−3 to +3)")
    overall_scores = [ins.overall_score for ins in insights]
    bar_cols = [SENT_COLORS["positive"] if s > 0 else SENT_COLORS["negative"]
                for s in overall_scores]
    bars = ax_ov.bar(product_names, overall_scores, color=bar_cols,
                      edgecolor=BG, linewidth=0.8, width=0.5)
    ax_ov.axhline(0, color=BORDER, linewidth=1)
    for bar, val in zip(bars, overall_scores):
        ax_ov.text(bar.get_x() + bar.get_width()/2,
                   val + (0.05 if val >= 0 else -0.12),
                   f"{val:+.2f}", ha="center", color=TXT_PRI, fontsize=8, fontweight="bold")
    ax_ov.set_xticklabels(product_names, rotation=15, ha="right", fontsize=7.5, color=TXT_SEC)
    ax_ov.set_ylim(-3, 3)

    # ── Panel 3: Avg Star Rating ───────────────────────────────
    ax_star = fig.add_subplot(gs[0, 3])
    _style(ax_star, "Avg Star Rating", ylabel="Stars")
    stars = [ins.avg_star_rating for ins in insights]
    ax_star.bar(product_names, stars, color=["#fbbf24"]*len(stars),
                 edgecolor=BG, linewidth=0.8, width=0.5)
    ax_star.set_ylim(0, 5.5)
    ax_star.axhline(4, color=BORDER, linestyle="--", linewidth=0.8, alpha=0.6)
    for i, (name, val) in enumerate(zip(product_names, stars)):
        ax_star.text(i, val + 0.1, f"★ {val:.2f}", ha="center",
                     color="#fbbf24", fontsize=9, fontweight="bold")
    ax_star.set_xticklabels(product_names, rotation=15, ha="right", fontsize=7.5, color=TXT_SEC)

    # ── Panel 4: Aspect Mention Frequency ─────────────────────
    ax_freq = fig.add_subplot(gs[1, 2:])
    _style(ax_freq, "Aspect Mention Frequency (all products)", xlabel="Mentions")

    total_mentions: Dict[str, int] = Counter()
    for ins in insights:
        for asp, cnt in ins.aspect_counts.items():
            total_mentions[asp] += cnt

    sorted_asp = sorted(total_mentions.items(), key=lambda x: x[1], reverse=True)
    asp_names  = [a.replace("_", " ").title() for a, _ in sorted_asp]
    asp_counts = [c for _, c in sorted_asp]
    bar_c      = ASPECT_COLORS[:len(asp_names)]

    bars_h = ax_freq.barh(asp_names, asp_counts, color=bar_c, edgecolor=BG, linewidth=0.8)
    for bar, val in zip(bars_h, asp_counts):
        ax_freq.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", color=TXT_PRI, fontsize=8)
    ax_freq.tick_params(axis="y", colors=TXT_PRI, labelsize=8)

    # ── Panel 5: Sentiment Distribution per Product ────────────
    ax_dist = fig.add_subplot(gs[2, 0:2])
    _style(ax_dist, "Review Sentiment Distribution per Product")

    sent_labels = ["positive", "negative", "neutral", "mixed"]
    x = np.arange(len(insights))
    width = 0.18

    for si, sentiment in enumerate(sent_labels):
        counts = []
        for ins in insights:
            analyses = [a for a in all_analyses if a.product_id == ins.product_id]
            cnt = sum(1 for a in analyses if a.sentiment_label == sentiment)
            counts.append(cnt)
        offset = (si - 1.5) * width
        bars_g = ax_dist.bar(x + offset, counts, width,
                              color=SENT_COLORS[sentiment], edgecolor=BG,
                              linewidth=0.6, label=sentiment.capitalize(), alpha=0.9)

    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(product_names, fontsize=8, color=TXT_SEC)
    ax_dist.legend(fontsize=7.5, facecolor=CARD2, edgecolor=BORDER,
                   labelcolor=TXT_PRI, loc="upper right")
    ax_dist.set_ylabel("Review Count", color=TXT_SEC, fontsize=8)

    # ── Panel 6: Star Distribution (stacked) ──────────────────
    ax_stars2 = fig.add_subplot(gs[2, 2:])
    _style(ax_stars2, "Star Rating Distribution")

    star_palette = {5:"#10b981", 4:"#84cc16", 3:"#f59e0b", 2:"#f97316", 1:"#ef4444"}
    bottom = np.zeros(len(insights))
    for star in [1, 2, 3, 4, 5]:
        vals = [ins.star_distribution.get(star, 0) for ins in insights]
        ax_stars2.bar(product_names, vals, bottom=bottom,
                      color=star_palette[star], label=f"★{star}",
                      edgecolor=BG, linewidth=0.5)
        bottom += np.array(vals)
    ax_stars2.legend(fontsize=7.5, facecolor=CARD2, edgecolor=BORDER,
                      labelcolor=TXT_PRI, loc="upper right",
                      ncol=5, bbox_to_anchor=(1, 1.12))
    ax_stars2.set_xticklabels(product_names, rotation=15, ha="right",
                               fontsize=8, color=TXT_SEC)
    ax_stars2.set_ylabel("Reviews", color=TXT_SEC, fontsize=8)

    # ── Panel 7: Top Praised & Criticized per Product ──────────
    ax_praise = fig.add_subplot(gs[3, :])
    ax_praise.set_facecolor(CARD)
    ax_praise.axis("off")
    ax_praise.set_title("Aspect Intelligence Summary  ·  Top Praised & Criticized per Product",
                         color=TXT_PRI, fontsize=9.5, fontweight="bold", loc="left", pad=10)

    cell_w = 1.0 / len(insights)
    for pi, ins in enumerate(insights):
        x_pos = pi * cell_w + 0.02

        # Product name
        ax_praise.text(x_pos, 0.92, ins.product_name, color=colors_p[pi],
                        fontsize=9.5, fontweight="bold", transform=ax_praise.transAxes)

        # Stars
        stars_str = "★" * int(round(ins.avg_star_rating)) + "☆" * (5 - int(round(ins.avg_star_rating)))
        ax_praise.text(x_pos, 0.80, f"{stars_str}  {ins.avg_star_rating:.1f}",
                        color="#fbbf24", fontsize=9, transform=ax_praise.transAxes)

        # Praised aspects
        ax_praise.text(x_pos, 0.66, "✅ Top Praised", color="#10b981",
                        fontsize=8, fontweight="bold", transform=ax_praise.transAxes)
        for ei, asp in enumerate(ins.top_praised[:3]):
            score = ins.aspect_scores.get(asp, 0)
            ax_praise.text(x_pos + 0.01, 0.55 - ei * 0.12,
                            f"  {asp.replace('_',' ').title()}: {score:+.2f}",
                            color=TXT_PRI, fontsize=7.5, transform=ax_praise.transAxes)

        # Criticized aspects
        ax_praise.text(x_pos, 0.20, "❌ Top Criticized", color="#ef4444",
                        fontsize=8, fontweight="bold", transform=ax_praise.transAxes)
        for ei, asp in enumerate(ins.top_criticized[:3]):
            score = ins.aspect_scores.get(asp, 0)
            ax_praise.text(x_pos + 0.01, 0.09 - ei * 0.09,
                            f"  {asp.replace('_',' ').title()}: {score:+.2f}",
                            color="#fca5a5", fontsize=7.5, transform=ax_praise.transAxes)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Dashboard saved → {output_path}")
