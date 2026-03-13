"""
absa_full_dashboard.py
──────────────────────
Comprehensive ABSA Python Dashboard — 5 full pages saved as PNG.

Page 1 → Executive Summary  (overview + KPIs + heatmap)
Page 2 → Product Deep-Dives (per-product aspect breakdowns, star dist)
Page 3 → Review-Level Analysis (individual reviews with aspect chips)
Page 4 → Comparative Intelligence (head-to-head aspect matrix)
Page 5 → Insights Report (winner per aspect + contrast analysis + churn signals)

Run standalone:  python absa_full_dashboard.py
Or imported via: build_full_dashboard(insights, analyses, output_dir)
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════
BG        = "#060b18"
CARD      = "#0d1526"
CARD2     = "#131f33"
BORDER    = "#1e2d47"
BORDER2   = "#253550"
TXT_PRI   = "#e8f0fe"
TXT_SEC   = "#7b90b8"
TXT_DIM   = "#3d5070"
ACCENT    = "#4f8ef7"

PROD_COLORS  = ["#4f8ef7", "#a78bfa", "#fb923c"]
PROD_GLOWS   = ["#4f8ef730", "#a78bfa30", "#fb923c30"]

SENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#f43f5e",
    "neutral":  "#64748b",
    "mixed":    "#f59e0b",
}

ASPECT_PALETTE = [
    "#4f8ef7","#a78bfa","#fb923c","#22c55e",
    "#f59e0b","#06b6d4","#ec4899","#84cc16",
    "#f43f5e","#8b5cf6",
]

ASPECT_ICONS = {
    "camera":            "📷",
    "display":           "🖥️",
    "battery":           "🔋",
    "performance":       "⚡",
    "build_quality":     "🏗️",
    "software":          "💿",
    "audio":             "🔊",
    "price_value":       "💰",
    "connectivity":      "📡",
    "delivery_packaging":"📦",
}

STAR_PALETTE = {5:"#22c55e", 4:"#86efac", 3:"#f59e0b", 2:"#fb923c", 1:"#f43f5e"}


def _score_color(s: float) -> str:
    if s >= 2.0:  return "#22c55e"
    if s >= 0.5:  return "#4ade80"
    if s >= -0.5: return "#64748b"
    if s >= -2.0: return "#fb923c"
    return "#f43f5e"


def _setup_fig(figsize=(24, 16), title_text=""):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    if title_text:
        fig.text(0.5, 0.985, title_text, ha="center", va="top",
                 color=TXT_PRI, fontsize=14, fontweight="bold",
                 fontfamily="monospace")
    return fig


def _ax_style(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT_SEC, labelsize=8.5, length=3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(BORDER2)
    if grid:
        ax.grid(color=BORDER, linestyle="--", linewidth=0.45, alpha=0.8)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TXT_PRI, fontsize=9.5,
                     fontweight="bold", pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=TXT_SEC, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=TXT_SEC, fontsize=8.5)


def _card_bg(ax, alpha=1.0):
    """Draw a card background behind an axes."""
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER2)
        spine.set_linewidth(0.8)


def _draw_kpi_box(ax, x, y, w, h, value, label, sublabel="",
                  color=ACCENT, bg=CARD2):
    """Draw a KPI card inside an axes using relative coordinates."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.01",
                          facecolor=bg, edgecolor=color+"55",
                          linewidth=1.2, transform=ax.transAxes,
                          zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h * 0.72, str(value),
            ha="center", va="center", color=color,
            fontsize=18, fontweight="black",
            fontfamily="monospace", transform=ax.transAxes, zorder=4)
    ax.text(x + w/2, y + h * 0.35, label,
            ha="center", va="center", color=TXT_SEC,
            fontsize=7.5, fontweight="bold",
            transform=ax.transAxes, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h * 0.12, sublabel,
                ha="center", va="center", color=TXT_DIM,
                fontsize=6.5, transform=ax.transAxes, zorder=4)


def _bidirectional_bar(ax, y_positions, values, labels,
                       colors, height=0.55, max_abs=5.0):
    """Horizontal bidirectional bar chart (positive right, negative left)."""
    for yi, (val, label, col) in enumerate(zip(values, labels, colors)):
        norm = val / max_abs
        if val >= 0:
            ax.barh(yi, val, height=height, left=0,
                    color=col, edgecolor=BG, linewidth=0.5)
        else:
            ax.barh(yi, abs(val), height=height, left=val,
                    color=col, edgecolor=BG, linewidth=0.5)
        offset = 0.08 if val >= 0 else -0.08
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, yi, f"{val:+.2f}",
                va="center", ha=ha, color=TXT_PRI,
                fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=TXT_PRI, fontsize=8.5)
    ax.axvline(0, color=BORDER2, linewidth=1.2)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1  —  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
def page_executive_summary(insights, all_analyses, output_path):
    fig = _setup_fig((26, 18),
        "AspectIQ  ·  Product Review Insights Dashboard  ·  Executive Summary")

    gs = gridspec.GridSpec(
        4, 6, figure=fig,
        hspace=0.55, wspace=0.45,
        left=0.04, right=0.98, top=0.94, bottom=0.05
    )

    product_names = [ins.product_name for ins in insights]
    short_names   = [n.split()[0] + " " + n.split()[1] for n in product_names]

    # ── Row 0: KPI strip ────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(BG)
    ax_kpi.axis("off")

    # Global stats
    total_reviews   = sum(i.total_reviews for i in insights)
    total_aspects   = sum(len(i.aspect_scores) for i in insights)
    avg_score       = np.mean([i.overall_score for i in insights])
    contrast_count  = sum(1 for a in all_analyses if a.contrast_detected)
    mixed_count     = sum(1 for a in all_analyses if a.sentiment_label == "mixed")

    kpis = [
        (f"{total_reviews}", "TOTAL REVIEWS", "across all products"),
        (f"{len(insights)}", "PRODUCTS ANALYSED", "in this report"),
        (f"{total_aspects}", "ASPECT DIMENSIONS", "detected total"),
        (f"{contrast_count}", "CONTRAST REVIEWS", "e.g. 'good X but bad Y'"),
        (f"{mixed_count}", "MIXED SENTIMENT", "reviews with both +/−"),
        (f"{avg_score:+.2f}", "AVG NLP SCORE", "−3 (worst) to +3 (best)"),
    ]

    box_w, gap = 0.148, 0.012
    for ki, (val, lbl, sub) in enumerate(kpis):
        col = PROD_COLORS[ki % len(PROD_COLORS)] if ki < 3 else ACCENT
        if lbl == "AVG NLP SCORE":
            col = _score_color(avg_score)
        _draw_kpi_box(ax_kpi, ki*(box_w+gap), 0.05,
                      box_w, 0.88, val, lbl, sub, color=col)

    # ── Row 1-2: Heatmap ────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1:3, 0:4])
    _ax_style(ax_heat, "Aspect Sentiment Heatmap  ·  Score per Product per Aspect",
              grid=False)

    all_aspects = sorted(set(
        a for ins in insights for a in ins.aspect_scores
    ))
    matrix = np.array([
        [ins.aspect_scores.get(a, np.nan) for a in all_aspects]
        for ins in insights
    ])

    cmap = LinearSegmentedColormap.from_list(
        "absa_heat", ["#f43f5e", "#1e1f3a", "#22c55e"], N=512
    )
    masked = np.ma.masked_invalid(matrix)
    im = ax_heat.imshow(masked, cmap=cmap, vmin=-3, vmax=3,
                        aspect="auto", interpolation="nearest")

    # Product labels (y)
    for yi, (ins, col) in enumerate(zip(insights, PROD_COLORS)):
        ax_heat.text(-0.5, yi, ins.product_name,
                     ha="right", va="center", color=col,
                     fontsize=9, fontweight="bold")

    # Aspect labels (x)
    ax_heat.set_xticks(range(len(all_aspects)))
    ax_heat.set_xticklabels(
        [f"{ASPECT_ICONS.get(a,'·')} {a.replace('_',' ')}" for a in all_aspects],
        rotation=30, ha="right", fontsize=8.5, color=TXT_SEC
    )
    ax_heat.set_yticks([])

    # Cell annotations
    for i in range(len(insights)):
        for j, aspect in enumerate(all_aspects):
            val = insights[i].aspect_scores.get(aspect)
            if val is not None:
                txt_col = "#ffffff" if abs(val) > 1.2 else TXT_PRI
                ax_heat.text(j, i, f"{val:+.1f}", ha="center", va="center",
                             fontsize=9.5, color=txt_col, fontweight="bold")
            else:
                ax_heat.text(j, i, "—", ha="center", va="center",
                             fontsize=10, color=TXT_DIM)

    # Grid lines between cells
    for x in np.arange(-0.5, len(all_aspects), 1):
        ax_heat.axvline(x, color=BG, linewidth=1.5)
    for y in np.arange(-0.5, len(insights), 1):
        ax_heat.axhline(y, color=BG, linewidth=1.5)

    cb = plt.colorbar(im, ax=ax_heat, fraction=0.012, pad=0.01)
    cb.ax.tick_params(colors=TXT_SEC, labelsize=7.5)
    cb.set_label("Sentiment Score", color=TXT_SEC, fontsize=8)

    # ── Overall NLP Score (col 4) ────────────────────────────────
    ax_ov = fig.add_subplot(gs[1, 4:])
    _ax_style(ax_ov, "Overall NLP Score per Product", ylabel="Score")
    overall_scores = [ins.overall_score for ins in insights]
    bar_cols = [_score_color(s) for s in overall_scores]
    bars = ax_ov.bar(short_names, overall_scores, color=bar_cols,
                     edgecolor=BG, linewidth=1, width=0.55, zorder=3)
    ax_ov.axhline(0, color=BORDER2, linewidth=1.5)
    ax_ov.set_ylim(-1.5, 3.5)
    for bar, val, col in zip(bars, overall_scores, PROD_COLORS):
        offset = 0.08 if val >= 0 else -0.18
        ax_ov.text(bar.get_x() + bar.get_width()/2, val + offset,
                   f"{val:+.2f}", ha="center", color=col,
                   fontsize=10, fontweight="black")
    ax_ov.set_xticklabels(short_names, color=TXT_SEC, fontsize=8, rotation=10)

    # ── Avg Star Rating ──────────────────────────────────────────
    ax_star = fig.add_subplot(gs[2, 4:])
    _ax_style(ax_star, "Average Star Rating", ylabel="Stars")
    star_vals = [ins.avg_star_rating for ins in insights]
    sbars = ax_star.bar(short_names, star_vals, color="#fbbf24",
                        edgecolor=BG, linewidth=1, width=0.55, zorder=3)
    ax_star.set_ylim(0, 5.8)
    ax_star.axhline(4.0, color=BORDER2, linestyle="--", linewidth=1, alpha=0.7)
    ax_star.text(len(insights)-0.3, 4.1, "4★ threshold",
                 color=TXT_DIM, fontsize=7, ha="right")
    for bar, val in zip(sbars, star_vals):
        ax_star.text(bar.get_x() + bar.get_width()/2, val + 0.12,
                     f"★ {val:.2f}", ha="center", color="#fbbf24",
                     fontsize=10, fontweight="black")
    ax_star.set_xticklabels(short_names, color=TXT_SEC, fontsize=8, rotation=10)

    # ── Row 3: Aspect Mention Frequency + Sentiment distribution ─
    ax_freq = fig.add_subplot(gs[3, 0:3])
    _ax_style(ax_freq, "Aspect Mention Frequency  ·  All Products Combined",
              xlabel="Total Mentions")

    total_mentions: Dict[str, int] = Counter()
    for ins in insights:
        for asp, cnt in ins.aspect_counts.items():
            total_mentions[asp] += cnt

    sorted_asp   = sorted(total_mentions.items(), key=lambda x: x[1], reverse=True)
    asp_labels   = [f"{ASPECT_ICONS.get(a,'·')} {a.replace('_',' ').title()}"
                    for a, _ in sorted_asp]
    asp_counts   = [c for _, c in sorted_asp]
    bar_c        = ASPECT_PALETTE[:len(asp_labels)]
    bh = ax_freq.barh(asp_labels, asp_counts, color=bar_c,
                      edgecolor=BG, linewidth=0.7, zorder=3)
    for bar, val in zip(bh, asp_counts):
        ax_freq.text(val + 0.4, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", color=TXT_PRI, fontsize=8.5,
                     fontweight="bold")
    ax_freq.tick_params(axis="y", colors=TXT_PRI, labelsize=8.5)
    ax_freq.invert_yaxis()

    # ── Sentiment distribution ────────────────────────────────────
    ax_sent = fig.add_subplot(gs[3, 3:])
    _ax_style(ax_sent, "Review Sentiment Label Distribution",
              ylabel="Review Count")

    sent_labels_list = ["positive", "negative", "neutral", "mixed"]
    x_pos = np.arange(len(insights))
    w     = 0.18
    for si, snt in enumerate(sent_labels_list):
        counts = []
        for ins in insights:
            prod_analyses = [a for a in all_analyses if a.product_id == ins.product_id]
            counts.append(sum(1 for a in prod_analyses if a.sentiment_label == snt))
        offset = (si - 1.5) * w
        ax_sent.bar(x_pos + offset, counts, w,
                    color=SENT_COLORS[snt], edgecolor=BG,
                    linewidth=0.6, label=snt.capitalize(),
                    alpha=0.92, zorder=3)

    ax_sent.set_xticks(x_pos)
    ax_sent.set_xticklabels(short_names, color=TXT_SEC, fontsize=8.5)
    legend = ax_sent.legend(fontsize=8, facecolor=CARD2, edgecolor=BORDER2,
                             labelcolor=TXT_PRI, loc="upper right")

    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Page 1 saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2  —  PRODUCT DEEP-DIVES
# ══════════════════════════════════════════════════════════════════════════
def page_product_deepdive(insights, all_analyses, output_path):
    n = len(insights)
    fig = _setup_fig((26, 20),
        "AspectIQ  ·  Product Deep-Dive  ·  Aspect Scores · Stars · Top Phrases")

    # 3 columns, one per product; 5 rows
    gs = gridspec.GridSpec(
        5, n, figure=fig,
        hspace=0.6, wspace=0.42,
        left=0.05, right=0.97, top=0.93, bottom=0.04
    )

    for pi, (ins, prod_col, prod_glow) in enumerate(
            zip(insights, PROD_COLORS, PROD_GLOWS)):
        prod_analyses = [a for a in all_analyses if a.product_id == ins.product_id]
        short = ins.product_name

        # ── Row 0: Product Header (info card) ───────────────────
        ax_hdr = fig.add_subplot(gs[0, pi])
        ax_hdr.set_facecolor(CARD)
        ax_hdr.axis("off")
        for spine in ax_hdr.spines.values():
            spine.set_edgecolor(prod_col + "66")
        # Product title
        ax_hdr.text(0.5, 0.88, ins.product_name,
                    ha="center", va="center", color=prod_col,
                    fontsize=13, fontweight="black",
                    transform=ax_hdr.transAxes)
        # NLP score
        score_col = _score_color(ins.overall_score)
        ax_hdr.text(0.25, 0.50, f"{ins.overall_score:+.2f}",
                    ha="center", va="center", color=score_col,
                    fontsize=22, fontweight="black",
                    fontfamily="monospace", transform=ax_hdr.transAxes)
        ax_hdr.text(0.25, 0.22, "NLP Score",
                    ha="center", va="center", color=TXT_SEC,
                    fontsize=7, transform=ax_hdr.transAxes)
        # Star
        ax_hdr.text(0.75, 0.50, f"★{ins.avg_star_rating:.1f}",
                    ha="center", va="center", color="#fbbf24",
                    fontsize=22, fontweight="black",
                    transform=ax_hdr.transAxes)
        ax_hdr.text(0.75, 0.22, f"{ins.total_reviews} reviews",
                    ha="center", va="center", color=TXT_SEC,
                    fontsize=7, transform=ax_hdr.transAxes)
        # Sentiment badge
        snt_col = SENT_COLORS[ins.overall_sentiment]
        ax_hdr.text(0.5, 0.06, f"● {ins.overall_sentiment.upper()}",
                    ha="center", va="center", color=snt_col,
                    fontsize=8, fontweight="bold",
                    transform=ax_hdr.transAxes)

        # ── Row 1: Aspect Score Bidirectional Bar ─────────────────
        ax_bar = fig.add_subplot(gs[1, pi])
        _ax_style(ax_bar, f"Aspect Scores", xlabel="Score (−3 to +3)")

        sorted_asp = sorted(ins.aspect_scores.items(), key=lambda x: x[1], reverse=True)
        asp_names  = [f"{ASPECT_ICONS.get(a,'·')} {a.replace('_',' ')}"
                      for a, _ in sorted_asp]
        asp_vals   = [v for _, v in sorted_asp]
        asp_colors = [_score_color(v) for v in asp_vals]

        _bidirectional_bar(ax_bar, range(len(asp_names)), asp_vals,
                           asp_names, asp_colors, max_abs=5.0)
        ax_bar.set_xlim(-4, 6)
        ax_bar.invert_yaxis()

        # ── Row 2: Star Distribution Donut ───────────────────────
        ax_donut = fig.add_subplot(gs[2, pi])
        ax_donut.set_facecolor(CARD)
        ax_donut.axis("off")
        ax_donut.set_title("Star Distribution", color=TXT_PRI,
                            fontsize=9.5, fontweight="bold", pad=10)

        star_vals_d = [ins.star_distribution.get(s, 0) for s in [5,4,3,2,1]]
        star_cols   = [STAR_PALETTE[s] for s in [5,4,3,2,1]]
        star_labels = [f"★{s}" for s in [5,4,3,2,1]]

        # Filter zeros
        nz = [(v, c, l) for v, c, l in zip(star_vals_d, star_cols, star_labels) if v > 0]
        if nz:
            vals_nz, cols_nz, lbls_nz = zip(*nz)
            wedges, _, autotexts = ax_donut.pie(
                vals_nz, colors=cols_nz, startangle=90,
                wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=2),
                autopct=lambda p: f"{p:.0f}%" if p > 8 else "",
                pctdistance=0.78
            )
            for at in autotexts:
                at.set(color=BG, fontsize=8, fontweight="bold")
            ax_donut.legend(wedges, lbls_nz, loc="lower center",
                            bbox_to_anchor=(0.5, -0.12), ncol=5,
                            fontsize=7.5, facecolor=CARD2,
                            edgecolor=BORDER2, labelcolor=TXT_PRI)
            # Centre text
            ax_donut.text(0, 0.1, f"★{ins.avg_star_rating:.1f}",
                          ha="center", va="center", color="#fbbf24",
                          fontsize=18, fontweight="black")
            ax_donut.text(0, -0.25, f"{ins.total_reviews} reviews",
                          ha="center", va="center", color=TXT_SEC, fontsize=8)

        # ── Row 3: Aspect mention count bar ───────────────────────
        ax_cnt = fig.add_subplot(gs[3, pi])
        _ax_style(ax_cnt, "Aspect Mention Count", xlabel="Mentions")

        sorted_cnt = sorted(ins.aspect_counts.items(), key=lambda x: x[1], reverse=True)
        cnt_labels = [f"{ASPECT_ICONS.get(a,'·')} {a.replace('_',' ')}"
                      for a, _ in sorted_cnt]
        cnt_vals   = [v for _, v in sorted_cnt]
        cnt_colors = [ASPECT_PALETTE[i % len(ASPECT_PALETTE)] for i in range(len(cnt_labels))]

        bh2 = ax_cnt.barh(cnt_labels, cnt_vals, color=cnt_colors,
                           edgecolor=BG, linewidth=0.6, zorder=3)
        for bar, val in zip(bh2, cnt_vals):
            ax_cnt.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                        str(val), va="center", color=TXT_PRI, fontsize=8.5)
        ax_cnt.tick_params(axis="y", colors=TXT_PRI, labelsize=8)
        ax_cnt.invert_yaxis()

        # ── Row 4: Praised / Criticized Summary ───────────────────
        ax_sum = fig.add_subplot(gs[4, pi])
        ax_sum.set_facecolor(CARD)
        ax_sum.axis("off")
        ax_sum.set_title("Praise & Criticism Summary", color=TXT_PRI,
                          fontsize=9.5, fontweight="bold", pad=10)

        # Praised
        ax_sum.text(0.02, 0.94, "✅  TOP PRAISED", color="#22c55e",
                    fontsize=8.5, fontweight="bold",
                    transform=ax_sum.transAxes)
        for ei, asp in enumerate(ins.top_praised[:4]):
            score = ins.aspect_scores.get(asp, 0)
            col   = _score_color(score)
            ax_sum.text(0.04, 0.80 - ei * 0.16,
                        f"{ASPECT_ICONS.get(asp,'·')} {asp.replace('_',' ').title()}: {score:+.2f}",
                        color=TXT_PRI, fontsize=8.5,
                        transform=ax_sum.transAxes)

        # Criticized
        ax_sum.text(0.02, 0.38, "❌  TOP CRITICIZED", color="#f43f5e",
                    fontsize=8.5, fontweight="bold",
                    transform=ax_sum.transAxes)
        for ei, asp in enumerate(ins.top_criticized[:3]):
            score = ins.aspect_scores.get(asp, 0)
            ax_sum.text(0.04, 0.24 - ei * 0.15,
                        f"{ASPECT_ICONS.get(asp,'·')} {asp.replace('_',' ').title()}: {score:+.2f}",
                        color="#fca5a5", fontsize=8.5,
                        transform=ax_sum.transAxes)

        if not ins.top_criticized:
            ax_sum.text(0.04, 0.24, "None detected", color=TXT_DIM,
                        fontsize=8, transform=ax_sum.transAxes)

    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Page 2 saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3  —  REVIEW-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
def page_review_analysis(insights, all_analyses, reviews_raw, output_path):
    """
    reviews_raw : list of RawReview objects (from review_data.py)
    """
    fig = _setup_fig((26, 22),
        "AspectIQ  ·  Review-Level Analysis  ·  Aspect Detection per Review")

    n_reviews = min(12, len(all_analyses))
    n_cols    = 3
    n_rows    = (n_reviews + n_cols - 1) // n_cols + 1

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.04, right=0.98, top=0.94, bottom=0.03
    )

    # ── Row 0: Review Aspect Heatmap (all reviews × aspects) ────
    ax_all = fig.add_subplot(gs[0, :])
    _ax_style(ax_all,
              "Per-Review Aspect Score Matrix  ·  All Reviews",
              grid=False)

    aspect_order = ["camera","display","battery","performance",
                    "build_quality","software","audio","price_value",
                    "connectivity","delivery_packaging"]

    # Build matrix [reviews × aspects]
    rev_ids = [a.review_id for a in all_analyses]
    mat = np.full((len(all_analyses), len(aspect_order)), np.nan)
    for ri, analysis in enumerate(all_analyses):
        for asp_obj in analysis.aspects:
            if asp_obj.aspect in aspect_order:
                ji = aspect_order.index(asp_obj.aspect)
                # Take the max-abs score for the aspect in this review
                cur = mat[ri, ji]
                if np.isnan(cur) or abs(asp_obj.score) > abs(cur):
                    mat[ri, ji] = asp_obj.score

    cmap2 = LinearSegmentedColormap.from_list(
        "rev_heat", ["#f43f5e", "#0d1526", "#22c55e"], N=512
    )
    masked2 = np.ma.masked_invalid(mat)
    im2 = ax_all.imshow(masked2.T, cmap=cmap2, vmin=-3, vmax=3,
                         aspect="auto", interpolation="nearest")

    ax_all.set_xticks(range(len(all_analyses)))
    # Label with review id + star
    xlabels = []
    for a in all_analyses:
        pid_short = a.product_id
        xlabels.append(f"{a.review_id}\n{'★'*a.star_rating if a.star_rating else ''}")
    ax_all.set_xticklabels(xlabels, rotation=0, ha="center",
                            fontsize=6.5, color=TXT_SEC)

    ax_all.set_yticks(range(len(aspect_order)))
    ax_all.set_yticklabels(
        [f"{ASPECT_ICONS.get(a,'·')} {a.replace('_',' ')}" for a in aspect_order],
        color=TXT_PRI, fontsize=8.5
    )

    # Grid
    for x in np.arange(-0.5, len(all_analyses), 1):
        ax_all.axvline(x, color=BG, linewidth=1)
    for y in np.arange(-0.5, len(aspect_order), 1):
        ax_all.axhline(y, color=BG, linewidth=1)

    # Colorbar
    cb = plt.colorbar(im2, ax=ax_all, fraction=0.008, pad=0.01, orientation="vertical")
    cb.ax.tick_params(colors=TXT_SEC, labelsize=7)
    cb.set_label("Score", color=TXT_SEC, fontsize=8)

    # Product separator lines
    product_ids = [a.product_id for a in all_analyses]
    for ri in range(1, len(all_analyses)):
        if product_ids[ri] != product_ids[ri-1]:
            ax_all.axvline(ri - 0.5, color="#fbbf24", linewidth=2, alpha=0.5)

    # ── Individual Review Cards ──────────────────────────────────
    for idx, analysis in enumerate(all_analyses[:n_reviews]):
        row = (idx // n_cols) + 1
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])
        _ax_style(ax, grid=False)
        ax.set_facecolor(CARD)
        ax.axis("off")

        # Find matching raw review
        raw = next((r for r in reviews_raw if r.review_id == analysis.review_id), None)
        prod_ins = next((i for i in insights if i.product_id == analysis.product_id), None)
        prod_col = PROD_COLORS[list({i.product_id for i in insights}).index(analysis.product_id)] if prod_ins else ACCENT

        # Header
        prod_name = prod_ins.product_name if prod_ins else analysis.product_id
        stars_str = "★" * (analysis.star_rating or 0)
        snt_col   = SENT_COLORS.get(analysis.sentiment_label, TXT_SEC)

        ax.text(0.02, 0.96, f"{analysis.review_id}  ·  {prod_name}",
                color=prod_col, fontsize=8.5, fontweight="bold",
                transform=ax.transAxes, va="top")
        ax.text(0.98, 0.96, f"{stars_str}  {analysis.overall_score:+.2f}",
                color="#fbbf24", fontsize=8, fontweight="bold",
                transform=ax.transAxes, va="top", ha="right")
        ax.text(0.02, 0.86,
                f"● {analysis.sentiment_label.upper()}"
                + ("  🔀 CONTRAST" if analysis.contrast_detected else ""),
                color=snt_col, fontsize=7.5, fontweight="bold",
                transform=ax.transAxes, va="top")

        # Review title
        if raw:
            ax.text(0.02, 0.76, f'"{raw.title}"',
                    color=TXT_PRI, fontsize=8, style="italic",
                    transform=ax.transAxes, va="top",
                    wrap=True)

        # Aspect rows
        unique_aspects = {}
        for asp_obj in analysis.aspects:
            if asp_obj.aspect not in unique_aspects:
                unique_aspects[asp_obj.aspect] = asp_obj

        y_start = 0.60
        dy      = min(0.13, 0.55 / max(len(unique_aspects), 1))

        for ai, (asp_name, asp_obj) in enumerate(list(unique_aspects.items())[:5]):
            y = y_start - ai * dy
            icon  = ASPECT_ICONS.get(asp_name, "·")
            col   = _score_color(asp_obj.score)
            words = ", ".join(asp_obj.opinion_words[:2]) if asp_obj.opinion_words else "—"

            # Mini bar
            bar_max = 0.45
            bar_len = min(abs(asp_obj.score) / 3 * bar_max, bar_max)
            bar_x   = 0.55 if asp_obj.score >= 0 else 0.55 - bar_len
            rect = FancyBboxPatch(
                (bar_x, y - 0.025), bar_len, 0.048,
                boxstyle="round,pad=0.002",
                facecolor=col + "33", edgecolor=col + "66",
                linewidth=0.6, transform=ax.transAxes, zorder=2
            )
            ax.add_patch(rect)
            ax.plot([0.55, 0.55], [y - 0.03, y + 0.03],
                    color=BORDER2, linewidth=0.8,
                    transform=ax.transAxes, zorder=1)

            ax.text(0.02, y, f"{icon} {asp_name.replace('_',' '):<16}",
                    color=TXT_PRI, fontsize=7.5,
                    transform=ax.transAxes, va="center")
            ax.text(0.53, y, f"{asp_obj.score:+.1f}",
                    color=col, fontsize=8, fontweight="bold",
                    transform=ax.transAxes, va="center", ha="right")
            ax.text(0.55 + bar_max + 0.02, y, f"  {words}",
                    color=TXT_DIM, fontsize=6.5,
                    transform=ax.transAxes, va="center")

        # Border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(BORDER2)
            spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Page 3 saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4  —  COMPARATIVE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════
def page_comparative(insights, all_analyses, output_path):
    fig = _setup_fig((26, 20),
        "AspectIQ  ·  Comparative Intelligence  ·  Head-to-Head Aspect Analysis")

    all_aspects = sorted(set(
        a for ins in insights for a in ins.aspect_scores
    ))
    n_asp    = len(all_aspects)
    n_prods  = len(insights)
    short    = [ins.product_name.split()[0]+" "+ins.product_name.split()[1]
                for ins in insights]

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.55, wspace=0.42,
        left=0.04, right=0.97, top=0.93, bottom=0.05
    )

    # ── Panel 1: Grouped Bar Chart per Aspect ────────────────────
    ax_grp = fig.add_subplot(gs[0:2, :])
    _ax_style(ax_grp,
              "Aspect Score Comparison  ·  All Products Side-by-Side",
              ylabel="Sentiment Score")

    x        = np.arange(n_asp)
    width    = 0.25
    offsets  = np.linspace(-(n_prods-1)/2 * width, (n_prods-1)/2 * width, n_prods)

    for pi, (ins, col) in enumerate(zip(insights, PROD_COLORS)):
        vals = [ins.aspect_scores.get(a, 0) for a in all_aspects]
        bars = ax_grp.bar(x + offsets[pi], vals, width,
                          color=col, edgecolor=BG, linewidth=0.6,
                          label=ins.product_name, alpha=0.92, zorder=3)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.3:
                y_offset = 0.08 if val >= 0 else -0.22
                ax_grp.text(bar.get_x() + bar.get_width()/2,
                            val + y_offset, f"{val:+.1f}",
                            ha="center", color=col, fontsize=7,
                            fontweight="bold")

    ax_grp.axhline(0, color=BORDER2, linewidth=1.5)
    ax_grp.set_xticks(x)
    ax_grp.set_xticklabels(
        [f"{ASPECT_ICONS.get(a,'·')} {a.replace('_',' ')}" for a in all_aspects],
        rotation=25, ha="right", fontsize=9, color=TXT_SEC
    )
    ax_grp.set_ylim(-3.5, 6.5)
    ax_grp.legend(fontsize=9, facecolor=CARD2, edgecolor=BORDER2,
                  labelcolor=TXT_PRI, loc="upper right",
                  framealpha=0.9)

    # ── Panel 2: Radar / Spider Chart ────────────────────────────
    ax_rad = fig.add_subplot(gs[2, 0], polar=True)
    ax_rad.set_facecolor(CARD)

    # Use top 8 aspects for radar
    radar_aspects = all_aspects[:8]
    n_rad = len(radar_aspects)
    angles = np.linspace(0, 2*np.pi, n_rad, endpoint=False).tolist()
    angles += angles[:1]

    ax_rad.set_facecolor(CARD)
    ax_rad.spines["polar"].set_color(BORDER2)
    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(
        [a.replace("_", "\n") for a in radar_aspects],
        size=7.5, color=TXT_SEC
    )
    ax_rad.tick_params(colors=TXT_SEC)
    ax_rad.set_ylim(-3, 5)
    ax_rad.yaxis.set_tick_params(labelcolor=TXT_DIM, labelsize=7)
    ax_rad.grid(color=BORDER, alpha=0.5)

    for ins, col in zip(insights, PROD_COLORS):
        vals_r = [ins.aspect_scores.get(a, 0) for a in radar_aspects]
        vals_r += vals_r[:1]
        ax_rad.plot(angles, vals_r, color=col, linewidth=2.0, zorder=3)
        ax_rad.fill(angles, vals_r, color=col, alpha=0.12)

    ax_rad.set_title("Aspect Radar", color=TXT_PRI, fontsize=9.5,
                      fontweight="bold", pad=18)

    # Legend for radar
    legend_patches = [mpatches.Patch(color=c, label=ins.product_name)
                      for ins, c in zip(insights, PROD_COLORS)]
    ax_rad.legend(handles=legend_patches, loc="lower right",
                  bbox_to_anchor=(1.45, -0.18), fontsize=8,
                  facecolor=CARD2, edgecolor=BORDER2,
                  labelcolor=TXT_PRI)

    # ── Panel 3: Winner per Aspect ────────────────────────────────
    ax_win = fig.add_subplot(gs[2, 1])
    _ax_style(ax_win, "Winner per Aspect", grid=False)
    ax_win.set_facecolor(CARD)
    ax_win.axis("off")

    ax_win.text(0.5, 0.97, "🏆  Best Product per Aspect",
                ha="center", va="top", color=TXT_PRI,
                fontsize=9.5, fontweight="bold",
                transform=ax_win.transAxes)

    y_pos = 0.88
    for asp in all_aspects:
        entries = [(ins, ins.aspect_scores.get(asp))
                   for ins in insights if asp in ins.aspect_scores]
        if not entries:
            continue
        winner, best_score = max(entries, key=lambda x: x[1])
        wi = insights.index(winner)
        col = PROD_COLORS[wi]
        icon = ASPECT_ICONS.get(asp, "·")

        ax_win.text(0.03, y_pos,
                    f"{icon} {asp.replace('_',' '):<20}",
                    color=TXT_SEC, fontsize=8,
                    transform=ax_win.transAxes, va="center")
        ax_win.text(0.58, y_pos,
                    f"{winner.product_name.split()[0]}",
                    color=col, fontsize=8, fontweight="bold",
                    transform=ax_win.transAxes, va="center")
        ax_win.text(0.92, y_pos, f"{best_score:+.2f}",
                    color=_score_color(best_score), fontsize=8,
                    fontweight="black", fontfamily="monospace",
                    transform=ax_win.transAxes, va="center", ha="right")
        y_pos -= 0.088

    # ── Panel 4: Scatter – Overall Score vs Avg Stars ─────────────
    ax_sc = fig.add_subplot(gs[2, 2])
    _ax_style(ax_sc, "NLP Score vs Star Rating",
              xlabel="Avg Star Rating", ylabel="NLP Overall Score")

    for ins, col in zip(insights, PROD_COLORS):
        ax_sc.scatter(ins.avg_star_rating, ins.overall_score,
                      s=220, color=col, edgecolors=BG,
                      linewidth=1.5, zorder=5)
        ax_sc.annotate(ins.product_name.split()[0],
                       (ins.avg_star_rating, ins.overall_score),
                       xytext=(8, 8), textcoords="offset points",
                       color=col, fontsize=8.5, fontweight="bold")

    ax_sc.axhline(0, color=BORDER2, linewidth=1)
    ax_sc.axvline(3.5, color=BORDER2, linewidth=1, linestyle="--", alpha=0.6)
    ax_sc.set_xlim(1, 6)
    ax_sc.set_ylim(-1, 3.5)

    # ── Panel 5: Aspect Gap Chart (best - worst gap) ──────────────
    ax_gap = fig.add_subplot(gs[3, :])
    _ax_style(ax_gap,
              "Aspect Score Gap  ·  Difference Between Best and Worst Product",
              xlabel="Aspect", ylabel="Score Gap (Best − Worst)")

    gaps = []
    for asp in all_aspects:
        vals_asp = [(ins.aspect_scores[asp], ins) for ins in insights
                    if asp in ins.aspect_scores]
        if len(vals_asp) < 2:
            gaps.append((asp, 0, None, None))
            continue
        best = max(vals_asp, key=lambda x: x[0])
        worst = min(vals_asp, key=lambda x: x[0])
        gaps.append((asp, best[0] - worst[0], best[1], worst[1]))

    gaps.sort(key=lambda x: x[1], reverse=True)
    gap_labels = [f"{ASPECT_ICONS.get(g[0],'·')} {g[0].replace('_',' ')}"
                  for g in gaps]
    gap_vals   = [g[1] for g in gaps]
    gap_best   = [PROD_COLORS[insights.index(g[2])] if g[2] else TXT_DIM for g in gaps]

    gbars = ax_gap.bar(gap_labels, gap_vals, color=gap_best,
                       edgecolor=BG, linewidth=0.7, zorder=3, alpha=0.9)
    for bar, val, g in zip(gbars, gap_vals, gaps):
        if val > 0.5:
            best_name  = g[2].product_name.split()[0] if g[2] else ""
            worst_name = g[3].product_name.split()[0] if g[3] else ""
            ax_gap.text(bar.get_x() + bar.get_width()/2,
                        val + 0.1, f"↑{best_name}\n↓{worst_name}",
                        ha="center", va="bottom", color=TXT_SEC,
                        fontsize=6.5)
    ax_gap.set_xticks(range(len(gap_labels)))
    ax_gap.set_xticklabels(gap_labels, rotation=20, ha="right",
                            fontsize=9, color=TXT_SEC)
    ax_gap.set_ylim(0, max(gap_vals) * 1.35 if gap_vals else 5)

    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Page 4 saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5  —  INSIGHTS REPORT
# ══════════════════════════════════════════════════════════════════════════
def page_insights_report(insights, all_analyses, reviews_raw, output_path):
    from datetime import datetime

    fig = _setup_fig((26, 20),
        "AspectIQ  ·  Insights Report  ·  Contrast Analysis · Confidence · Opinion Words")

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.55, wspace=0.42,
        left=0.04, right=0.97, top=0.93, bottom=0.04
    )

    # ── Panel 1: Confidence Distribution per Aspect ───────────────
    ax_conf = fig.add_subplot(gs[0, 0:2])
    _ax_style(ax_conf,
              "Average Aspect Detection Confidence per Product",
              ylabel="Confidence Score (0–1)")

    all_asp_names = sorted(set(
        asp_obj.aspect for a in all_analyses for asp_obj in a.aspects
    ))

    x_conf   = np.arange(len(all_asp_names))
    w_conf   = 0.25
    offsets  = np.linspace(-(len(insights)-1)/2*w_conf,
                            (len(insights)-1)/2*w_conf, len(insights))

    for pi, (ins, col) in enumerate(zip(insights, PROD_COLORS)):
        prod_analyses = [a for a in all_analyses if a.product_id == ins.product_id]
        conf_vals = []
        for asp in all_asp_names:
            scores = [
                asp_obj.confidence
                for a in prod_analyses
                for asp_obj in a.aspects
                if asp_obj.aspect == asp
            ]
            conf_vals.append(np.mean(scores) if scores else 0)

        ax_conf.bar(x_conf + offsets[pi], conf_vals, w_conf,
                    color=col, edgecolor=BG, linewidth=0.6,
                    label=ins.product_name, alpha=0.9, zorder=3)

    ax_conf.axhline(0.5, color=BORDER2, linewidth=1,
                    linestyle="--", alpha=0.7)
    ax_conf.text(len(all_asp_names)-0.3, 0.52, "0.5 threshold",
                 color=TXT_DIM, fontsize=7.5, ha="right")
    ax_conf.set_xticks(x_conf)
    ax_conf.set_xticklabels(
        [f"{ASPECT_ICONS.get(a,'·')} {a.replace('_',' ')}" for a in all_asp_names],
        rotation=25, ha="right", fontsize=8.5, color=TXT_SEC
    )
    ax_conf.set_ylim(0, 1.15)
    ax_conf.legend(fontsize=8.5, facecolor=CARD2, edgecolor=BORDER2,
                   labelcolor=TXT_PRI, loc="upper left")

    # ── Panel 2: Contrast Analysis ────────────────────────────────
    ax_ctr = fig.add_subplot(gs[0, 2])
    _ax_style(ax_ctr, "Contrast Review Analysis",
              ylabel="Count")

    contrast_by_prod = {ins.product_id: 0 for ins in insights}
    non_contrast     = {ins.product_id: 0 for ins in insights}
    for a in all_analyses:
        if a.product_id in contrast_by_prod:
            if a.contrast_detected:
                contrast_by_prod[a.product_id] += 1
            else:
                non_contrast[a.product_id] += 1

    x_ctr = np.arange(len(insights))
    w_ctr = 0.35
    ax_ctr.bar(x_ctr - w_ctr/2,
               [contrast_by_prod[i.product_id] for i in insights],
               w_ctr, color="#f59e0b", edgecolor=BG,
               label="Contrast detected", zorder=3)
    ax_ctr.bar(x_ctr + w_ctr/2,
               [non_contrast[i.product_id] for i in insights],
               w_ctr, color=BORDER2, edgecolor=BG,
               label="No contrast", zorder=3)
    ax_ctr.set_xticks(x_ctr)
    ax_ctr.set_xticklabels(
        [i.product_name.split()[0] for i in insights],
        color=TXT_SEC, fontsize=9
    )
    ax_ctr.legend(fontsize=8, facecolor=CARD2, edgecolor=BORDER2,
                  labelcolor=TXT_PRI)

    # ── Panel 3: Top Opinion Words per Aspect ─────────────────────
    ax_ops = fig.add_subplot(gs[1, 0:2])
    _ax_style(ax_ops, "Top Opinion Words  ·  Most Frequent Across All Reviews",
              xlabel="Frequency")

    word_freq: Dict[str, int] = Counter()
    word_sent: Dict[str, str] = {}
    for a in all_analyses:
        for asp_obj in a.aspects:
            for w in asp_obj.opinion_words:
                word_freq[w] += 1
                if w not in word_sent:
                    word_sent[w] = asp_obj.sentiment

    top_words = word_freq.most_common(18)
    wlabels   = [w for w, _ in top_words]
    wcounts   = [c for _, c in top_words]
    wcolors   = [SENT_COLORS.get(word_sent.get(w, "neutral"), TXT_SEC)
                 for w in wlabels]

    bh_op = ax_ops.barh(wlabels, wcounts, color=wcolors,
                         edgecolor=BG, linewidth=0.6, zorder=3)
    for bar, val in zip(bh_op, wcounts):
        ax_ops.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", color=TXT_PRI, fontsize=8.5)
    ax_ops.tick_params(axis="y", colors=TXT_PRI, labelsize=9)
    ax_ops.invert_yaxis()

    # ── Panel 4: Score vs Word Count scatter ──────────────────────
    ax_wc = fig.add_subplot(gs[1, 2])
    _ax_style(ax_wc, "Review Length vs NLP Score",
              xlabel="Word Count", ylabel="Overall NLP Score")

    for a in all_analyses:
        pi = next((i for i, ins in enumerate(insights)
                   if ins.product_id == a.product_id), 0)
        col = PROD_COLORS[pi]
        ax_wc.scatter(a.word_count, a.overall_score,
                      s=70, color=col, alpha=0.8, edgecolors=BG,
                      linewidth=0.8, zorder=3)

    ax_wc.axhline(0, color=BORDER2, linewidth=1)
    legend_patches = [mpatches.Patch(color=c, label=ins.product_name.split()[0])
                      for ins, c in zip(insights, PROD_COLORS)]
    ax_wc.legend(handles=legend_patches, fontsize=8,
                 facecolor=CARD2, edgecolor=BORDER2, labelcolor=TXT_PRI)

    # ── Panel 5: Full Comparative Text Table ──────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.set_facecolor(CARD)
    ax_tbl.axis("off")
    ax_tbl.set_title(
        "Comprehensive Aspect-Sentiment Summary Table",
        color=TXT_PRI, fontsize=10, fontweight="bold",
        loc="left", pad=12
    )

    all_asp_sorted = sorted(set(
        a for ins in insights for a in ins.aspect_scores
    ))

    # Headers
    col_w   = 0.90 / (len(insights) + 1)
    headers = ["Aspect"] + [ins.product_name for ins in insights]
    for hi, hdr in enumerate(headers):
        col_c = PROD_COLORS[hi-1] if hi > 0 else TXT_PRI
        ax_tbl.text(0.02 + hi * col_w, 0.95, hdr,
                    color=col_c, fontsize=8.5, fontweight="bold",
                    transform=ax_tbl.transAxes, va="top")

    ax_tbl.plot([0.01, 0.99], [0.90, 0.90], color=BORDER2, linewidth=0.8,
               transform=ax_tbl.transAxes)

    row_h = 0.85 / len(all_asp_sorted)
    for ri, asp in enumerate(all_asp_sorted):
        y = 0.87 - ri * row_h
        icon = ASPECT_ICONS.get(asp, "·")
        ax_tbl.text(0.02, y,
                    f"{icon} {asp.replace('_', ' ').title()}",
                    color=TXT_SEC, fontsize=8.5,
                    transform=ax_tbl.transAxes, va="center")

        for pi, ins in enumerate(insights):
            x = 0.02 + (pi + 1) * col_w
            score = ins.aspect_scores.get(asp)
            count = ins.aspect_counts.get(asp, 0)
            if score is not None:
                col  = _score_color(score)
                snt  = ins.aspect_sentiments.get(asp, "neutral")
                sico = "✅" if snt == "positive" else ("❌" if snt == "negative" else "➖")
                ax_tbl.text(x, y,
                            f"{sico} {score:+.2f}  ({count}×)",
                            color=col, fontsize=8,
                            fontweight="bold", fontfamily="monospace",
                            transform=ax_tbl.transAxes, va="center")
            else:
                ax_tbl.text(x, y, "—", color=TXT_DIM, fontsize=9,
                            transform=ax_tbl.transAxes, va="center")

        if ri % 2 == 0:
            rect = FancyBboxPatch(
                (0.005, y - row_h*0.45), 0.99, row_h * 0.9,
                boxstyle="round,pad=0.001",
                facecolor="rgba(255,255,255,0.015)" if False else "#ffffff08",
                edgecolor="none",
                transform=ax_tbl.transAxes, zorder=0
            )
            ax_tbl.add_patch(rect)

    # Timestamp
    fig.text(0.98, 0.01,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  AspectIQ ABSA Engine",
             ha="right", va="bottom", color=TXT_DIM, fontsize=7)

    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Page 5 saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def build_full_dashboard(insights, all_analyses, reviews_raw, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    pages = [
        ("page1_executive_summary.png",   page_executive_summary),
        ("page2_product_deepdive.png",     page_product_deepdive),
        ("page3_review_analysis.png",      page_review_analysis),
        ("page4_comparative.png",          page_comparative),
        ("page5_insights_report.png",      page_insights_report),
    ]

    print(f"\n  📊  Building 5-page ABSA dashboard...")
    for fname, fn in pages:
        path = os.path.join(output_dir, fname)
        if fn in (page_review_analysis, page_insights_report):
            fn(insights, all_analyses, reviews_raw, path)
        else:
            fn(insights, all_analyses, path)

    print(f"\n  ✓  All 5 pages saved to {output_dir}")
    return [os.path.join(output_dir, f) for f, _ in pages]


# ──────────────────────────────────────────────────────────────
# Standalone run
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from absa_engine import analyze_review, aggregate_product_insights
    from review_data import REVIEWS, PRODUCTS

    print("  Running ABSA analysis...")
    analyses = []
    for rev in REVIEWS:
        result = analyze_review(
            text=rev.body, review_id=rev.review_id,
            product_id=rev.product_id, star_rating=rev.star_rating
        )
        analyses.append(result)

    insights = []
    for pid, pname in PRODUCTS.items():
        prod_analyses = [a for a in analyses if a.product_id == pid]
        ins = aggregate_product_insights(prod_analyses, pid, pname)
        insights.append(ins)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    build_full_dashboard(insights, analyses, REVIEWS, out_dir)
