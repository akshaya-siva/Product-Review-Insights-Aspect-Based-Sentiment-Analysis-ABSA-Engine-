"""
main.py
───────
End-to-end ABSA Pipeline Runner.
Run: python main.py
"""

import os
import sys
import json
import csv
from datetime import datetime
from collections import defaultdict

try:
    from colorama import Fore, Back, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class _S:
        def __getattr__(self, _): return ""
    Fore = Back = Style = _S()

sys.path.insert(0, os.path.dirname(__file__))
from absa_engine import analyze_review, aggregate_product_insights, ReviewAnalysis, ProductInsight
from review_data import REVIEWS, PRODUCTS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SENT_COLOR = {
    "positive": Fore.GREEN,
    "negative": Fore.RED,
    "neutral":  Fore.WHITE,
    "mixed":    Fore.YELLOW,
}
SENT_ICON = {
    "positive": "😊 POSITIVE",
    "negative": "😠 NEGATIVE",
    "neutral":  "😐 NEUTRAL",
    "mixed":    "🔀 MIXED",
}

def _bar(val, lo=-3, hi=3, width=24):
    """Bidirectional bar centered at 0."""
    mid = width // 2
    if val >= 0:
        filled = int(round((val / hi) * mid))
        return " " * mid + Fore.GREEN + "█" * filled + "░" * (mid - filled) + Style.RESET_ALL
    else:
        filled = int(round((abs(val) / abs(lo)) * mid))
        return Fore.RED + "░" * (mid - filled) + "█" * filled + Style.RESET_ALL + " " * mid

def _score_bar(val, width=20):
    """Simple 0-based bar for display."""
    norm = (val + 3) / 6  # normalize -3..+3 to 0..1
    filled = int(round(norm * width))
    col = Fore.GREEN if val > 0 else (Fore.RED if val < 0 else Fore.WHITE)
    return col + "█" * filled + "░" * (width - filled) + Style.RESET_ALL


# ═══════════════════════════════════════════════════════
# STEP 1 — ANALYSE ALL REVIEWS
# ═══════════════════════════════════════════════════════
def run_analysis_pipeline(reviews):
    print(f"\n{Fore.CYAN}{'═'*80}")
    print(f"  🔬  ABSA PIPELINE  —  {len(reviews)} reviews across {len(PRODUCTS)} products")
    print(f"{'═'*80}{Style.RESET_ALL}\n")

    analyses = []
    for rev in reviews:
        result = analyze_review(
            text=rev.body,
            review_id=rev.review_id,
            product_id=rev.product_id,
            star_rating=rev.star_rating,
        )
        analyses.append(result)
        col = SENT_COLOR.get(result.sentiment_label, "")
        icon = SENT_ICON.get(result.sentiment_label, "")
        aspects_str = ", ".join([
            f"{a.aspect}({'+' if a.score >= 0 else ''}{a.score:.1f})"
            for a in result.aspects[:4]
        ])
        print(
            f"  {Fore.WHITE}{rev.review_id:<5}{Style.RESET_ALL}  "
            f"{Fore.BLUE}[{rev.product_id}]{Style.RESET_ALL}  "
            f"★{rev.star_rating}  "
            f"{col}{icon:<18}{Style.RESET_ALL}  "
            f"overall={col}{result.overall_score:+.2f}{Style.RESET_ALL}  "
            f"{Fore.CYAN}{aspects_str}{Style.RESET_ALL}"
        )

    return analyses


# ═══════════════════════════════════════════════════════
# STEP 2 — DETAILED REVIEW BREAKDOWN
# ═══════════════════════════════════════════════════════
def print_review_detail(analyses, max_reviews=6):
    print(f"\n{Fore.CYAN}{'═'*80}")
    print("  📄  DETAILED ASPECT ANALYSIS  (first 6 reviews)")
    print(f"{'═'*80}{Style.RESET_ALL}\n")

    for analysis in analyses[:max_reviews]:
        rev = next(r for r in REVIEWS if r.review_id == analysis.review_id)
        col = SENT_COLOR.get(analysis.sentiment_label, "")

        print(f"  {Fore.WHITE}{'─'*78}{Style.RESET_ALL}")
        print(
            f"  {Fore.WHITE}{analysis.review_id}{Style.RESET_ALL}  "
            f"{Fore.BLUE}{PRODUCTS[analysis.product_id]}{Style.RESET_ALL}  "
            f"★{analysis.star_rating}  "
            f"{col}{SENT_ICON[analysis.sentiment_label]}{Style.RESET_ALL}"
            f"{'  🔀 CONTRAST DETECTED' if analysis.contrast_detected else ''}"
        )
        print(f"  {Fore.WHITE}'{rev.title}'{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Review:{Style.RESET_ALL} {Fore.WHITE}{analysis.raw_text[:160]}{'...' if len(analysis.raw_text)>160 else ''}{Style.RESET_ALL}")
        print()

        if not analysis.aspects:
            print(f"    {Fore.WHITE}No aspects detected.{Style.RESET_ALL}\n")
            continue

        print(f"  {Fore.WHITE}Detected Aspects:{Style.RESET_ALL}")
        for asp in analysis.aspects:
            c = Fore.GREEN if asp.sentiment == "positive" else (Fore.RED if asp.sentiment == "negative" else Fore.WHITE)
            icon = "✅" if asp.sentiment == "positive" else ("❌" if asp.sentiment == "negative" else "➖")
            opinions = ", ".join([f'"{w}"' for w in asp.opinion_words[:4]]) if asp.opinion_words else "no opinion words"
            print(
                f"    {icon}  {Fore.WHITE}{asp.aspect:<18}{Style.RESET_ALL}  "
                f"{c}{asp.score:+.2f}{Style.RESET_ALL}  "
                f"{_score_bar(asp.score, 16)}  "
                f"conf={asp.confidence:.2f}  "
                f"→ {Fore.CYAN}{opinions}{Style.RESET_ALL}"
            )
            print(f"         {Fore.WHITE}evidence:{Style.RESET_ALL} \"{asp.sentence[:90]}{'...' if len(asp.sentence)>90 else ''}\"")
        print(f"\n  {Fore.WHITE}Overall Score:{Style.RESET_ALL}  {col}{analysis.overall_score:+.3f}{Style.RESET_ALL}  {_score_bar(analysis.overall_score)}")
        print()


# ═══════════════════════════════════════════════════════
# STEP 3 — PRODUCT INSIGHTS
# ═══════════════════════════════════════════════════════
def print_product_insights(insights):
    print(f"\n{Fore.CYAN}{'═'*80}")
    print("  🏭  PRODUCT-LEVEL INSIGHTS")
    print(f"{'═'*80}{Style.RESET_ALL}\n")

    for ins in insights:
        col = SENT_COLOR.get(ins.overall_sentiment, "")
        print(f"  {Fore.WHITE}{'━'*78}{Style.RESET_ALL}")
        print(
            f"  {Fore.BLUE}{ins.product_name}{Style.RESET_ALL}  "
            f"[{ins.product_id}]  "
            f"{col}●  {ins.overall_sentiment.upper()}{Style.RESET_ALL}  "
            f"overall={col}{ins.overall_score:+.3f}{Style.RESET_ALL}  "
            f"★ avg={Fore.YELLOW}{ins.avg_star_rating:.2f}{Style.RESET_ALL}  "
            f"reviews={ins.total_reviews}"
        )
        print()

        # Aspect scores table
        print(f"  {Fore.WHITE}Aspect Breakdown:{Style.RESET_ALL}")
        sorted_aspects = sorted(ins.aspect_scores.items(), key=lambda x: x[1], reverse=True)
        for aspect, score in sorted_aspects:
            count = ins.aspect_counts.get(aspect, 0)
            sentiment = ins.aspect_sentiments.get(aspect, "neutral")
            c = Fore.GREEN if score > 0 else (Fore.RED if score < 0 else Fore.WHITE)
            icon = "✅" if score > 0.3 else ("❌" if score < -0.3 else "➖")
            print(
                f"    {icon}  {Fore.WHITE}{aspect.replace('_',' '):<20}{Style.RESET_ALL}  "
                f"{c}{score:+.3f}{Style.RESET_ALL}  "
                f"{_score_bar(score, 18)}  "
                f"mentions: {count}"
            )
        print()

        # Top praised/criticized
        if ins.top_praised:
            print(f"  {Fore.GREEN}Top Praised:    {', '.join(ins.top_praised)}{Style.RESET_ALL}")
        if ins.top_criticized:
            print(f"  {Fore.RED}Top Criticized: {', '.join(ins.top_criticized)}{Style.RESET_ALL}")

        # Star distribution
        star_str = "  ".join([
            f"{'★'*k}: {v}" for k, v in sorted(ins.star_distribution.items(), reverse=True) if v > 0
        ])
        print(f"  {Fore.YELLOW}Stars: {star_str}{Style.RESET_ALL}")
        print()


# ═══════════════════════════════════════════════════════
# STEP 4 — COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════
def print_comparative(insights):
    print(f"\n{Fore.CYAN}{'═'*80}")
    print("  📊  COMPARATIVE ASPECT MATRIX")
    print(f"{'═'*80}{Style.RESET_ALL}\n")

    all_aspects = sorted(set(
        a for ins in insights for a in ins.aspect_scores
    ))

    # Header
    header = f"  {'Aspect':<22}"
    for ins in insights:
        header += f"  {ins.product_name[:18]:<18}"
    print(Fore.WHITE + header + Style.RESET_ALL)
    print(f"  {'─'*78}")

    for aspect in all_aspects:
        row = f"  {aspect.replace('_',' '):<22}"
        for ins in insights:
            score = ins.aspect_scores.get(aspect, None)
            if score is None:
                row += f"  {'—':<18}"
            else:
                c = Fore.GREEN if score > 0.3 else (Fore.RED if score < -0.3 else Fore.WHITE)
                row += f"  {c}{score:+.3f}{Style.RESET_ALL}{'        ':8}"
        print(row)

    print()

    # Winner per aspect
    print(f"  {Fore.WHITE}Best per Aspect:{Style.RESET_ALL}")
    for aspect in all_aspects:
        scores = [(ins.product_name, ins.aspect_scores.get(aspect, -999)) for ins in insights
                  if aspect in ins.aspect_scores]
        if scores:
            best = max(scores, key=lambda x: x[1])
            print(f"    {aspect.replace('_',' '):<22} → {Fore.GREEN}{best[0]}{Style.RESET_ALL} ({best[1]:+.3f})")
    print()


# ═══════════════════════════════════════════════════════
# STEP 5 — EXPORTS
# ═══════════════════════════════════════════════════════
def export_csv(analyses, path):
    rows = []
    for a in analyses:
        for asp in a.aspects:
            rows.append({
                "review_id":     a.review_id,
                "product_id":    a.product_id,
                "product_name":  PRODUCTS.get(a.product_id, a.product_id),
                "star_rating":   a.star_rating,
                "overall_score": round(a.overall_score, 3),
                "overall_sentiment": a.sentiment_label,
                "contrast":      a.contrast_detected,
                "aspect":        asp.aspect,
                "aspect_score":  round(asp.score, 3),
                "aspect_sentiment": asp.sentiment,
                "confidence":    round(asp.confidence, 3),
                "opinion_words": "|".join(asp.opinion_words),
                "evidence":      asp.sentence[:200],
            })
    with open(path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  ✓  CSV exported  → {path}")


def export_json(insights, analyses, path):
    out = {
        "generated_at": datetime.now().isoformat(),
        "products":     [],
        "review_analyses": []
    }
    for ins in insights:
        out["products"].append({
            "product_id":    ins.product_id,
            "product_name":  ins.product_name,
            "total_reviews": ins.total_reviews,
            "overall_score": ins.overall_score,
            "avg_star_rating": ins.avg_star_rating,
            "overall_sentiment": ins.overall_sentiment,
            "aspect_scores":    ins.aspect_scores,
            "aspect_counts":    ins.aspect_counts,
            "aspect_sentiments": ins.aspect_sentiments,
            "top_praised":      ins.top_praised,
            "top_criticized":   ins.top_criticized,
            "star_distribution": ins.star_distribution,
        })
    for a in analyses:
        out["review_analyses"].append({
            "review_id":     a.review_id,
            "product_id":    a.product_id,
            "star_rating":   a.star_rating,
            "overall_score": a.overall_score,
            "sentiment":     a.sentiment_label,
            "contrast":      a.contrast_detected,
            "aspects": [
                {"aspect": asp.aspect, "sentiment": asp.sentiment,
                 "score": asp.score, "opinions": asp.opinion_words,
                 "evidence": asp.sentence}
                for asp in a.aspects
            ]
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  ✓  JSON exported → {path}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    start = datetime.now()

    print(f"\n{Fore.CYAN}{'█'*80}")
    print(f"  AspectIQ  ·  Aspect-Based Sentiment Analysis (ABSA)")
    print(f"  E-Commerce Product Review Intelligence  ·  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*80}{Style.RESET_ALL}")

    # Step 1: Analyse
    analyses = run_analysis_pipeline(REVIEWS)

    # Step 2: Detailed breakdown
    print_review_detail(analyses, max_reviews=6)

    # Step 3: Aggregate per product
    insights = []
    for pid, pname in PRODUCTS.items():
        product_analyses = [a for a in analyses if a.product_id == pid]
        ins = aggregate_product_insights(product_analyses, pid, pname)
        insights.append(ins)

    print_product_insights(insights)

    # Step 4: Comparative
    print_comparative(insights)

    # Step 5: Exports
    print(f"\n{Fore.CYAN}{'═'*80}")
    print("  💾  EXPORTING")
    print(f"{'═'*80}{Style.RESET_ALL}\n")

    export_csv(analyses, os.path.join(OUTPUT_DIR, "absa_results.csv"))
    export_json(insights, analyses, os.path.join(OUTPUT_DIR, "absa_insights.json"))

    try:
        from absa_visualizer import build_dashboard
        build_dashboard(insights, os.path.join(OUTPUT_DIR, "absa_dashboard.png"), analyses)
    except Exception as e:
        print(f"  ⚠  Dashboard skipped: {e}")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  {Fore.GREEN}✓  Pipeline complete in {elapsed:.2f}s{Style.RESET_ALL}")
    print(f"  Outputs → {Fore.WHITE}{OUTPUT_DIR}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
