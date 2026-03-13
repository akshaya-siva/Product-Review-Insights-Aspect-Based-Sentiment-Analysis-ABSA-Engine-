# Product-Review-Insights-Aspect-Based-Sentiment-Analysis-ABSA-Engine-
Designed a sentiment tool to extract specific feedback (e.g., "battery life," "camera") from reviews; helped identify top product issues beyond simple star ratings using Python.
## PROJECT STRUCTURE

```
absa_engine.py          ← Core NLP engine (the brain — import this)
review_data.py          ← Data model + 22 sample reviews across 3 products
absa_main.py            ← Pipeline runner → CLI output + CSV + JSON + dashboard
absa_visualizer.py      ← Original single-page dashboard (7 panels)
absa_full_dashboard.py  ← Full 5-page analytics dashboard (standalone)

output/
  absa_results.csv           ← Flat export: 1 row per aspect per review
  absa_insights.json         ← Nested export: full product + review tree
  absa_dashboard.png         ← Original 7-panel overview dashboard
  page1_executive_summary.png
  page2_product_deepdive.png
  page3_review_analysis.png
  page4_comparative.png
  page5_insights_report.png
```

---

## HOW TO RUN

### Full pipeline (CLI + exports + dashboards)
```bash
python absa_main.py
```

### 5-page dashboard only
```bash
python absa_full_dashboard.py
```

### Use the engine in your own code
```python
from absa_engine import analyze_review, aggregate_product_insights

result = analyze_review(
    text="The camera is absolutely incredible but the battery drains too fast.",
    review_id="R001",
    product_id="P001",
    star_rating=3
)

for aspect in result.aspects:
    print(f"{aspect.aspect}: {aspect.score:+.2f} ({aspect.sentiment})")
    print(f"  Evidence: '{aspect.sentence}'")
    print(f"  Opinion words: {aspect.opinion_words}")
```

---

## WHAT ABSA DOES (vs simple sentiment)

| Simple Sentiment | ABSA |
|---|---|
| "This phone is okay" → 3/5 ★ | camera=+2.74, battery=-0.49, display=+3.49 |
| One score per review | One score per aspect per review |
| Can't tell you WHY | Shows which feature customers love/hate |
| Useless for product teams | Actionable: fix battery, keep camera |

---

## PRODUCTS IN SAMPLE DATASET

| ID | Name | Reviews | NLP Score | Avg ★ |
|---|---|---|---|---|
| P001 | NovaTech X12 Pro | 8 | +1.01 | 3.12 |
| P002 | Lumina S8 Ultra | 7 | +2.08 | 4.43 |
| P003 | SwiftPhone 7 | 7 | +0.30 | 3.57 |

---

## DEPENDENCIES

```
numpy
matplotlib
colorama  (optional — graceful fallback if missing)
```

Install: `pip install numpy matplotlib colorama`

---

## KEY NLP TECHNIQUES USED

- **Sentence splitting with contrast awareness** — splits on "but", "however", "although" so mixed reviews are correctly separated
- **Aspect taxonomy** — 10 aspect categories, each with domain-specific trigger phrases
- **Opinion lexicon** — 70+ positive and negative adjectives with calibrated weights (−3 to +3)
- **Sliding context window** — looks ±6 tokens around each aspect trigger to find nearby opinion words
- **Intensifier detection** — "absolutely incredible" scores 1.9× higher than "incredible"
- **Negation handling** — "not disappointed" flips polarity with −0.6× dampening
- **Product aggregation** — averages aspect scores across all reviews per product
