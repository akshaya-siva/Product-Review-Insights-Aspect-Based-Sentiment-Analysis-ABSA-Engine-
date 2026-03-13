"""
absa_engine.py
──────────────
Aspect-Based Sentiment Analysis (ABSA) Engine
Pure Python + NumPy — no ML models, no internet, no heavy dependencies.

Pipeline:
  Raw Review Text
      ↓
  Sentence Splitting
      ↓
  Aspect Detection  (lexicon + pattern matching per sentence)
      ↓
  Opinion Word Extraction  (adjectives + sentiment lexicon)
      ↓
  Sentiment Scoring  (per aspect, with intensifiers + negation)
      ↓
  Aspect Sentiment Objects  (aspect, sentiment, score, evidence)
      ↓
  Review-Level Aggregation  (per-product aspect averages)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


# ══════════════════════════════════════════════════════════════════
# ASPECT TAXONOMY
# Each product category has a set of aspects with trigger keywords.
# When any trigger appears in a sentence, that aspect is "detected".
# ══════════════════════════════════════════════════════════════════

ASPECT_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "battery": {
        "triggers": [
            "battery", "charge", "charging", "battery life", "power",
            "drain", "drains", "mah", "standby", "runtime", "battery drain",
            "plug in", "unplugged", "lasts", "last all day", "dies quickly",
            "recharge", "full charge"
        ],
        "icon": "🔋",
        "category": "hardware"
    },
    "camera": {
        "triggers": [
            "camera", "photo", "photos", "picture", "pictures", "image",
            "images", "shot", "shots", "lens", "megapixel", "mp",
            "selfie", "selfies", "video", "zoom", "bokeh", "night mode",
            "portrait", "flash", "autofocus", "shutter", "photography"
        ],
        "icon": "📷",
        "category": "hardware"
    },
    "display": {
        "triggers": [
            "screen", "display", "resolution", "brightness", "amoled",
            "oled", "lcd", "refresh rate", "hz", "pixels", "vibrant",
            "color accuracy", "sunlight", "viewing angle", "touch",
            "responsive", "smooth", "panel"
        ],
        "icon": "🖥️",
        "category": "hardware"
    },
    "performance": {
        "triggers": [
            "performance", "speed", "fast", "slow", "lag", "lags",
            "processor", "cpu", "gpu", "benchmark", "gaming", "game",
            "apps", "multitask", "multitasking", "heating", "heat",
            "throttle", "ram", "memory", "snapdragon", "chip", "heavy apps"
        ],
        "icon": "⚡",
        "category": "software"
    },
    "build_quality": {
        "triggers": [
            "build", "build quality", "design", "premium", "plastic",
            "metal", "glass", "feel", "sturdy", "flimsy", "durable",
            "finish", "weight", "heavy", "light", "thin", "thick",
            "material", "construction", "look", "aesthetic", "in hand"
        ],
        "icon": "🏗️",
        "category": "hardware"
    },
    "software": {
        "triggers": [
            "software", "os", "android", "ios", "ui", "ux",
            "interface", "bloatware", "update", "updates", "bug",
            "bugs", "stable", "smooth", "one ui", "miui", "gesture",
            "notification", "feature", "customization", "settings", "stock android"
        ],
        "icon": "💿",
        "category": "software"
    },
    "audio": {
        "triggers": [
            "audio", "speaker", "speakers", "sound", "bass", "treble",
            "headphone", "earphone", "jack", "stereo", "mono",
            "volume", "loud", "microphone", "call quality", "noise",
            "music", "podcast", "dolby"
        ],
        "icon": "🔊",
        "category": "hardware"
    },
    "price_value": {
        "triggers": [
            "price", "value", "money", "worth", "expensive", "cheap",
            "affordable", "budget", "cost", "rupees", "dollars", "inr",
            "usd", "bang for buck", "premium price", "overpriced",
            "underpriced", "mid range", "flagship"
        ],
        "icon": "💰",
        "category": "business"
    },
    "connectivity": {
        "triggers": [
            "wifi", "wi-fi", "bluetooth", "nfc", "5g", "4g",
            "signal", "network", "connectivity", "usb", "type-c",
            "sim", "dual sim", "esim", "hotspot", "range"
        ],
        "icon": "📡",
        "category": "hardware"
    },
    "delivery_packaging": {
        "triggers": [
            "delivery", "shipping", "package", "packaging", "box",
            "arrived", "courier", "damage", "damaged", "unboxing",
            "accessories", "charger included", "earphones included"
        ],
        "icon": "📦",
        "category": "service"
    },
}

# ══════════════════════════════════════════════════════════════════
# OPINION LEXICONS  —  Adjectives that carry sentiment
# ══════════════════════════════════════════════════════════════════

POSITIVE_OPINIONS: Dict[str, float] = {
    # Strong positive (3.0)
    "excellent": 3.0, "outstanding": 3.0, "exceptional": 3.0,
    "incredible": 3.0, "phenomenal": 3.0, "spectacular": 3.0,
    "flawless": 3.0, "perfect": 3.0, "brilliant": 3.0, "superb": 3.0,
    # Good positive (2.0)
    "great": 2.0, "amazing": 2.0, "fantastic": 2.0, "wonderful": 2.0,
    "impressive": 2.0, "sharp": 2.0, "crisp": 2.0, "stunning": 2.0,
    "vibrant": 2.0, "powerful": 2.0, "responsive": 2.0,
    # Mild positive (1.0)
    "good": 1.0, "nice": 1.0, "decent": 1.0, "solid": 1.0,
    "fast": 1.0, "smooth": 1.0, "clear": 1.0, "bright": 1.0,
    "clean": 1.0, "reliable": 1.0, "stable": 1.0, "accurate": 1.0,
    "efficient": 1.0, "comfortable": 1.0, "useful": 1.0,
    # Contextual
    "long": 0.8, "lasting": 0.8, "improved": 0.8, "better": 0.8,
    "worth": 0.8, "love": 1.5, "loved": 1.5, "enjoying": 1.2,
    "happy": 1.2, "satisfied": 1.0, "recommend": 1.5, "impressive": 2.0,
}

NEGATIVE_OPINIONS: Dict[str, float] = {
    # Strong negative (-3.0)
    "terrible": -3.0, "horrible": -3.0, "awful": -3.0, "atrocious": -3.0,
    "dreadful": -3.0, "appalling": -3.0, "disaster": -3.0, "useless": -3.0,
    # Moderate negative (-2.0)
    "bad": -2.0, "poor": -2.0, "disappointing": -2.0, "disappointed": -2.0,
    "worse": -2.0, "worst": -2.0, "mediocre": -2.0, "subpar": -2.0,
    "inferior": -2.0, "defective": -2.0, "broken": -2.0, "faulty": -2.0,
    # Mild negative (-1.0)
    "slow": -1.0, "weak": -1.0, "dull": -1.0, "dim": -1.0, "dark": -0.5,
    "blurry": -1.0, "grainy": -1.0, "laggy": -1.5, "choppy": -1.5,
    "overheating": -2.0, "heating": -1.5, "drains": -1.5, "drain": -1.5,
    "short": -1.0, "limited": -0.8, "outdated": -1.0, "bloated": -1.2,
    "buggy": -2.0, "unstable": -1.5, "crashes": -2.0, "freezes": -2.0,
    "noisy": -1.0, "tinny": -1.5, "scratches": -1.0, "flimsy": -1.5,
    "overpriced": -2.0, "expensive": -1.0, "cheap": -0.8, "miss": -1.0,
    "lacking": -1.0, "missing": -1.0, "annoying": -1.5, "frustrating": -2.0,
    "waste": -2.5, "regret": -2.5, "return": -1.5,
}

ALL_OPINIONS = {**POSITIVE_OPINIONS, **NEGATIVE_OPINIONS}

INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "extremely": 2.0, "really": 1.4, "absolutely": 1.9,
    "incredibly": 1.8, "exceptionally": 1.8, "quite": 1.2, "pretty": 1.1,
    "super": 1.6, "totally": 1.5, "completely": 1.7, "highly": 1.5,
    "so": 1.3, "too": 1.4, "overly": 1.5,
}

NEGATIONS = {
    "not", "no", "never", "don't", "doesn't", "didn't", "won't",
    "isn't", "aren't", "wasn't", "weren't", "can't", "cannot", "hardly",
    "barely", "neither", "nor", "without",
}

CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "while",
                  "whereas", "despite", "except", "unfortunately", "sadly"}


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class AspectSentiment:
    aspect:        str          # e.g. "battery"
    sentiment:     str          # "positive" | "negative" | "neutral" | "mixed"
    score:         float        # -3.0 to +3.0
    opinion_words: List[str]    # detected opinion words
    sentence:      str          # source sentence
    confidence:    float        # 0.0 to 1.0


@dataclass
class ReviewAnalysis:
    review_id:     str
    product_id:    str
    raw_text:      str
    star_rating:   Optional[int]
    aspects:       List[AspectSentiment]
    overall_score: float        # weighted average across aspects
    sentiment_label: str        # "positive" | "negative" | "neutral" | "mixed"
    contrast_detected: bool     # did we find "X is good but Y is bad" patterns?
    dominant_aspects:  List[str]
    word_count:    int


@dataclass
class ProductInsight:
    product_id:   str
    product_name: str
    total_reviews: int
    aspect_scores: Dict[str, float]        # aspect → avg score
    aspect_counts: Dict[str, int]          # aspect → mention count
    aspect_sentiments: Dict[str, str]      # aspect → dominant sentiment label
    aspect_samples: Dict[str, List[str]]   # aspect → sample sentences
    overall_score: float
    overall_sentiment: str
    top_praised:   List[str]               # top 3 positively mentioned aspects
    top_criticized: List[str]              # top 3 negatively mentioned aspects
    star_distribution: Dict[int, int]      # 1-5 → count
    avg_star_rating: float


# ══════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> List[str]:
    """Split review text into sentences, handling common abbreviations."""
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on .!? followed by space+uppercase, or on connectors
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on contrast connectors mid-sentence
    result = []
    for sent in sentences:
        # Split on " but ", " however ", " although " etc.
        parts = re.split(r'\s+(?:but|however|although|though|whereas)\s+', sent, flags=re.IGNORECASE)
        result.extend([p.strip() for p in parts if p.strip()])
    return [s for s in result if len(s) > 5]


def _tokenize(text: str) -> List[str]:
    """Extract word tokens preserving contractions."""
    return re.findall(r"[a-z']+", text.lower())


def _get_context_window(tokens: List[str], index: int, window: int = 4) -> List[str]:
    """Return tokens within window before given index."""
    start = max(0, index - window)
    return tokens[start:index]


def _compute_opinion_score(tokens: List[str], opinion_idx: int) -> float:
    """Compute score for an opinion word considering context."""
    word = tokens[opinion_idx]
    if word not in ALL_OPINIONS:
        return 0.0

    base_score = ALL_OPINIONS[word]
    context = _get_context_window(tokens, opinion_idx, window=4)

    # Check negation
    is_negated = any(w in NEGATIONS for w in context)
    # Check intensifier
    intensifier = max((INTENSIFIERS.get(w, 1.0) for w in context), default=1.0)

    score = base_score * intensifier
    if is_negated:
        score = -score * 0.6  # flip with dampening

    return score


def _detect_aspects_in_sentence(sentence: str) -> List[str]:
    """Find all aspects mentioned in a sentence."""
    lower = sentence.lower()
    found = []
    for aspect, cfg in ASPECT_TAXONOMY.items():
        for trigger in cfg["triggers"]:
            # Match whole word / phrase
            pattern = r'\b' + re.escape(trigger) + r'\b'
            if re.search(pattern, lower):
                if aspect not in found:
                    found.append(aspect)
                break
    return found


def _extract_opinion_words(tokens: List[str], aspect_token_indices: List[int],
                            window: int = 6) -> Tuple[List[str], float]:
    """
    Extract opinion words within a window of aspect token positions.
    Returns (opinion_words, aggregate_score).
    """
    relevant_indices = set()
    for ai in aspect_token_indices:
        for i in range(max(0, ai - window), min(len(tokens), ai + window + 1)):
            relevant_indices.add(i)

    opinion_words = []
    total_score = 0.0

    for idx in sorted(relevant_indices):
        word = tokens[idx]
        if word in ALL_OPINIONS:
            score = _compute_opinion_score(tokens, idx)
            if score != 0.0:
                opinion_words.append(word)
                total_score += score

    return opinion_words, total_score


def _score_to_sentiment(score: float, count: int) -> Tuple[str, float]:
    """Convert numeric score to sentiment label and confidence."""
    if count == 0:
        return "neutral", 0.3

    avg = score / count if count > 0 else score

    if avg >= 1.5:
        sentiment, confidence = "positive", min(0.95, 0.6 + avg * 0.1)
    elif avg >= 0.3:
        sentiment, confidence = "positive", min(0.85, 0.5 + avg * 0.1)
    elif avg <= -1.5:
        sentiment, confidence = "negative", min(0.95, 0.6 + abs(avg) * 0.1)
    elif avg <= -0.3:
        sentiment, confidence = "negative", min(0.85, 0.5 + abs(avg) * 0.1)
    else:
        sentiment, confidence = "neutral", 0.4

    return sentiment, confidence


def analyze_review(text: str, review_id: str = "R000",
                   product_id: str = "P000",
                   star_rating: Optional[int] = None) -> ReviewAnalysis:
    """Full ABSA pipeline for a single review."""

    sentences = _split_sentences(text)
    all_aspects: List[AspectSentiment] = []
    contrast_detected = False

    for sentence in sentences:
        # Check for contrast pattern
        lower_sent = sentence.lower()
        if any(w in lower_sent for w in CONTRAST_WORDS):
            contrast_detected = True

        detected_aspects = _detect_aspects_in_sentence(sentence)
        if not detected_aspects:
            continue

        tokens = _tokenize(sentence)

        for aspect in detected_aspects:
            # Find positions of aspect trigger tokens in this sentence
            triggers = ASPECT_TAXONOMY[aspect]["triggers"]
            aspect_positions = []
            for i, tok in enumerate(tokens):
                for trig in triggers:
                    trig_tokens = trig.split()
                    if len(trig_tokens) == 1 and tok == trig:
                        aspect_positions.append(i)
                    elif len(trig_tokens) > 1:
                        # Multi-word: check substring
                        if trig in lower_sent:
                            aspect_positions.append(i)

            if not aspect_positions:
                aspect_positions = [0]  # fallback

            opinion_words, score = _extract_opinion_words(tokens, aspect_positions)
            sentiment, confidence = _score_to_sentiment(score, len(opinion_words))

            all_aspects.append(AspectSentiment(
                aspect=aspect,
                sentiment=sentiment,
                score=round(score, 3),
                opinion_words=opinion_words,
                sentence=sentence.strip(),
                confidence=confidence,
            ))

    # Overall score
    if all_aspects:
        overall_score = np.mean([a.score for a in all_aspects])
        # Determine if mixed (has both positive and negative aspects)
        has_pos = any(a.sentiment == "positive" for a in all_aspects)
        has_neg = any(a.sentiment == "negative" for a in all_aspects)
        if has_pos and has_neg:
            sentiment_label = "mixed"
        elif has_pos:
            sentiment_label = "positive"
        elif has_neg:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
    else:
        overall_score = 0.0
        sentiment_label = "neutral"

    # Dominant aspects (most mentioned)
    aspect_mention_count = defaultdict(int)
    for a in all_aspects:
        aspect_mention_count[a.aspect] += 1
    dominant = sorted(aspect_mention_count, key=lambda x: aspect_mention_count[x], reverse=True)[:3]

    return ReviewAnalysis(
        review_id=review_id,
        product_id=product_id,
        raw_text=text,
        star_rating=star_rating,
        aspects=all_aspects,
        overall_score=round(float(overall_score), 3),
        sentiment_label=sentiment_label,
        contrast_detected=contrast_detected,
        dominant_aspects=dominant,
        word_count=len(text.split()),
    )


# ══════════════════════════════════════════════════════════════════
# PRODUCT-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════

def aggregate_product_insights(analyses: List[ReviewAnalysis],
                                product_id: str,
                                product_name: str) -> ProductInsight:
    """Aggregate multiple review analyses into product-level insights."""

    aspect_scores_raw: Dict[str, List[float]] = defaultdict(list)
    aspect_samples: Dict[str, List[str]] = defaultdict(list)
    star_dist: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for analysis in analyses:
        if analysis.star_rating and 1 <= analysis.star_rating <= 5:
            star_dist[analysis.star_rating] += 1
        for aspect in analysis.aspects:
            aspect_scores_raw[aspect.aspect].append(aspect.score)
            if len(aspect_samples[aspect.aspect]) < 3:  # keep 3 sample sentences
                aspect_samples[aspect.aspect].append(aspect.sentence)

    # Average scores per aspect
    aspect_scores = {a: round(float(np.mean(scores)), 3)
                     for a, scores in aspect_scores_raw.items()}
    aspect_counts = {a: len(scores) for a, scores in aspect_scores_raw.items()}

    # Dominant sentiment label per aspect
    aspect_sentiments = {}
    for aspect, scores in aspect_scores_raw.items():
        avg = np.mean(scores)
        if avg >= 0.3:
            aspect_sentiments[aspect] = "positive"
        elif avg <= -0.3:
            aspect_sentiments[aspect] = "negative"
        else:
            aspect_sentiments[aspect] = "neutral"

    # Top praised / criticized
    sorted_aspects = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)
    top_praised   = [a for a, s in sorted_aspects if s > 0][:3]
    top_criticized = [a for a, s in reversed(sorted_aspects) if s < 0][:3]

    overall = round(float(np.mean([a.overall_score for a in analyses])), 3) if analyses else 0.0
    total_stars = sum(k * v for k, v in star_dist.items())
    total_rated = sum(star_dist.values())
    avg_star = round(total_stars / total_rated, 2) if total_rated > 0 else 0.0

    if overall >= 0.5:
        overall_sentiment = "positive"
    elif overall <= -0.5:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "mixed"

    return ProductInsight(
        product_id=product_id,
        product_name=product_name,
        total_reviews=len(analyses),
        aspect_scores=aspect_scores,
        aspect_counts=aspect_counts,
        aspect_sentiments=aspect_sentiments,
        aspect_samples=dict(aspect_samples),
        overall_score=overall,
        overall_sentiment=overall_sentiment,
        top_praised=top_praised,
        top_criticized=top_criticized,
        star_distribution=star_dist,
        avg_star_rating=avg_star,
    )
