"""Score comments with the TF-IDF model bundle and derive a moderation recommendation."""

from functools import lru_cache

import joblib

from portfolio_projekt.data import MODELS_DIR

BUNDLE_PATH = MODELS_DIR / "tfidf_en.joblib"

ACTIONS = {
    0: "No action",
    1: "Monitor",
    2: "Manual review",
    3: "Hide until review",
    4: "Delete",
    5: "Hide and escalate to a moderator",
}


@lru_cache(maxsize=1)
def load_bundle():
    """Load the model bundle once and keep it in memory."""
    return joblib.load(BUNDLE_PATH)


def score(text: str) -> dict[str, float]:
    """Return the predicted probability for each label."""
    bundle = load_bundle()
    X = bundle["vectorizer"].transform([text])
    return {label: float(model.predict_proba(X)[0, 1]) for label, model in bundle["models"].items()}


def recommend(scores: dict[str, float], thresholds: dict[str, float]) -> dict:
    """Map label probabilities to an action level (0-5) with a reason."""
    toxicity = scores["toxicity"]

    def above(label: str) -> bool:
        return scores[label] >= thresholds[label]

    if above("threat"):
        level, reason = 5, "Possible threat. The threat model is weak, so a human must decide."
    elif toxicity >= 0.8 and (above("insult") or above("obscene")):
        level, reason = 4, "Clearly toxic with insult or obscenity."
    elif toxicity >= 0.8:
        level, reason = 3, "Highly toxic."
    elif above("identity_attack"):
        level, reason = 3, "Possible identity attack. Check carefully: identity terms can cause false alarms."
    elif above("sexual_explicit"):
        level, reason = 3, "Possibly sexually explicit."
    elif toxicity >= thresholds["toxicity"]:
        level, reason = 2, "Likely toxic, but not clear-cut."
    elif toxicity >= 0.15:
        level, reason = 1, "Borderline."
    else:
        level, reason = 0, "No signs of toxicity."

    flagged = [label for label in scores if label != "toxicity" and above(label)]
    return {"level": level, "action": ACTIONS[level], "reason": reason, "flagged": flagged}


@lru_cache(maxsize=1)
def _feature_names():
    return load_bundle()["vectorizer"].get_feature_names_out()


def explain(text: str, label: str = "toxicity", top_n: int = 5) -> list[tuple[str, float]]:
    """Return the words or word pairs that push the score of a label up the most."""
    bundle = load_bundle()
    X = bundle["vectorizer"].transform([text])
    weights = bundle["models"][label].coef_[0]
    names = _feature_names()

    contributions = [(names[i], X[0, i] * weights[i]) for i in X.nonzero()[1]]
    contributions = [(word, round(float(value), 3)) for word, value in contributions if value > 0]
    return sorted(contributions, key=lambda pair: pair[1], reverse=True)[:top_n]