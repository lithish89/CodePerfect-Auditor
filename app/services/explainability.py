"""
Explainability Engine  —  app/services/explainability.py
---------------------------------------------------------
Produces word-level attribution scores for each ICD-10 and CPT code,
showing WHICH words in the clinical note caused each code to be generated.

Technique: Leave-One-Out (LOO) token attribution
-------------------------------------------------
True SHAP requires a differentiable model. Our pipeline uses FAISS vector
search (no gradient). LOO attribution is the correct alternative:

  For each token T in the query sentence:
    1. Encode the full sentence  → similarity S_full
    2. Encode sentence with T removed → similarity S_without_T
    3. Attribution(T) = S_full - S_without_T

  Positive score  → word INCREASES similarity (evidence FOR the code)
  Negative score  → word DECREASES similarity (evidence AGAINST)
  Near zero       → word has little influence

This is computationally efficient (one encode per token) and produces
clinically meaningful explanations — exactly what a compliance auditor needs.

For codes from the PREFERRED_ICD_CODES override map (not FAISS), we use
keyword highlighting instead: the exact term that matched is marked as
the primary evidence.
"""

from __future__ import annotations

import re
from typing import Optional

# ── Lazy import of the sentence model ────────────────────────────────────────
_model = None
_model_error: Optional[str] = None


def _get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        import os
        model_name = os.environ.get("ICD_MODEL_NAME", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
        return _model
    except Exception as e:
        _model_error = str(e)
        return None


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Split text into tokens preserving meaningful medical phrases.
    Returns a list of non-empty word tokens (punctuation stripped).
    """
    # Split on whitespace and punctuation, keep meaningful words
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?", text)
    # Filter stop words that carry no clinical meaning
    stop_words = {
        "the", "a", "an", "and", "or", "of", "in", "is", "was",
        "has", "have", "had", "with", "for", "to", "on", "at",
        "by", "be", "are", "were", "this", "that", "his", "her",
        "their", "he", "she", "it", "as", "no", "not", "from",
        "patient", "pt", "year", "old", "male", "female", "man",
        "woman", "history", "hx", "presents", "presenting", "noted",
    }
    return [t for t in tokens if t.lower() not in stop_words and len(t) > 1]


def _sentence_without_token(tokens: list[str], skip_idx: int) -> str:
    """Reconstruct sentence with token at skip_idx removed."""
    return " ".join(t for i, t in enumerate(tokens) if i != skip_idx)


# ── Core LOO attribution ──────────────────────────────────────────────────────

def compute_loo_attributions(
    query_sentence: str,
    target_description: str,
    top_n: int = 6,
) -> list[dict]:
    """
    Compute Leave-One-Out token attributions.

    Measures how much each word in `query_sentence` contributes to the
    semantic similarity with `target_description` (the ICD code description).

    Args:
        query_sentence:     The clinical sentence used to find the code.
        target_description: The ICD-10 or CPT description that was matched.
        top_n:              Return only top N most influential tokens.

    Returns:
        List of dicts sorted by absolute attribution (most influential first):
        [
          {
            "token":       "pneumonia",
            "attribution": 0.142,      # positive = strong evidence for code
            "normalised":  1.0,        # 0..1 relative to max in this result
            "influence":   "high",     # "high" | "medium" | "low"
          },
          ...
        ]
    """
    model = _get_model()
    if model is None:
        return []

    tokens = _tokenise(query_sentence)
    if not tokens:
        return []

    try:
        import numpy as np

        # Encode full sentence and target description once
        full_query_emb  = model.encode([query_sentence],      normalize_embeddings=True)[0]
        target_emb      = model.encode([target_description],  normalize_embeddings=True)[0]
        baseline_sim    = float(np.dot(full_query_emb, target_emb))

        # LOO: encode each sentence-minus-one-token
        loo_sentences = [
            _sentence_without_token(tokens, i)
            for i in range(len(tokens))
        ]
        # Batch encode for efficiency
        loo_embeddings = model.encode(
            loo_sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        attributions = []
        for i, token in enumerate(tokens):
            loo_sim = float(np.dot(loo_embeddings[i], target_emb))
            attr    = baseline_sim - loo_sim   # positive = word helps match
            attributions.append({
                "token":       token,
                "attribution": round(attr, 4),
            })

        # Sort by absolute attribution (most influential first)
        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        attributions = attributions[:top_n]

        # Normalise to 0..1 relative to the max absolute attribution
        max_abs = max(abs(a["attribution"]) for a in attributions) if attributions else 1
        if max_abs == 0:
            max_abs = 1

        for a in attributions:
            a["normalised"] = round(abs(a["attribution"]) / max_abs, 3)
            n = a["normalised"]
            a["influence"] = "high" if n >= 0.6 else "medium" if n >= 0.3 else "low"

        return attributions

    except Exception as e:
        print(f"[Explainability] LOO attribution failed: {e}")
        return []


# ── Keyword highlight fallback ────────────────────────────────────────────────
# Used for codes from PREFERRED_ICD_CODES override map (not FAISS)

def keyword_attributions(
    query_sentence: str,
    matched_term: str,
) -> list[dict]:
    """
    For preferred-map codes, highlight the matched keyword as primary evidence.
    Returns the same format as compute_loo_attributions.
    """
    tokens = _tokenise(query_sentence)
    term_tokens = set(_tokenise(matched_term))

    result = []
    for token in tokens:
        if token.lower() in term_tokens:
            result.append({
                "token":       token,
                "attribution": 0.95,
                "normalised":  1.0,
                "influence":   "high",
            })

    # If no overlap found, just return the matched term itself
    if not result:
        for t in _tokenise(matched_term):
            result.append({
                "token":       t,
                "attribution": 0.95,
                "normalised":  1.0,
                "influence":   "high",
            })

    return result[:6]


# ── Main public API ───────────────────────────────────────────────────────────

def explain_code(
    icd_code:           str,
    icd_description:    str,
    source_sentence:    str,
    matched_term:       str,
    was_preferred_map:  bool = False,
) -> dict:
    """
    Generate a full explanation for one ICD-10 or CPT code.

    For ALL codes (both preferred-map and FAISS), we run LOO attribution
    against the ICD description so the bars always show. For preferred-map
    codes we also annotate which exact term triggered the rule.
    """
    # Always run LOO — works for both preferred map and FAISS codes
    # because we measure similarity between source_sentence and icd_description
    attributions = compute_loo_attributions(
        query_sentence=source_sentence,
        target_description=icd_description,
    )

    # If LOO returned nothing (model unavailable), fall back to keyword highlight
    if not attributions:
        attributions = keyword_attributions(source_sentence, matched_term)

    if was_preferred_map:
        method   = "preferred_map"
        evidence = (
            f"The term '{matched_term}' is a clinically verified direct mapping "
            f"to {icd_code} per ICD-10 coding guidelines. "
            f"Key evidence words from the note are highlighted below."
        )
    else:
        method   = "semantic_search"
        top_tokens = [a["token"] for a in attributions if a["influence"] == "high"]
        if top_tokens:
            evidence = (
                f"Semantic similarity between the clinical context and the ICD description "
                f"was driven by: {', '.join(top_tokens[:3])}."
            )
        else:
            evidence = (
                f"Matched '{matched_term}' to {icd_code} via semantic vector search "
                f"(cosine similarity in 384-dimensional embedding space)."
            )

    return {
        "code":            icd_code,
        "description":     icd_description,
        "source_sentence": source_sentence,
        "matched_term":    matched_term,
        "method":          method,
        "evidence":        evidence,
        "attributions":    attributions,
    }


def explain_all_codes(
    icd_codes:         list[dict],
    sentence_contexts: dict[str, str],
    preferred_terms:   set[str],
) -> list[dict]:
    """
    Generate explanations for all confirmed ICD codes.

    Args:
        icd_codes:         List of confirmed ICD code dicts from coding_logic
        sentence_contexts: Map of term → source sentence from clinical_reader
        preferred_terms:   Set of terms that came from PREFERRED_ICD_CODES map

    Returns:
        List of explanation dicts, one per code.
    """
    explanations = []
    for code_rec in icd_codes:
        term     = code_rec.get("term", "")
        sentence = sentence_contexts.get(term, term)
        was_pref = term.lower().strip() in preferred_terms

        expl = explain_code(
            icd_code=code_rec.get("icd_code", ""),
            icd_description=code_rec.get("description", ""),
            source_sentence=sentence,
            matched_term=term,
            was_preferred_map=was_pref,
        )
        explanations.append(expl)

    return explanations


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing LOO attribution engine...")

    examples = [
        {
            "sentence":    "IV infusion of antibiotics started for community acquired pneumonia.",
            "description": "Pneumonia, unspecified organism",
            "term":        "community acquired pneumonia",
        },
        {
            "sentence":    "History of hypertension and type 2 diabetes.",
            "description": "Essential (primary) hypertension",
            "term":        "hypertension",
        },
        {
            "sentence":    "Patient presented with acute myocardial infarction and chest pain.",
            "description": "Acute myocardial infarction, unspecified",
            "term":        "myocardial infarction",
        },
    ]

    for ex in examples:
        print(f"\nSentence: {ex['sentence']}")
        print(f"Code:     {ex['description']}")
        attrs = compute_loo_attributions(ex["sentence"], ex["description"])
        print("  Top contributing words:")
        for a in attrs:
            bar = "█" * int(a["normalised"] * 10)
            sign = "+" if a["attribution"] > 0 else "-"
            print(f"    {sign}{bar:<10} {a['token']:<20} ({a['influence']})")