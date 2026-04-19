"""
Auditor Agent
-------------
Compares AI-generated ICD-10 / CPT codes against human-coded entries and:
  - Identifies MISSING codes  (AI found them, human missed them)
  - Identifies EXTRA codes    (human added them, AI didn't generate them)
  - Identifies MATCHED codes  (both agree)
  - Scores revenue risk for each discrepancy
  - Generates a structured audit report with explainability notes

This is the compliance and governance layer of CodePerfect Auditor.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Revenue risk scoring
# ---------------------------------------------------------------------------

# Per-code approximate CMS reimbursement estimates (USD)
# Based on Medicare national average allowed amounts — rough heuristics only
CPT_REIMBURSEMENT: dict[str, int] = {
    "71250": 380,   # CT thorax
    "71046": 55,    # Chest X-ray
    "85025": 18,    # CBC
    "93000": 28,    # ECG
    "93306": 520,   # Echocardiogram
    "96365": 95,    # IV infusion
    "62270": 280,   # Lumbar puncture
    "45378": 470,   # Colonoscopy
    "43239": 390,   # Upper GI endoscopy
    "88305": 95,    # Biopsy
    "90935": 240,   # Hemodialysis
    "92920": 1800,  # Angioplasty
    "92928": 2400,  # Stenting
    "92960": 320,   # Cardioversion
    "44950": 1400,  # Appendectomy
    "47562": 1600,  # Cholecystectomy
    "31500": 210,   # Intubation
    "97110": 40,    # Physical therapy
    "81001": 8,     # Urinalysis
    "87040": 25,    # Blood culture
    "76700": 140,   # Abdominal ultrasound
    "70553": 480,   # MRI brain
    "74177": 420,   # CT abdomen
}

# ICD DRG weight estimates — how much extra a diagnosis adds to a hospital bill
ICD_CHAPTER_VALUE: dict[str, int] = {
    "I": 2800,   # Circulatory (heart failure, MI) — high DRG weight
    "C": 4500,   # Neoplasms — very high
    "N": 1800,   # Genitourinary (CKD)
    "J": 1200,   # Respiratory (pneumonia, COPD)
    "E": 900,    # Endocrine (diabetes)
    "K": 1400,   # Digestive
    "S": 2200,   # Injury / trauma
    "T": 1600,   # Poisoning
    "A": 800,    # Infectious
    "G": 1100,   # Neurological
    "M": 600,    # Musculoskeletal
    "F": 700,    # Mental health
    "R": 150,    # Symptoms / signs (lowest value)
    "Z": 0,      # Status codes — no direct reimbursement
}

# ICD chapter prefixes with high reimbursement impact
HIGH_IMPACT_ICD_PREFIXES: tuple[str, ...] = (
    "I", "C", "N", "J", "E", "K", "S", "T",
)


def _estimate_code_value(code: str, code_type: str) -> int:
    """Return approximate USD value of a single code."""
    code_clean = code.upper().replace(".", "").strip()
    if code_type == "cpt":
        return CPT_REIMBURSEMENT.get(code_clean, 150)
    # ICD — look up chapter value
    chapter = code_clean[0] if code_clean else "R"
    return ICD_CHAPTER_VALUE.get(chapter, 200)


def _classify_revenue_risk(code: str, code_type: str = "icd") -> str:
    """Classify revenue risk for a single discrepant code."""
    val = _estimate_code_value(code, code_type)
    if val >= 800:
        return "high"
    if val >= 200:
        return "medium"
    return "low"


def _calculate_revenue_impact(missing_codes: list, extra_codes: list) -> dict:
    """
    Calculate a realistic revenue impact estimate from the actual codes.
    Missing codes = potential revenue loss (undercoding).
    Extra codes = potential penalty / clawback risk (upcoding).
    """
    revenue_lost   = sum(_estimate_code_value(c["code"], "cpt" if c.get("type") == "CPT" else "icd") for c in missing_codes)
    penalty_risk   = sum(_estimate_code_value(c["code"], "cpt" if c.get("type") == "CPT" else "icd") for c in extra_codes)

    def fmt(v: int) -> str:
        if v == 0:
            return "$0"
        if v < 100:
            return f"~${v}"
        # Round to nearest $50
        rounded = round(v / 50) * 50
        return f"~${rounded:,}"

    total_at_risk = revenue_lost + penalty_risk
    if total_at_risk >= 1000:
        overall = "high"
    elif total_at_risk >= 200:
        overall = "medium"
    elif total_at_risk > 0:
        overall = "low"
    else:
        overall = "clear"

    return {
        "revenue_lost_estimate":   fmt(revenue_lost),
        "penalty_risk_estimate":   fmt(penalty_risk),
        "total_at_risk":           fmt(total_at_risk),
        "range":                   f"{fmt(revenue_lost)} potential revenue loss · {fmt(penalty_risk)} penalty risk",
        "overall_risk":            overall,
    }


import re  # noqa: E402 — placed here to keep imports at top readable


def _explain_discrepancy(code: str, description: str, discrepancy_type: str) -> str:
    """Generate a short plain-English explanation for each discrepancy."""
    if discrepancy_type == "missing":
        return (
            f"Code {code} ({description}) was identified by AI but not entered by the "
            f"human coder. This may indicate undercoding, which can lead to revenue loss."
        )
    if discrepancy_type == "extra":
        return (
            f"Code {code} ({description}) was entered by the human coder but not "
            f"supported by the clinical note. This may indicate upcoding, which carries "
            f"compliance and penalty risk."
        )
    return f"Code {code} matched between AI and human coder."


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_code(code: str) -> str:
    """Uppercase, strip whitespace and dots for comparison."""
    return code.upper().replace(".", "").strip()


def _icd_subcategory(code: str) -> str:
    """
    Return the 3-character ICD subcategory prefix.
    J159 → J15,  J18.9 → J18,  I10 → I10,  E119 → E11
    This allows J159 and J18.9 to be treated as 'same disease block'
    when no exact match exists.
    """
    return _normalise_code(code)[:3]


def _build_code_lookup(code_list: list[dict], id_field: str) -> dict[str, dict]:
    """Build a normalised-code → record dict for fast lookup."""
    return {_normalise_code(c[id_field]): c for c in code_list if id_field in c}


def _find_fuzzy_icd_match(key: str, lookup: dict[str, dict]) -> Optional[str]:
    """
    Try to find a fuzzy match for `key` in `lookup` by ICD subcategory.
    Returns the matched key in `lookup` if found, else None.

    Example: key=J159, lookup has J189 → both have subcategory J1x → match.
    Only matches within the same 3-char block (J15x matches J18x would be wrong;
    we use first 3 chars so J15 only matches other J15x codes).
    """
    sub = _icd_subcategory(key)
    for existing_key in lookup:
        if _icd_subcategory(existing_key) == sub:
            return existing_key
    return None


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------

def audit_codes(
    ai_codes:     dict,
    human_codes:  list[dict],
) -> dict:
    """
    Compare AI-generated codes against human codes and produce an audit report.

    Args:
        ai_codes:    Output from generate_icd_codes() — contains 'icd_codes' and 'cpt_codes'.
        human_codes: List of dicts, each with either 'icd_code' or 'cpt_code' and 'description'.
                     Example:
                       [
                         {"icd_code": "J18.9", "description": "Pneumonia, unspecified"},
                         {"cpt_code": "93000", "description": "ECG routine"},
                       ]

    Returns:
        {
          "summary":        {matched, missing_count, extra_count, total_discrepancies},
          "matched":        [code records that agree],
          "missing_codes":  [codes AI found but human missed — revenue risk: undercoding],
          "extra_codes":    [codes human added but AI didn't find — compliance risk: upcoding],
          "risk_level":     "high" | "medium" | "low" | "clear",
          "revenue_risk":   str,
          "recommendations": [str],
          "audit_log":      { timestamp, total_ai_codes, total_human_codes, ... }
        }
    """
    ai_icd  = ai_codes.get("icd_codes", [])
    ai_cpt  = ai_codes.get("cpt_codes", [])
    ai_warnings = ai_codes.get("warnings", [])

    # Separate human codes into ICD vs CPT
    human_icd = [c for c in human_codes if "icd_code" in c]
    human_cpt = [c for c in human_codes if "cpt_code" in c]

    # Build normalised lookups
    ai_icd_map     = _build_code_lookup(ai_icd,     "icd_code")
    ai_cpt_map     = _build_code_lookup(ai_cpt,     "cpt_code")
    human_icd_map  = _build_code_lookup(human_icd,  "icd_code")
    human_cpt_map  = _build_code_lookup(human_cpt,  "cpt_code")

    matched:       list[dict] = []
    missing_codes: list[dict] = []  # AI has, human missing
    extra_codes:   list[dict] = []  # Human has, AI missing

    # ---- Compare ICD-10 codes -----------------------------------------------
    # We use a two-pass approach:
    # Pass 1: exact matches
    # Pass 2: subcategory fuzzy matches (J159 ↔ J189 = both bacterial pneumonia)
    all_icd_keys   = set(ai_icd_map) | set(human_icd_map)
    human_matched  : set[str] = set()   # human keys that have been matched
    ai_matched     : set[str] = set()   # ai keys that have been matched

    # Pass 1 — exact code matches
    for key in all_icd_keys:
        if key in ai_icd_map and key in human_icd_map:
            rec = ai_icd_map[key]
            matched.append({
                "code":        rec["icd_code"],
                "description": rec["description"],
                "type":        "ICD-10",
                "confidence":  rec.get("confidence", 1.0),
                "match_type":  "exact",
            })
            ai_matched.add(key)
            human_matched.add(key)

    # Pass 2 — fuzzy subcategory matches for unmatched codes
    unmatched_ai    = [k for k in ai_icd_map    if k not in ai_matched]
    unmatched_human = [k for k in human_icd_map if k not in human_matched]

    for ai_key in list(unmatched_ai):
        # Look for a human code in the same 3-char ICD subcategory
        fuzzy_human_key = None
        for hk in unmatched_human:
            if _icd_subcategory(ai_key) == _icd_subcategory(hk):
                fuzzy_human_key = hk
                break
        if fuzzy_human_key:
            ai_rec    = ai_icd_map[ai_key]
            hum_rec   = human_icd_map[fuzzy_human_key]
            matched.append({
                "code":        ai_rec["icd_code"],
                "description": ai_rec["description"],
                "type":        "ICD-10",
                "confidence":  ai_rec.get("confidence", 0.85),
                "match_type":  "subcategory",
                "note":        f"AI coded {ai_rec['icd_code']}, human coded {hum_rec['icd_code']} — same disease category",
            })
            ai_matched.add(ai_key)
            human_matched.add(fuzzy_human_key)
            unmatched_human.remove(fuzzy_human_key)

    # Remaining unmatched AI codes → missing
    for key in ai_icd_map:
        if key in ai_matched:
            continue
        rec  = ai_icd_map[key]
        risk = _classify_revenue_risk(rec["icd_code"], "icd")
        missing_codes.append({
            "code":        rec["icd_code"],
            "description": rec["description"],
            "type":        "ICD-10",
            "confidence":  rec.get("confidence", 0.0),
            "risk":        risk,
            "explanation": _explain_discrepancy(rec["icd_code"], rec["description"], "missing"),
        })

    # Remaining unmatched human codes → extra
    for key in human_icd_map:
        if key in human_matched:
            continue
        rec  = human_icd_map[key]
        risk = _classify_revenue_risk(rec["icd_code"], "icd")
        extra_codes.append({
            "code":        rec["icd_code"],
            "description": rec.get("description", ""),
            "type":        "ICD-10",
            "risk":        risk,
            "explanation": _explain_discrepancy(rec["icd_code"], rec.get("description", ""), "extra"),
        })

    # ---- Compare CPT codes --------------------------------------------------
    all_cpt_keys = set(ai_cpt_map) | set(human_cpt_map)

    for key in all_cpt_keys:
        in_ai    = key in ai_cpt_map
        in_human = key in human_cpt_map

        if in_ai and in_human:
            rec = ai_cpt_map[key]
            matched.append({
                "code":        rec["cpt_code"],
                "description": rec["description"],
                "type":        "CPT",
                "confidence":  rec.get("confidence", 1.0),
            })

        elif in_ai and not in_human:
            rec  = ai_cpt_map[key]
            risk = _classify_revenue_risk(rec["cpt_code"], "cpt")
            missing_codes.append({
                "code":        rec["cpt_code"],
                "description": rec["description"],
                "type":        "CPT",
                "confidence":  rec.get("confidence", 0.0),
                "risk":        risk,
                "explanation": _explain_discrepancy(rec["cpt_code"], rec["description"], "missing"),
            })

        else:  # human only
            rec  = human_cpt_map[key]
            risk = _classify_revenue_risk(rec.get("cpt_code", ""), "cpt")
            extra_codes.append({
                "code":        rec.get("cpt_code", key),
                "description": rec.get("description", ""),
                "type":        "CPT",
                "risk":        risk,
                "explanation": _explain_discrepancy(
                    rec.get("cpt_code", key), rec.get("description", ""), "extra"
                ),
            })

    # ---- Overall risk level --------------------------------------------------
    all_risks = [c["risk"] for c in missing_codes + extra_codes]
    if "high" in all_risks:
        overall_risk = "high"
    elif "medium" in all_risks:
        overall_risk = "medium"
    elif all_risks:
        overall_risk = "low"
    else:
        overall_risk = "clear"

    # ---- Dynamic revenue impact calculation ---------------------------------
    rev = _calculate_revenue_impact(missing_codes, extra_codes)

    ACTION_MAP = {
        "high":   "URGENT — bill amendment required",
        "medium": "Review and correct before submission",
        "low":    "Flag for audit log",
        "clear":  "No action required",
    }

    # ---- Recommendations ----------------------------------------------------
    recommendations: list[str] = []

    if missing_codes:
        recommendations.append(
            f"{len(missing_codes)} potentially missing code(s) detected. "
            f"Estimated revenue loss: {rev['revenue_lost_estimate']}. "
            "Review and add to avoid undercoding."
        )
    if extra_codes:
        recommendations.append(
            f"{len(extra_codes)} potentially unsupported code(s) flagged. "
            f"Estimated penalty exposure: {rev['penalty_risk_estimate']}. "
            "Verify clinical documentation before submitting."
        )
    if ai_warnings:
        recommendations.extend(ai_warnings)
    if overall_risk == "clear":
        recommendations.append("All codes match. Claim appears accurate — safe to submit.")

    # ---- Audit log ----------------------------------------------------------
    audit_log = {
        "timestamp":          datetime.utcnow().isoformat() + "Z",
        "total_ai_codes":     len(ai_icd) + len(ai_cpt),
        "total_human_codes":  len(human_icd) + len(human_cpt),
        "matched_count":      len(matched),
        "missing_count":      len(missing_codes),
        "extra_count":        len(extra_codes),
        "overall_risk":       overall_risk,
    }

    return {
        "summary": {
            "matched":              len(matched),
            "missing_count":        len(missing_codes),
            "extra_count":          len(extra_codes),
            "total_discrepancies":  len(missing_codes) + len(extra_codes),
        },
        "matched":                  matched,
        "missing_codes":            missing_codes,
        "extra_codes":              extra_codes,
        "risk_level":               overall_risk,
        "revenue_risk":             rev["range"],
        "revenue_lost_estimate":    rev["revenue_lost_estimate"],
        "penalty_risk_estimate":    rev["penalty_risk_estimate"],
        "total_at_risk":            rev["total_at_risk"],
        "action_required":          ACTION_MAP.get(overall_risk, "Review"),
        "recommendations":          recommendations,
        "audit_log":                audit_log,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ai_output = {
        "icd_codes": [
            {"icd_code": "J18.9", "description": "Pneumonia, unspecified",       "confidence": 0.91},
            {"icd_code": "I10",   "description": "Essential hypertension",        "confidence": 0.88},
            {"icd_code": "E11.9", "description": "Type 2 diabetes, unspecified",  "confidence": 0.85},
        ],
        "cpt_codes": [
            {"cpt_code": "85025", "description": "CBC with differential",         "confidence": 0.90},
            {"cpt_code": "71250", "description": "CT thorax",                     "confidence": 0.90},
        ],
        "warnings": [],
    }

    human_input = [
        {"icd_code": "J18.9", "description": "Pneumonia"},             # match
        {"icd_code": "I10",   "description": "Hypertension"},          # match
        # E11.9 missing — undercoding
        {"icd_code": "Z87.01","description": "History of pneumonia"},  # extra — upcoding risk
        {"cpt_code": "85025", "description": "Blood test"},            # match
        # 71250 missing
    ]

    result = audit_codes(ai_output, human_input)
    print("=== Audit Report ===")
    print(f"  Matched:  {result['summary']['matched']}")
    print(f"  Missing:  {result['summary']['missing_count']}")
    print(f"  Extra:    {result['summary']['extra_count']}")
    print(f"  Risk:     {result['risk_level'].upper()}")
    print(f"  Revenue:  {result['revenue_risk']}")
    print("\n  Recommendations:")
    for r in result["recommendations"]:
        print(f"    → {r}")
    print("\n  Missing codes:")
    for c in result["missing_codes"]:
        print(f"    [{c['code']}] {c['description']}  risk={c['risk']}")
    print("\n  Extra codes:")
    for c in result["extra_codes"]:
        print(f"    [{c['code']}] {c['description']}  risk={c['risk']}")