"""
Coding Logic Agent  v2  —  Improved Accuracy
---------------------------------------------
Key improvements over v1:
  1. Sends full sentence context to FAISS (not bare keyword)
     → "history of hypertension" matches I10, not I97.3
  2. Confidence threshold filter — codes below 0.50 go to manual_review list
  3. Symptom suppression — R-codes not generated when a diagnosis explains them
  4. Procedure deduplication already handled upstream in clinical_reader
  5. CPT.xlsx column auto-detection (works regardless of column name)
  6. top_k=3 from FAISS then pick best, with fallback to next candidate
"""

from __future__ import annotations

import re
import os
from typing import Optional
import pandas as pd

_cpt_df: Optional[pd.DataFrame] = None
_icd_engine_available: bool = True

try:
    from app.services.icd_semantic_engine import search_icd
except Exception:
    _icd_engine_available = False
    def search_icd(query: str, top_k: int = 3) -> list[dict]:
        return []


# ── CPT procedure map ─────────────────────────────────────────────────────
# Sorted longest-key-first so specific matches win over generic ones
CPT_PROCEDURE_MAP: dict[str, str] = {
    "ct scan of chest":        "71250",
    "ct scan of abdomen":      "74177",
    "ct scan of head":         "70450",
    "chest x-ray":             "71046",
    "mri of brain":            "70553",
    "mri of spine":            "72148",
    "abdominal ultrasound":    "76700",
    "cardiac catheterization": "93454",
    "arterial blood gas":      "82803",
    "liver function test":     "80076",
    "blood culture":           "87040",
    "blood test":              "85025",
    "urinalysis":              "81001",
    "lumbar puncture":         "62270",
    "spinal tap":              "62270",
    "echocardiogram":          "93306",
    "nuclear stress test":     "78452",
    "pet scan":                "78816",
    "colonoscopy":             "45378",
    "endoscopy":               "43239",
    "bronchoscopy":            "31622",
    "incision and drainage":   "10060",
    "appendectomy":            "44950",
    "cholecystectomy":         "47562",
    "laparoscopy":             "49320",
    "angioplasty":             "92920",
    "pacemaker insertion":     "33206",
    "stenting":                "92928",
    "cardioversion":           "92960",
    "defibrillation":          "92960",
    "intubation":              "31500",
    "biopsy":                  "88305",
    "debridement":             "97597",
    "wound care":              "97597",
    "amputation":              "27590",
    "catheterization":         "51702",
    "hemodialysis":            "90935",
    "dialysis":                "90935",
    "blood transfusion":       "86890",
    "transfusion":             "86890",
    "chemotherapy":            "96413",
    "radiation therapy":       "77385",
    "radiation":               "77385",
    "physical therapy":        "97110",
    "iv infusion":             "96365",
    "ct scan":                 "71250",
    "x-ray":                   "71046",
    "xray":                    "71046",
    "ultrasound":              "76700",
    "mri":                     "70553",
    "ecg":                     "93000",
    "surgery":                 "99999",   # generic — needs manual review
    "operation":               "99999",
    "therapy":                 "97110",
}

# Sort so longest keys are tried first
_SORTED_CPT_KEYS = sorted(CPT_PROCEDURE_MAP.keys(), key=len, reverse=True)


# ── Symptom → diagnosis coverage map ─────────────────────────────────────
# If a diagnosis is confirmed, these symptoms are already implied and
# should NOT generate separate R-codes
DIAGNOSIS_COVERS_SYMPTOMS: dict[str, list[str]] = {
    "pneumonia":             ["fever", "cough", "dyspnea", "shortness of breath"],
    "community acquired pneumonia": ["fever", "cough", "dyspnea", "shortness of breath"],
    "heart failure":         ["dyspnea", "shortness of breath", "edema", "swelling", "fatigue"],
    "myocardial infarction": ["chest pain", "pain", "dyspnea", "shortness of breath"],
    "urinary tract infection":["fever", "dysuria", "hematuria"],
    "sepsis":                ["fever", "tachycardia"],
    "chronic obstructive pulmonary disease": ["dyspnea", "cough", "shortness of breath"],
    "asthma":                ["dyspnea", "cough", "shortness of breath", "wheezing"],
    "deep vein thrombosis":  ["swelling", "edema", "pain"],
    "appendicitis":          ["abdominal pain", "pain", "nausea", "vomiting", "fever"],
    "gastroenteritis":       ["nausea", "vomiting", "diarrhea", "abdominal pain"],
    "diabetes":              ["polyuria", "polydipsia", "fatigue", "weight loss"],
    "type 2 diabetes":       ["polyuria", "polydipsia", "fatigue", "weight loss"],
}

# Confidence threshold — below this, code goes to manual_review instead of confirmed
CONFIDENCE_THRESHOLD = 0.50   # slightly lower — scispaCy terms have richer context

# ── Preferred code overrides ──────────────────────────────────────────────
# Checked before FAISS. Key = lowercase term or phrase fragment.
# scispaCy returns natural language so we also do fuzzy substring matching.
PREFERRED_ICD_CODES: dict[str, dict] = {
    # Cardiovascular
    "hypertension":                        {"icd_code": "I10",    "description": "Essential (primary) hypertension"},
    "essential hypertension":              {"icd_code": "I10",    "description": "Essential (primary) hypertension"},
    "coronary artery disease":             {"icd_code": "I25.10", "description": "Atherosclerotic heart disease of native coronary artery"},
    "heart failure":                       {"icd_code": "I50.9",  "description": "Heart failure, unspecified"},
    "congestive heart failure":            {"icd_code": "I50.9",  "description": "Heart failure, unspecified"},
    "decompensated heart failure":         {"icd_code": "I50.9",  "description": "Heart failure, unspecified"},
    "atrial fibrillation":                 {"icd_code": "I48.91", "description": "Unspecified atrial fibrillation"},
    "myocardial infarction":               {"icd_code": "I21.9",  "description": "Acute myocardial infarction, unspecified"},
    "deep vein thrombosis":                {"icd_code": "I82.401","description": "Acute DVT of unspecified deep veins of lower extremity"},
    "pulmonary embolism":                  {"icd_code": "I26.99", "description": "Other pulmonary embolism without acute cor pulmonale"},
    "thromboembolic":                      {"icd_code": "I82.401","description": "Venous thromboembolism"},
    "peripheral vascular disease":         {"icd_code": "I73.9",  "description": "Peripheral vascular disease, unspecified"},
    # Respiratory
    "pneumonia":                           {"icd_code": "J18.9",  "description": "Pneumonia, unspecified organism"},
    "community acquired pneumonia":        {"icd_code": "J18.9",  "description": "Pneumonia, unspecified organism"},
    "hospital acquired pneumonia":         {"icd_code": "J15.8",  "description": "Pneumonia due to other specified bacteria"},
    "hospital-acquired pneumonia":         {"icd_code": "J15.8",  "description": "Pneumonia due to other specified bacteria"},
    "lower lobe infiltrate":               {"icd_code": "J18.9",  "description": "Pneumonia, unspecified organism"},
    "lower lobe infiltrates":              {"icd_code": "J18.9",  "description": "Pneumonia, unspecified organism"},
    "bilateral infiltrate":                {"icd_code": "J18.9",  "description": "Pneumonia, unspecified organism"},
    "pulmonary infiltrate":                {"icd_code": "J18.9",  "description": "Pneumonia, unspecified organism"},
    "lung infiltrate":                     {"icd_code": "J18.9",  "description": "Pneumonia, unspecified organism"},
    "chronic obstructive pulmonary disease":{"icd_code": "J44.1", "description": "COPD with acute exacerbation"},
    "copd":                                {"icd_code": "J44.1",  "description": "COPD with acute exacerbation"},
    "asthma":                              {"icd_code": "J45.909","description": "Unspecified asthma, uncomplicated"},
    "pleural effusion":                    {"icd_code": "J90",    "description": "Pleural effusion, not elsewhere classified"},
    "respiratory failure":                 {"icd_code": "J96.90", "description": "Respiratory failure, unspecified"},
    # Endocrine / metabolic
    "type 2 diabetes":                     {"icd_code": "E11.9",  "description": "Type 2 diabetes mellitus without complications"},
    "type 1 diabetes":                     {"icd_code": "E10.9",  "description": "Type 1 diabetes mellitus without complications"},
    "diabetes mellitus":                   {"icd_code": "E11.9",  "description": "Type 2 diabetes mellitus without complications"},
    "diabetes":                            {"icd_code": "E11.9",  "description": "Type 2 diabetes mellitus without complications"},
    "hyperglycaemia":                      {"icd_code": "E11.9",  "description": "Type 2 diabetes mellitus without complications"},
    "hyperglycemia":                       {"icd_code": "E11.9",  "description": "Type 2 diabetes mellitus without complications"},
    "poorly controlled hyperglycaemia":    {"icd_code": "E11.65", "description": "Type 2 diabetes with hyperglycemia"},
    "poorly controlled hyperglycemia":     {"icd_code": "E11.65", "description": "Type 2 diabetes with hyperglycemia"},
    "poorly controlled diabetes":          {"icd_code": "E11.65", "description": "Type 2 diabetes with hyperglycemia"},
    "diabetic ketoacidosis":               {"icd_code": "E11.10", "description": "Type 2 diabetes with ketoacidosis without coma"},
    "hypothyroidism":                      {"icd_code": "E03.9",  "description": "Hypothyroidism, unspecified"},
    "hyperthyroidism":                     {"icd_code": "E05.90", "description": "Thyrotoxicosis, unspecified, without thyrotoxic crisis"},
    "obesity":                             {"icd_code": "E66.9",  "description": "Obesity, unspecified"},
    "hyperlipidemia":                      {"icd_code": "E78.5",  "description": "Hyperlipidemia, unspecified"},
    "hyperlipidaemia":                     {"icd_code": "E78.5",  "description": "Hyperlipidemia, unspecified"},
    "dehydration":                         {"icd_code": "E86.0",  "description": "Dehydration"},
    "hyponatremia":                        {"icd_code": "E87.1",  "description": "Hypo-osmolality and hyponatraemia"},
    "hyperkalemia":                        {"icd_code": "E87.5",  "description": "Hyperkalaemia"},
    # Renal
    "chronic kidney disease":              {"icd_code": "N18.9",  "description": "Chronic kidney disease, unspecified"},
    "chronic renal impairment":            {"icd_code": "N18.9",  "description": "Chronic kidney disease, unspecified"},
    "chronic renal failure":               {"icd_code": "N18.9",  "description": "Chronic kidney disease, unspecified"},
    "stage 3 chronic renal":               {"icd_code": "N18.3",  "description": "Chronic kidney disease, stage 3"},
    "stage 3 ckd":                         {"icd_code": "N18.3",  "description": "Chronic kidney disease, stage 3"},
    "stage 4 chronic renal":               {"icd_code": "N18.4",  "description": "Chronic kidney disease, stage 4"},
    "stage 5 chronic renal":               {"icd_code": "N18.5",  "description": "Chronic kidney disease, stage 5"},
    "acute kidney injury":                 {"icd_code": "N17.9",  "description": "Acute kidney failure, unspecified"},
    "acute renal failure":                 {"icd_code": "N17.9",  "description": "Acute kidney failure, unspecified"},
    "urinary tract infection":             {"icd_code": "N39.0",  "description": "Urinary tract infection, site not specified"},
    "nephrotic syndrome":                  {"icd_code": "N04.9",  "description": "Nephrotic syndrome with unspecified morphological changes"},
    # Neurological
    "stroke":                              {"icd_code": "I63.9",  "description": "Cerebral infarction, unspecified"},
    "cerebral infarction":                 {"icd_code": "I63.9",  "description": "Cerebral infarction, unspecified"},
    "transient ischemic attack":           {"icd_code": "G45.9",  "description": "Transient cerebral ischaemic attack, unspecified"},
    "transient ischaemic attack":          {"icd_code": "G45.9",  "description": "Transient cerebral ischaemic attack, unspecified"},
    "seizure":                             {"icd_code": "R56.9",  "description": "Unspecified convulsions"},
    "epilepsy":                            {"icd_code": "G40.909","description": "Epilepsy, unspecified, not intractable"},
    "dementia":                            {"icd_code": "F03.90", "description": "Unspecified dementia without behavioral disturbance"},
    "altered mental status":               {"icd_code": "R41.3",  "description": "Other amnesia and altered awareness"},
    # Infectious
    "sepsis":                              {"icd_code": "A41.9",  "description": "Sepsis, unspecified organism"},
    "bacteraemia":                         {"icd_code": "A49.9",  "description": "Bacterial infection, unspecified"},
    "bacteremia":                          {"icd_code": "A49.9",  "description": "Bacterial infection, unspecified"},
    "infection":                           {"icd_code": "A49.9",  "description": "Bacterial infection, unspecified"},
    "cellulitis":                          {"icd_code": "L03.90", "description": "Cellulitis, unspecified"},
    "cholera":                             {"icd_code": "A00.9",  "description": "Cholera, unspecified"},
    "typhoid":                             {"icd_code": "A01.00", "description": "Typhoid fever, unspecified"},
    # GI
    "cirrhosis":                           {"icd_code": "K74.60", "description": "Unspecified cirrhosis of liver"},
    "gastrointestinal bleeding":           {"icd_code": "K92.2",  "description": "Gastrointestinal haemorrhage, unspecified"},
    "peptic ulcer":                        {"icd_code": "K27.9",  "description": "Peptic ulcer, unspecified"},
    "pancreatitis":                        {"icd_code": "K85.9",  "description": "Acute pancreatitis, unspecified"},
    "appendicitis":                        {"icd_code": "K37",    "description": "Unspecified appendicitis"},
    # Musculoskeletal / oncology / other
    "fracture":                            {"icd_code": "M84.40", "description": "Pathological fracture, unspecified site"},
    "anaemia":                             {"icd_code": "D64.9",  "description": "Anaemia, unspecified"},
    "anemia":                              {"icd_code": "D64.9",  "description": "Anaemia, unspecified"},
    "cancer":                              {"icd_code": "C80.1",  "description": "Malignant neoplasm, unspecified"},
    "malignancy":                          {"icd_code": "C80.1",  "description": "Malignant neoplasm, unspecified"},
    # Symptoms
    "fever":                               {"icd_code": "R50.9",  "description": "Fever, unspecified"},
    "chest pain":                          {"icd_code": "R07.9",  "description": "Chest pain, unspecified"},
    "shortness of breath":                 {"icd_code": "R06.09", "description": "Other forms of dyspnoea"},
    "dyspnoea":                            {"icd_code": "R06.09", "description": "Other forms of dyspnoea"},
    "dyspnea":                             {"icd_code": "R06.09", "description": "Other forms of dyspnoea"},
}

# ── Fuzzy preferred map lookup ────────────────────────────────────────────
# scispaCy returns natural language that may not exactly match map keys.
# We do substring matching in both directions so:
#   "bilateral lower lobe infiltrates" → matches "lower lobe infiltrates" ✓
#   "poorly controlled hyperglycaemia" → matches exactly ✓
#   "stage 3 chronic renal impairment" → matches "stage 3 chronic renal" ✓

# Pre-sort keys longest-first so the most specific match wins
_PREFERRED_KEYS_SORTED = sorted(PREFERRED_ICD_CODES.keys(), key=len, reverse=True)


def _lookup_preferred(term: str) -> Optional[dict]:
    """
    Find the best preferred code for a diagnosis term.
    Tries in order:
      1. Exact match on the full term
      2. Any preferred key is a substring of the term (longest key wins)
      3. The term is a substring of a preferred key
      4. Full word-overlap — ALL key words must appear in the term
         (threshold=1.0 prevents cross-condition false matches)
    """
    term_lower = term.lower().strip()

    # 1. Exact match
    if term_lower in PREFERRED_ICD_CODES:
        return PREFERRED_ICD_CODES[term_lower]

    # 2. Term contains a preferred key (most specific wins — keys sorted longest-first)
    for key in _PREFERRED_KEYS_SORTED:
        if key in term_lower:
            return PREFERRED_ICD_CODES[key]

    # 3. A preferred key contains the term (term is more specific than key)
    for key in _PREFERRED_KEYS_SORTED:
        if term_lower in key and len(term_lower) >= 5:
            return PREFERRED_ICD_CODES[key]

    # 4. Full word-overlap — require ALL words in the key to appear in the term.
    # This prevents "hyperglycaemia" matching "stage 3 chronic renal" because
    # none of {stage, 3, chronic, renal} appear in "hyperglycaemia".
    # Only use multi-word keys (single-word keys handled above via substring).
    term_words = set(term_lower.split())
    for key in _PREFERRED_KEYS_SORTED:
        key_words = set(key.split())
        if len(key_words) < 2:
            continue
        # All key words must be present in the term
        if key_words.issubset(term_words):
            return PREFERRED_ICD_CODES[key]

    return None


def _split_long_entity(entity_text: str) -> list[str]:
    """
    scispaCy sometimes returns very long spans that contain multiple conditions.
    e.g. "bilateral lower lobe infiltrates consistent with hospital-acquired pneumonia"
    Split on connectors and return the parts for individual lookup.
    """
    # Connectors that separate distinct conditions within one span
    split_patterns = [
        r"\bconsistent with\b",
        r"\bsecondary to\b",
        r"\bdue to\b",
        r"\bwith\b",
        r"\band\b",
        r"\bor\b",
        r",",
    ]
    parts = [entity_text]
    for pattern in split_patterns:
        new_parts = []
        for part in parts:
            split = re.split(pattern, part, flags=re.IGNORECASE)
            new_parts.extend(split)
        parts = new_parts

    # Clean and filter
    cleaned = []
    for p in parts:
        p = p.strip().strip(".,;:")
        if len(p) >= 5:   # skip fragments too short to be a diagnosis
            cleaned.append(p)
    return cleaned


# ── CPT lookup ────────────────────────────────────────────────────────────

def _load_cpt_xlsx(path: str = "app/data/CPT.xlsx") -> Optional[pd.DataFrame]:
    global _cpt_df
    if _cpt_df is not None:
        return _cpt_df
    try:
        # Try with header row first, then without
        try:
            df = pd.read_excel(path, dtype=str)
        except Exception:
            df = pd.read_excel(path, header=None, dtype=str)

        df.columns = [str(c).strip() for c in df.columns]

        # Auto-detect code and description columns
        code_col = next(
            (c for c in df.columns if "code" in c.lower()), df.columns[0]
        )
        desc_col = next(
            (c for c in df.columns if "desc" in c.lower() or "name" in c.lower()),
            df.columns[1] if len(df.columns) > 1 else df.columns[0],
        )

        df = df[[code_col, desc_col]].dropna()
        df.columns = ["code", "description"]
        df["code"] = df["code"].astype(str).str.strip()
        df["description_lower"] = df["description"].str.lower().str.strip()
        _cpt_df = df
        return _cpt_df
    except Exception as e:
        print(f"[CodingLogic] CPT.xlsx load failed: {e}")
        return None


def _lookup_cpt_from_xlsx(keyword: str) -> Optional[dict]:
    df = _load_cpt_xlsx()
    if df is None:
        return None
    kw = keyword.lower().strip()
    matches = df[df["description_lower"].str.contains(re.escape(kw), na=False)]
    if matches.empty:
        first_word = kw.split()[0] if kw.split() else kw
        matches = df[df["description_lower"].str.contains(re.escape(first_word), na=False)]
    if not matches.empty:
        row = matches.iloc[0]
        return {"cpt_code": row["code"], "description": str(row["description"])}
    return None


def _get_cpt_code(procedure: str) -> dict:
    proc_lower = procedure.lower().strip()
    # Try sorted keys longest-first
    for kw in _SORTED_CPT_KEYS:
        if kw in proc_lower:
            code = CPT_PROCEDURE_MAP[kw]
            return {"cpt_code": code, "description": procedure, "source": "builtin"}
    # Fall back to xlsx
    result = _lookup_cpt_from_xlsx(proc_lower)
    if result:
        result["source"] = "CPT.xlsx"
        return result
    return {"cpt_code": "UNKNOWN", "description": procedure, "source": "unmatched"}


# ── Confidence scoring ────────────────────────────────────────────────────

def _score_confidence(similarity: Optional[float], source: str) -> float:
    """
    Return confidence [0.0 – 1.0].

    For semantic results, FAISS IndexFlatIP on normalised vectors returns
    cosine SIMILARITY directly (not distance). 1.0 = perfect match, 0.0 = no relation.
    Typical good ICD matches score 0.60 – 0.90.
    """
    if source == "semantic" and similarity is not None:
        return max(0.0, min(1.0, round(float(similarity), 2)))
    if source == "builtin":
        return 0.90
    if source == "CPT.xlsx":
        return 0.75
    return 0.40


# ── Symptom suppression ───────────────────────────────────────────────────

def _symptoms_covered_by_diagnoses(
    symptoms: list[str], diagnoses: list[str]
) -> set[str]:
    """Return the set of symptoms already explained by confirmed diagnoses."""
    covered = set()
    for diag in diagnoses:
        diag_lower = diag.lower()
        for diag_key, covered_syms in DIAGNOSIS_COVERS_SYMPTOMS.items():
            if diag_key in diag_lower or diag_lower in diag_key:
                for sym in covered_syms:
                    if any(sym in s.lower() for s in symptoms):
                        covered.add(sym)
    return covered


# ── Main generation ───────────────────────────────────────────────────────

def generate_icd_codes(extracted_entities: dict) -> dict:
    """
    Generate ICD-10 and CPT codes from extracted medical entities.

    Returns:
        {
          "icd_codes":     confirmed codes (confidence >= threshold)
          "cpt_codes":     procedure codes
          "manual_review": codes that need human verification (low confidence)
          "warnings":      CMS rule flags
        }
    """
    diagnoses  = extracted_entities.get("diagnoses", [])
    procedures = extracted_entities.get("procedures", [])
    symptoms   = extracted_entities.get("symptoms", [])
    negated    = extracted_entities.get("negated", [])
    # v2: use full sentence context for FAISS queries
    contexts   = extracted_entities.get("sentence_contexts", {})

    icd_confirmed:   list[dict] = []
    icd_manual:      list[dict] = []
    cpt_results:     list[dict] = []
    warnings:        list[str]  = []
    preferred_terms_used: set[str] = set()   # tracks which terms used override map

    # ── ICD-10 for diagnoses ──────────────────────────────────────────────
    # Track confirmed ICD codes to avoid duplicates
    confirmed_icd_codes: set[str] = set()

    for diagnosis in diagnoses:
        diag_lower = diagnosis.lower().strip()

        # ── Step 1: Check preferred override map ─────────────────────────
        # First try the full entity, then try splitting it into components.
        # scispaCy often returns long spans like:
        # "bilateral lower lobe infiltrates consistent with hospital-acquired pneumonia"
        # Splitting on "consistent with" gives us "hospital-acquired pneumonia" → J15.8

        override = _lookup_preferred(diag_lower)

        if not override:
            # Try splitting the long entity into component conditions
            parts = _split_long_entity(diag_lower)
            if len(parts) > 1:
                for part in parts:
                    override = _lookup_preferred(part)
                    if override:
                        # Use the part that matched as the effective term label
                        diag_lower = part
                        break

        if override:
            code_norm = override["icd_code"].replace(".", "").upper()
            if code_norm not in confirmed_icd_codes:
                icd_confirmed.append({
                    "icd_code":    override["icd_code"],
                    "description": override["description"],
                    "term":        diagnosis,
                    "confidence":  0.97,
                })
                confirmed_icd_codes.add(code_norm)
                preferred_terms_used.add(diag_lower)

            # Also check remaining parts for additional codes
            # e.g. the same span may contain TWO distinct conditions
            parts = _split_long_entity(diagnosis.lower().strip())
            for part in parts:
                if part == diag_lower:
                    continue
                extra_override = _lookup_preferred(part)
                if extra_override:
                    extra_norm = extra_override["icd_code"].replace(".", "").upper()
                    if extra_norm not in confirmed_icd_codes:
                        icd_confirmed.append({
                            "icd_code":    extra_override["icd_code"],
                            "description": extra_override["description"],
                            "term":        part,
                            "confidence":  0.97,
                        })
                        confirmed_icd_codes.add(extra_norm)
                        preferred_terms_used.add(part)
            continue

        # ── Step 2: FAISS semantic search for everything else ─────────────
        sentence = contexts.get(diagnosis, diagnosis)
        if diag_lower in sentence.lower():
            search_query = f"{diagnosis}: {sentence}"
        else:
            search_query = diagnosis

        if _icd_engine_available:
            candidates = search_icd(search_query, top_k=5)
        else:
            candidates = []

        matched = False
        for candidate in candidates:
            code = candidate["icd_code"].replace(".", "").upper()
            if code in confirmed_icd_codes:
                continue
            confidence = _score_confidence(candidate.get("similarity"), "semantic")
            record = {
                "icd_code":    candidate["icd_code"],
                "description": candidate["description"],
                "term":        diagnosis,
                "confidence":  confidence,
            }
            if confidence >= CONFIDENCE_THRESHOLD:
                icd_confirmed.append(record)
                confirmed_icd_codes.add(code)
                matched = True
                break
            else:
                record["note"] = f"Low confidence ({confidence}) — verify manually"
                icd_manual.append(record)
                matched = True
                break

        if not matched:
            icd_manual.append({
                "icd_code":    "UNRESOLVED",
                "description": f"No match found for: {diagnosis}",
                "term":        diagnosis,
                "confidence":  0.0,
                "note":        "No ICD-10 candidate found — manual coding required",
            })
            warnings.append(f"No ICD-10 match for '{diagnosis}' — needs manual review.")

    # ── ICD-10 for symptoms (only if NOT covered by a diagnosis) ──────────
    covered_symptoms = _symptoms_covered_by_diagnoses(symptoms, diagnoses)

    for symptom in symptoms:
        # Skip if a confirmed diagnosis already explains this symptom
        if any(symptom.lower() in cs or cs in symptom.lower()
               for cs in covered_symptoms):
            continue
        # Skip if already represented in confirmed diagnoses
        already_coded = any(
            symptom.lower() in r.get("term", "").lower()
            for r in icd_confirmed
        )
        if already_coded:
            continue

        search_query = contexts.get(symptom, symptom)
        if diagnosis.lower() in search_query.lower() if diagnoses else False:
            search_query = f"{symptom}: {search_query}"
        if _icd_engine_available:
            candidates = search_icd(search_query, top_k=3)
            for best in candidates:
                code = best["icd_code"].replace(".", "").upper()
                if code in confirmed_icd_codes:
                    continue
                confidence = _score_confidence(best.get("similarity"), "semantic")
                record = {
                    "icd_code":    best["icd_code"],
                    "description": best["description"],
                    "term":        symptom,
                    "confidence":  round(confidence * 0.9, 2),
                    "note":        "Symptom code — check if diagnosis code is more appropriate",
                }
                if confidence >= CONFIDENCE_THRESHOLD:
                    icd_confirmed.append(record)
                    confirmed_icd_codes.add(code)
                else:
                    icd_manual.append(record)
                break

    # ── CMS rule: >25 diagnosis codes ────────────────────────────────────
    if len(icd_confirmed) > 25:
        warnings.append("More than 25 ICD-10 codes — review for specificity.")

    # ── CMS rule: negated entity accidentally coded ───────────────────────
    for neg in negated:
        for r in icd_confirmed:
            if neg.lower() in r.get("term", "").lower():
                warnings.append(
                    f"WARNING: Negated finding '{neg}' may be coded as '{r['icd_code']}' — verify."
                )

    # ── CPT codes for procedures ──────────────────────────────────────────
    for procedure in procedures:
        cpt = _get_cpt_code(procedure)
        cpt["term"]       = procedure
        cpt["confidence"] = _score_confidence(None, cpt.get("source", "unmatched"))
        if cpt["cpt_code"] == "99999":
            cpt["note"] = "Generic surgical code — specify exact procedure for accurate CPT"
            warnings.append(f"Procedure '{procedure}' needs a specific CPT code — generic used.")
        cpt_results.append(cpt)

    return {
        "icd_codes":      icd_confirmed,
        "cpt_codes":      cpt_results,
        "manual_review":  icd_manual,
        "warnings":       warnings,
        # Pass preferred_terms set so the explainability engine knows
        # which codes came from the override map vs FAISS
        "_preferred_terms": preferred_terms_used,
    }


generate_codes = generate_icd_codes


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "diagnoses":  ["myocardial infarction", "hypertension", "type 2 diabetes"],
        "procedures": ["ecg", "ct scan of chest", "iv infusion", "angioplasty"],
        "symptoms":   ["chest pain", "dyspnea"],
        "negated":    ["cancer"],
        "sentence_contexts": {
            "myocardial infarction": "Patient presented with acute myocardial infarction.",
            "hypertension":          "History of hypertension and type 2 diabetes.",
            "type 2 diabetes":       "History of hypertension and type 2 diabetes.",
            "chest pain":            "Patient presented with chest pain and dyspnea.",
            "dyspnea":               "Patient presented with chest pain and dyspnea.",
        },
    }
    result = generate_icd_codes(sample)
    print(f"\nConfirmed ICD-10 ({len(result['icd_codes'])}):")
    for c in result["icd_codes"]:
        print(f"  [{c['icd_code']}] {c['description'][:50]}  conf={c['confidence']}")
    print(f"\nManual review ({len(result['manual_review'])}):")
    for c in result["manual_review"]:
        print(f"  [{c['icd_code']}] {c['description'][:50]}  conf={c['confidence']}")
    print(f"\nCPT ({len(result['cpt_codes'])}):")
    for c in result["cpt_codes"]:
        print(f"  [{c['cpt_code']}] {c['description']}")
    if result["warnings"]:
        print(f"\nWarnings:")
        for w in result["warnings"]:
            print(f"  ⚠  {w}")