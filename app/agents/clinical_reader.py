"""
Clinical Reader Agent  v3  —  BioBERT / scispaCy NER Upgrade
-------------------------------------------------------------
NER architecture (layered, graceful fallback):

  Layer 1 — scispaCy medical NER  (best accuracy)
    Model: en_ner_bc5cdr_md
    Catches: "bilateral lower lobe infiltrates", "poorly controlled glucose",
             "2+ pitting oedema", "elevated WBC", unusual phrasings
    Entity types: DISEASE, CHEMICAL
    Install: pip install scispacy
             pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

  Layer 2 — Keyword matcher  (always runs, catches procedures + fills gaps)
    The keyword lists from v2 still run in all cases.
    scispaCy handles free-text diagnoses; keywords handle procedures and
    structured abbreviations that scispaCy doesn't classify.

  Layer 3 — Abbreviation expansion  (runs first, always)
    Same ordered regex map from v2.

Why this design:
  - Never breaks if scispaCy is not installed (falls back gracefully)
  - scispaCy catches what keywords miss (unusual phrasings, misspellings,
    compound conditions, severity modifiers)
  - Keywords catch what scispaCy misses (procedures, lab names, device names)
  - Combined output is deduplicated before returning
"""

from __future__ import annotations

import re
import spacy
from typing import Optional

# ── NER model loading ─────────────────────────────────────────────────────────
# We try models in order of preference, falling back gracefully.

_nlp_main   = None   # scispaCy medical NER (best)
_nlp_struct = None   # standard spaCy (sentence splitting + fallback)
_scispacy_available = False

def _load_models() -> None:
    global _nlp_main, _nlp_struct, _scispacy_available

    # Always load a standard spaCy model for sentence splitting
    for model_name in ("en_core_web_md", "en_core_web_sm"):
        try:
            _nlp_struct = spacy.load(model_name)
            break
        except OSError:
            continue

    # Try scispaCy medical NER model
    for sci_model in ("en_ner_bc5cdr_md", "en_core_sci_md", "en_core_sci_sm"):
        try:
            _nlp_main = spacy.load(sci_model)
            _scispacy_available = True
            print(f"[ClinicalReader] scispaCy model loaded: {sci_model}")
            break
        except OSError:
            continue

    if not _scispacy_available:
        print(
            "[ClinicalReader] scispaCy not available — using keyword NER.\n"
            "  To upgrade: pip install scispacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
        )

_load_models()


# ── scispaCy entity → our category mapping ────────────────────────────────────
# BC5CDR model labels: DISEASE, CHEMICAL
# We treat DISEASE as diagnoses/symptoms, CHEMICAL as medications (ignored for coding)

SCISPACY_DIAGNOSIS_LABELS = {"DISEASE"}

# scispaCy catches a lot of drug names as CHEMICAL — we don't want those
# as diagnoses, so we filter them out
DRUG_INDICATORS = {
    "mg", "mcg", "ml", "tablet", "capsule", "injection", "solution",
    "dose", "infusion", "antibiotic", "drug", "medication", "therapy",
    "treatment", "administered", "prescribed",
}


def _is_drug_mention(text: str, sentence: str) -> bool:
    """Heuristic: is this entity likely a drug rather than a disease?"""
    text_lower = text.lower()
    sent_lower = sentence.lower()
    # If the entity appears right before mg/dose/administration keywords
    idx = sent_lower.find(text_lower)
    if idx == -1:
        return False
    after = sent_lower[idx + len(text_lower):idx + len(text_lower) + 30]
    return any(d in after for d in DRUG_INDICATORS)


# ── Abbreviation expansion (ordered list — specific before generic) ───────────

ABBREVIATION_MAP: list[tuple[str, str]] = [
    # ── Specific diabetes types first (before generic DM) ──
    (r"\bDM2\b",   "type 2 diabetes"),
    (r"\bDM1\b",   "type 1 diabetes"),
    (r"\bT2DM\b",  "type 2 diabetes"),
    (r"\bT1DM\b",  "type 1 diabetes"),
    (r"\bDM\b",    "diabetes mellitus"),
    # ── Cardiovascular ──
    (r"\bSTEMI\b", "ST-elevation myocardial infarction"),
    (r"\bNSTEMI\b","non-ST-elevation myocardial infarction"),
    (r"\bMI\b",    "myocardial infarction"),
    (r"\bHTN\b",   "hypertension"),
    (r"\bCAD\b",   "coronary artery disease"),
    (r"\bCHF\b",   "congestive heart failure"),
    (r"\bHF\b",    "heart failure"),
    (r"\bAF\b",    "atrial fibrillation"),
    (r"\bAfib\b",  "atrial fibrillation"),
    (r"\bDVT\b",   "deep vein thrombosis"),
    (r"\bPE\b",    "pulmonary embolism"),
    (r"\bPVD\b",   "peripheral vascular disease"),
    # ── Respiratory ──
    (r"\bSOB\b",   "shortness of breath"),
    (r"\bDyspnea\b","dyspnea"),
    (r"\bCAP\b",   "community acquired pneumonia"),
    (r"\bHAP\b",   "hospital acquired pneumonia"),
    (r"\bCOPD\b",  "chronic obstructive pulmonary disease"),
    # ── Renal / metabolic ──
    (r"\bCKD\b",   "chronic kidney disease"),
    (r"\bAKI\b",   "acute kidney injury"),
    (r"\bUTI\b",   "urinary tract infection"),
    (r"\bHLD\b",   "hyperlipidemia"),
    (r"\bOSA\b",   "obstructive sleep apnea"),
    (r"\bBPH\b",   "benign prostatic hyperplasia"),
    # ── Neurological / other ──
    (r"\bCVA\b",   "stroke"),
    (r"\bTIA\b",   "transient ischemic attack"),
    (r"\bMS\b",    "multiple sclerosis"),
    (r"\bRA\b",    "rheumatoid arthritis"),
    (r"\bSLE\b",   "systemic lupus erythematosus"),
    (r"\bGERD\b",  "gastroesophageal reflux disease"),
    (r"\bIBD\b",   "inflammatory bowel disease"),
    # ── Symptoms ──
    (r"\bCP\b",    "chest pain"),
    (r"\bN/V\b",   "nausea and vomiting"),
    (r"\bN&V\b",   "nausea and vomiting"),
    (r"\bHA\b",    "headache"),
    (r"\bLOC\b",   "loss of consciousness"),
    (r"\bAMS\b",   "altered mental status"),
    # ── Procedures (longer patterns first) ──
    (r"\bCXR\b",   "chest x-ray"),
    (r"\bCBC\b",   "blood test"),
    (r"\bBMP\b",   "blood test"),
    (r"\bCMP\b",   "blood test"),
    (r"\bECG\b",   "ecg"),
    (r"\bEKG\b",   "ecg"),
    (r"\bMRI\b",   "mri"),
    (r"\bCT\b",    "ct scan"),
    (r"\bUS\b",    "ultrasound"),
    (r"\bLP\b",    "lumbar puncture"),
    (r"\bI&D\b",   "incision and drainage"),
    (r"\bLFT\b",   "liver function test"),
]


def expand_abbreviations(text: str) -> str:
    """Expand medical abbreviations in processing order (specific → generic)."""
    for pattern, replacement in ABBREVIATION_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Keyword fallback lists ────────────────────────────────────────────────────
# Used when scispaCy is not available, AND always for procedures
# (scispaCy doesn't classify procedures)

DIAGNOSIS_KEYWORDS: list[str] = sorted([
    "infection", "sepsis", "bacteremia", "cholera", "typhoid", "tuberculosis",
    "pneumonia", "community acquired pneumonia", "hospital acquired pneumonia",
    "bronchitis", "cellulitis", "appendicitis", "pancreatitis", "hepatitis",
    "gastroenteritis", "meningitis", "endocarditis", "osteomyelitis",
    "urinary tract infection", "pyelonephritis",
    "myocardial infarction", "st-elevation myocardial infarction",
    "non-st-elevation myocardial infarction", "heart failure",
    "congestive heart failure", "coronary artery disease",
    "atrial fibrillation", "deep vein thrombosis", "pulmonary embolism",
    "hypertension", "hypotension", "peripheral vascular disease",
    "cardiomyopathy", "arrhythmia", "angina",
    "chronic obstructive pulmonary disease", "asthma", "pleural effusion",
    "respiratory failure", "pulmonary fibrosis",
    "type 2 diabetes", "type 1 diabetes", "diabetes mellitus", "diabetes",
    "hyperglycemia", "hypoglycemia", "diabetic ketoacidosis",
    "hypothyroidism", "hyperthyroidism", "obesity", "malnutrition",
    "hyperlipidemia", "hyponatremia", "hyperkalemia", "hypokalemia",
    "chronic kidney disease", "acute kidney injury", "nephrotic syndrome",
    "stroke", "transient ischemic attack", "seizure", "epilepsy",
    "altered mental status", "dementia", "parkinson",
    "cirrhosis", "gastrointestinal bleeding", "peptic ulcer",
    "inflammatory bowel disease", "crohn", "colitis",
    "fracture", "arthritis", "rheumatoid arthritis", "osteoporosis",
    "cancer", "carcinoma", "tumor", "lymphoma", "leukemia",
    "anemia", "fever", "dehydration", "shock", "coagulopathy",
    "anxiety", "depression", "bipolar", "schizophrenia",
    "systemic lupus erythematosus", "multiple sclerosis",
    "obstructive sleep apnea", "benign prostatic hyperplasia",
    "gastroesophageal reflux disease",
], key=len, reverse=True)

PROCEDURE_KEYWORDS: list[str] = sorted([
    "ct scan of chest", "ct scan of abdomen", "ct scan of head",
    "ct scan", "chest x-ray", "x-ray", "xray",
    "mri of brain", "mri of spine", "mri",
    "abdominal ultrasound", "ultrasound", "echocardiogram",
    "nuclear stress test", "pet scan",
    "blood culture", "blood test", "urinalysis", "lumbar puncture",
    "liver function test", "arterial blood gas", "spinal tap",
    "ecg", "cardioversion", "defibrillation", "angioplasty",
    "cardiac catheterization", "stenting", "pacemaker insertion",
    "appendectomy", "cholecystectomy", "laparoscopy", "colonoscopy",
    "endoscopy", "bronchoscopy", "biopsy", "incision and drainage",
    "amputation", "debridement", "wound care",
    "iv infusion", "blood transfusion", "transfusion",
    "hemodialysis", "dialysis", "intubation", "ventilation",
    "chemotherapy", "radiation therapy", "radiation",
    "physical therapy", "catheterization",
    "surgery", "operation", "therapy",
], key=len, reverse=True)

SYMPTOM_KEYWORDS: list[str] = sorted([
    "shortness of breath", "chest pain", "altered mental status",
    "loss of consciousness", "weight loss", "weight gain",
    "nausea and vomiting", "abdominal pain", "back pain", "joint pain",
    "difficulty breathing",
    "pain", "fever", "cough", "fatigue", "nausea", "vomiting",
    "diarrhea", "dyspnea", "headache", "dizziness", "syncope",
    "edema", "swelling", "rash", "jaundice", "bleeding",
    "hematuria", "dysuria", "polyuria", "polydipsia",
    "anorexia", "confusion", "palpitations", "tachycardia",
    "bradycardia", "hypotension", "hypertension",
], key=len, reverse=True)

NEGATION_CUES: tuple[str, ...] = (
    "no history of ", "no evidence of ", "no signs of ",
    "no complaints of ", "no known ", "not consistent with ",
    "rules out ", "ruled out ", "negative for ",
    "denies ", "denied ", "without ", "absent ", "not ",
    "no ", "never ",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_negated(sentence: str, entity: str) -> bool:
    sentence_lower = sentence.lower()
    entity_lower   = entity.lower()
    idx = sentence_lower.find(entity_lower)
    if idx == -1:
        return False
    context_window = sentence_lower[max(0, idx - 100): idx]
    return any(cue in context_window for cue in NEGATION_CUES)


def _find_keywords(text: str, keyword_list: list[str]) -> list[str]:
    text_lower = text.lower()
    found = [kw for kw in keyword_list if kw in text_lower]
    # Remove substrings: if "ct scan" and "ct scan of chest" both match, keep longer
    return [
        kw for kw in found
        if not any(kw != other and kw in other for other in found)
    ]


def _normalise_entity(text: str) -> str:
    """Clean up scispaCy entity text for use as a diagnosis term."""
    text = text.strip().lower()
    # Remove leading articles
    text = re.sub(r"^(the|a|an)\s+", "", text)
    # Remove trailing punctuation
    text = text.rstrip(".,;:")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# ── scispaCy NER extraction ───────────────────────────────────────────────────

def _extract_with_scispacy(expanded_text: str, original_text: str) -> dict:
    """
    Use scispaCy's medical NER to extract disease entities.
    Returns dict with diagnoses, negated, and sentence_contexts.
    """
    doc_sci    = _nlp_main(expanded_text)
    doc_struct = _nlp_struct(expanded_text) if _nlp_struct else doc_sci

    # Build sentence lookup from the structural model (better sentence splitting)
    sentences = [sent.text for sent in doc_struct.sents]

    def _find_sentence(entity_text: str) -> str:
        """Find which sentence contains this entity."""
        entity_lower = entity_text.lower()
        for sent in sentences:
            if entity_lower in sent.lower():
                return sent.strip()
        return entity_text

    diagnoses_confirmed: list[str] = []
    diagnoses_negated:   list[str] = []
    sentence_contexts:   dict[str, str] = {}

    seen_entities: set[str] = set()

    for ent in doc_sci.ents:
        if ent.label_ not in SCISPACY_DIAGNOSIS_LABELS:
            continue

        entity_text = _normalise_entity(ent.text)
        if not entity_text or len(entity_text) < 3:
            continue
        if entity_text in seen_entities:
            continue
        seen_entities.add(entity_text)

        sentence = _find_sentence(ent.text)

        # Skip drug mentions misclassified as diseases
        if _is_drug_mention(ent.text, sentence):
            continue

        if _is_negated(sentence, ent.text):
            diagnoses_negated.append(entity_text)
        else:
            diagnoses_confirmed.append(entity_text)
            sentence_contexts[entity_text] = sentence

    return {
        "diagnoses_confirmed": diagnoses_confirmed,
        "diagnoses_negated":   diagnoses_negated,
        "sentence_contexts":   sentence_contexts,
    }


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_medical_entities(text: str) -> dict:
    """
    Parse a clinical note and return structured medical entities.

    Extraction pipeline:
      1. Expand abbreviations
      2. scispaCy NER for diagnoses (if available)
      3. Keyword matching for procedures (always) + diagnoses (fallback/supplement)
      4. Symptom extraction via keywords
      5. Global deduplication

    Returns:
        {
            "diagnoses":         list[str],
            "procedures":        list[str],
            "symptoms":          list[str],
            "negated":           list[str],
            "sentence_contexts": dict[str, str],
            "ner_method":        "scispacy" | "keyword",
            "raw_text":          str,
        }
    """
    # ── Step 1: Expand abbreviations ──────────────────────────────────────
    expanded_text = expand_abbreviations(text)

    # ── Step 2: Sentence tokenisation (always via standard spaCy) ─────────
    nlp_for_sents = _nlp_struct or _nlp_main
    if nlp_for_sents is None:
        # Absolute fallback: split on periods
        sentences = [s.strip() for s in expanded_text.split(".") if s.strip()]
        doc = None
    else:
        doc = nlp_for_sents(expanded_text)
        sentences = [sent.text for sent in doc.sents]

    diagnoses_confirmed: list[str] = []
    diagnoses_negated:   list[str] = []
    procedures_found:    list[str] = []
    symptoms_found:      list[str] = []
    sentence_contexts:   dict[str, str] = {}
    ner_method = "keyword"

    # ── Step 3: scispaCy NER (best path) ──────────────────────────────────
    if _scispacy_available and _nlp_main is not None:
        ner_method = "scispacy"
        sci_result = _extract_with_scispacy(expanded_text, text)

        diagnoses_confirmed = sci_result["diagnoses_confirmed"]
        diagnoses_negated   = sci_result["diagnoses_negated"]
        sentence_contexts.update(sci_result["sentence_contexts"])

        # Supplement with keywords ONLY for terms scispaCy clearly missed.
        # We check using simple word overlap to avoid spelling variant issues
        # (e.g. "hyperglycaemia" vs "hyperglycemia").
        sci_text = " ".join(diagnoses_confirmed + diagnoses_negated).lower()

        for sent_text in sentences:
            for kw in _find_keywords(sent_text, DIAGNOSIS_KEYWORDS):
                kw_norm = _normalise_entity(kw)
                kw_words = set(kw_norm.split())
                # Skip if ANY significant word from this keyword already appears
                # in what scispaCy found — avoids spelling variant duplicates
                already_covered = any(
                    w in sci_text for w in kw_words
                    if len(w) > 4  # only check substantive words
                )
                if not already_covered:
                    if _is_negated(sent_text, kw):
                        diagnoses_negated.append(kw_norm)
                    else:
                        diagnoses_confirmed.append(kw_norm)
                        sentence_contexts[kw_norm] = sent_text.strip()

    else:
        # ── Step 3 fallback: keyword-only NER ─────────────────────────────
        for sent_text in sentences:
            for kw in _find_keywords(sent_text, DIAGNOSIS_KEYWORDS):
                if _is_negated(sent_text, kw):
                    diagnoses_negated.append(kw)
                else:
                    diagnoses_confirmed.append(kw)
                    sentence_contexts[kw] = sent_text.strip()

    # ── Step 4: Procedures (always keyword — scispaCy doesn't handle these) ──
    for sent_text in sentences:
        for kw in _find_keywords(sent_text, PROCEDURE_KEYWORDS):
            procedures_found.append(kw)
            sentence_contexts[kw] = sent_text.strip()

    # ── Step 5: Symptoms (keyword — keep fast, usually supplementary) ─────
    for sent_text in sentences:
        for kw in _find_keywords(sent_text, SYMPTOM_KEYWORDS):
            if not _is_negated(sent_text, kw):
                symptoms_found.append(kw)
                if kw not in sentence_contexts:
                    sentence_contexts[kw] = sent_text.strip()

    # ── Step 6: Global deduplication ─────────────────────────────────────
    def dedup(items: list[str]) -> list[str]:
        seen = []
        items_lower = [i.lower() for i in items]
        for item in items:
            item_l = item.lower()
            if item_l in seen:
                continue
            # Skip if a longer version of this item is already in the list
            is_sub = any(
                item_l != other and item_l in other
                for other in items_lower
            )
            if not is_sub:
                seen.append(item_l)
        return sorted(seen)

    return {
        "diagnoses":         dedup(diagnoses_confirmed),
        "procedures":        dedup(procedures_found),
        "symptoms":          dedup(symptoms_found),
        "negated":           sorted(set(d.lower() for d in diagnoses_negated)),
        "sentence_contexts": sentence_contexts,
        "ner_method":        ner_method,
        "raw_text":          text,
    }


# Backwards-compatible alias
def extract_diagnosis(text: str) -> dict:
    return extract_medical_entities(text)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"scispaCy available: {_scispacy_available}")
    print(f"NER mode: {'scispaCy medical NER' if _scispacy_available else 'keyword fallback'}\n")

    notes = [
        # Standard abbreviation note
        "67M with SOB, CP and fever. Hx of HTN and DM2. No hx of cancer. "
        "CXR and CBC ordered. CT chest done. IV infusion of abx for CAP. No evidence of PE.",

        # Free-text that keyword NER would miss
        "Patient presents with bilateral lower lobe infiltrates consistent with "
        "hospital-acquired pneumonia. Background of poorly controlled hyperglycaemia "
        "and stage 3 chronic renal impairment. No prior DVT or thromboembolic events. "
        "ECG performed. CT pulmonary angiogram ordered to rule out PE.",

        # Complex multi-condition
        "78F admitted with acute decompensated heart failure, new onset AF with RVR. "
        "Known COPD on home oxygen. BNP markedly elevated. Echo and BMP ordered. "
        "Diuresis initiated. Denies chest pain or syncope.",
    ]

    for note in notes:
        print("─" * 60)
        print(f"Note: {note[:80]}...")
        result = extract_medical_entities(note)
        print(f"  Method:     {result['ner_method']}")
        print(f"  Diagnoses:  {result['diagnoses']}")
        print(f"  Procedures: {result['procedures']}")
        print(f"  Symptoms:   {result['symptoms']}")
        print(f"  Negated:    {result['negated']}")
        print()