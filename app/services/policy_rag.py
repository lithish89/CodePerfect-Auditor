"""
Payer Policy RAG Engine  —  app/services/policy_rag.py
-------------------------------------------------------
Retrieval-Augmented Generation (RAG) system for payer policy validation.

How it works:
  1. Policy rules are stored as text chunks in a ChromaDB vector store
  2. For each AI-generated code, we retrieve the most relevant policy rules
  3. Each code is validated against retrieved rules and flagged accordingly

Policy flag types:
  - PRIOR_AUTH_REQUIRED  : code needs pre-authorisation from payer
  - BUNDLED              : code is bundled into another code (cannot bill separately)
  - EXCLUDED             : payer does not cover this code
  - FREQUENCY_LIMIT      : code has a usage frequency restriction
  - DIAGNOSIS_REQUIRED   : code requires specific diagnosis documentation
  - APPROVED             : no policy issues found

Built-in policies (no setup needed):
  Uses a curated set of CMS Medicare and common payer rules embedded at startup.
  For production: load actual payer PDFs using load_policy_pdf().

Dependencies:
  pip install chromadb sentence-transformers
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import fitz  # PyMuPDF

# ── Lazy imports ──────────────────────────────────────────────────────────────
_chroma_client      = None
_policy_collection  = None
_embed_model        = None
_rag_ready          = False
_rag_error: Optional[str] = None

COLLECTION_NAME = "payer_policies"


# ── Built-in CMS / common payer policy rules ──────────────────────────────────
# These are simplified representations of real CMS LCD/NCD rules.
# Each entry: (rule_text, metadata)
# rule_text is what gets embedded and searched.
# metadata carries structured flags for the validator.

BUILTIN_POLICIES: list[tuple[str, dict]] = [
    # ── Prior authorisation rules ────────────────────────────────────────────
    (
        "MRI brain with contrast CPT 70553 requires prior authorisation for outpatient "
        "elective procedures. Emergency inpatient MRI is exempt. Payer: Medicare.",
        {"code": "70553", "flag": "PRIOR_AUTH_REQUIRED",
         "payer": "Medicare", "condition": "outpatient elective only",
         "source": "CMS LCD L35000"},
    ),
    (
        "CT pulmonary angiography CPT 71275 requires prior authorisation when performed "
        "outpatient without documented acute symptoms. ICD Z87 history codes insufficient.",
        {"code": "71275", "flag": "PRIOR_AUTH_REQUIRED",
         "payer": "General", "condition": "outpatient without acute symptoms",
         "source": "CMS NCD 220.1"},
    ),
    (
        "Chemotherapy administration CPT 96413 requires prior authorisation with "
        "oncology treatment plan documentation before claim submission.",
        {"code": "96413", "flag": "PRIOR_AUTH_REQUIRED",
         "payer": "General", "condition": "treatment plan required",
         "source": "CMS NCD 110.17"},
    ),
    (
        "Cardiac catheterisation CPT 93454 requires prior authorisation for elective "
        "procedures. Must document failed conservative management.",
        {"code": "93454", "flag": "PRIOR_AUTH_REQUIRED",
         "payer": "General", "condition": "elective — failed conservative management required",
         "source": "CMS LCD L34000"},
    ),
    (
        "Colonoscopy CPT 45378 diagnostic requires prior auth when performed within "
        "10 years of a previous screening colonoscopy in asymptomatic patients.",
        {"code": "45378", "flag": "PRIOR_AUTH_REQUIRED",
         "payer": "Medicare", "condition": "repeat within 10 years asymptomatic",
         "source": "CMS NCD 210.3"},
    ),
    (
        "Physical therapy CPT 97110 requires prior authorisation after 20 sessions "
        "per calendar year. Documentation of functional improvement required.",
        {"code": "97110", "flag": "FREQUENCY_LIMIT",
         "payer": "General", "condition": "max 20 sessions/year without auth",
         "source": "CMS LCD L33788"},
    ),
    # ── Bundling rules ───────────────────────────────────────────────────────
    (
        "ECG CPT 93000 is bundled into cardiac catheterisation CPT 93454. "
        "Cannot bill ECG separately on the same day as catheterisation.",
        {"code": "93000", "flag": "BUNDLED",
         "bundled_with": "93454", "payer": "General",
         "source": "CMS NCCI Edit"},
    ),
    (
        "Urinalysis CPT 81001 is bundled into comprehensive metabolic panel CPT 80053. "
        "Do not bill separately when CMP is billed on the same date of service.",
        {"code": "81001", "flag": "BUNDLED",
         "bundled_with": "80053", "payer": "General",
         "source": "CMS NCCI Edit"},
    ),
    (
        "Wound debridement CPT 97597 is bundled into surgical procedure codes when "
        "performed as part of the same operative session.",
        {"code": "97597", "flag": "BUNDLED",
         "bundled_with": "surgical procedure", "payer": "General",
         "source": "CMS NCCI Edit"},
    ),
    (
        "CBC blood test CPT 85025 is bundled into comprehensive metabolic panel 80053 "
        "and complete blood count with differential 85027. Cannot bill duplicate lab panels.",
        {"code": "85025", "flag": "BUNDLED",
         "bundled_with": "80053 / 85027", "payer": "General",
         "source": "CMS NCCI Edit"},
    ),
    # ── Diagnosis required ───────────────────────────────────────────────────
    (
        "Hemodialysis CPT 90935 requires ICD-10 diagnosis code N18.5 or N18.6 "
        "(end-stage renal disease) or N17.9 acute kidney failure for Medicare coverage.",
        {"code": "90935", "flag": "DIAGNOSIS_REQUIRED",
         "required_icd": ["N18.5", "N18.6", "N17.9"], "payer": "Medicare",
         "source": "CMS NCD 230.3"},
    ),
    (
        "IV infusion CPT 96365 for antibiotic therapy requires supporting diagnosis of "
        "active infection. ICD codes A or B chapter or J chapter required.",
        {"code": "96365", "flag": "DIAGNOSIS_REQUIRED",
         "required_icd_chapter": ["A", "B", "J"], "payer": "General",
         "source": "CMS LCD L36143"},
    ),
    (
        "Lumbar puncture CPT 62270 requires documentation of clinical indication. "
        "Acceptable diagnoses include meningitis, subarachnoid haemorrhage, or CNS infection.",
        {"code": "62270", "flag": "DIAGNOSIS_REQUIRED",
         "required_diagnoses": ["meningitis", "subarachnoid haemorrhage", "CNS infection"],
         "payer": "General", "source": "CMS LCD L35150"},
    ),
    # ── Frequency limits ─────────────────────────────────────────────────────
    (
        "Chest X-ray CPT 71046 limited to 1 per day for inpatient and "
        "1 per 12 months for routine outpatient screening under Medicare.",
        {"code": "71046", "flag": "FREQUENCY_LIMIT",
         "payer": "Medicare", "condition": "1/day inpatient, 1/12mo outpatient screening",
         "source": "CMS NCD 220.5"},
    ),
    (
        "ECG CPT 93000 covered once per year for routine preventive screening. "
        "Additional ECGs require documented cardiac symptoms or monitoring indication.",
        {"code": "93000", "flag": "FREQUENCY_LIMIT",
         "payer": "Medicare", "condition": "1/year screening; additional require symptoms",
         "source": "CMS NCD 20.15"},
    ),
    (
        "Blood culture CPT 87040 limited to 3 sets per episode of suspected sepsis. "
        "Repeat cultures require new clinical indication.",
        {"code": "87040", "flag": "FREQUENCY_LIMIT",
         "payer": "General", "condition": "max 3 sets per sepsis episode",
         "source": "CMS LCD L35103"},
    ),
    # ── Coverage / exclusion rules ───────────────────────────────────────────
    (
        "Intubation CPT 31500 performed in emergency department is covered under "
        "facility fee. Physician billing requires separate documentation of medical necessity.",
        {"code": "31500", "flag": "APPROVED",
         "payer": "General", "note": "ED facility fee — physician bills separately",
         "source": "CMS NCD"},
    ),
    (
        "Angioplasty CPT 92920 coronary must be supported by documented coronary artery "
        "disease ICD I25.xx. Pre-procedural stress test or angiography findings required.",
        {"code": "92920", "flag": "DIAGNOSIS_REQUIRED",
         "required_icd_prefix": "I25", "payer": "General",
         "source": "CMS NCD 240.4"},
    ),
    (
        "Echocardiogram CPT 93306 covered for documented cardiac symptoms, known "
        "structural heart disease, or pre-surgical evaluation. Not covered for routine screening.",
        {"code": "93306", "flag": "DIAGNOSIS_REQUIRED",
         "payer": "Medicare", "condition": "documented cardiac symptoms or known disease",
         "source": "CMS NCD 20.29"},
    ),
    # ── ICD-specific rules ───────────────────────────────────────────────────
    (
        "Z-codes (Z00-Z99) history and status codes cannot be used as primary diagnosis "
        "for inpatient admissions. Must be supported by an active condition code.",
        {"code": "Z", "flag": "DIAGNOSIS_REQUIRED",
         "payer": "General", "condition": "Z-codes cannot be primary inpatient diagnosis",
         "source": "ICD-10-CM Official Guidelines"},
    ),
    (
        "Unspecified codes ending in .9 or .90 should be avoided when a more specific "
        "code is available. Payers may reject or downcode unspecified claims.",
        {"code": "UNSPECIFIED", "flag": "DIAGNOSIS_REQUIRED",
         "payer": "General", "condition": "use most specific code available",
         "source": "ICD-10-CM Official Guidelines I.A.19"},
    ),
]


# ── Initialisation ────────────────────────────────────────────────────────────

def _init_rag() -> None:
    """Initialise ChromaDB, embed model, and load built-in policies."""
    global _chroma_client, _policy_collection, _embed_model, _rag_ready, _rag_error

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        import os

        # In-memory ChromaDB — no file setup needed
        _chroma_client     = chromadb.Client()
        _policy_collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        _embed_model = SentenceTransformer(
            os.environ.get("ICD_MODEL_NAME", "all-MiniLM-L6-v2")
        )

        # Load built-in policies if collection is empty
        if _policy_collection.count() == 0:
            _load_builtin_policies()

        _rag_ready = True
        print(f"[PolicyRAG] Ready — {_policy_collection.count()} policy rules loaded.")

    except ImportError as e:
        _rag_error = (
            f"chromadb not installed: {e}. "
            "Run: pip install chromadb"
        )
        print(f"[PolicyRAG] {_rag_error}")
    except Exception as e:
        _rag_error = str(e)
        print(f"[PolicyRAG] Initialisation failed: {e}")


def _load_builtin_policies() -> None:
    """Embed and store all built-in policy rules."""
    texts     = [p[0] for p in BUILTIN_POLICIES]
    metadatas = [p[1] for p in BUILTIN_POLICIES]
    ids       = [
        hashlib.md5(t.encode()).hexdigest()[:16]
        for t in texts
    ]
    embeddings = _embed_model.encode(texts, normalize_embeddings=True).tolist()
    _policy_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


# ── PDF ingestion (for real payer policy documents) ───────────────────────────

def load_policy_pdf(pdf_path: str, payer_name: str, chunk_size: int = 400) -> int:
    """
    Load a payer policy PDF into the vector store.

    Chunks the PDF text into overlapping windows, embeds each chunk,
    and stores with payer metadata. Returns number of chunks added.

    Usage:
        from app.services.policy_rag import load_policy_pdf
        load_policy_pdf("policies/medicare_lcd.pdf", "Medicare")
    """
    if not _rag_ready:
        _init_rag()
    if not _rag_ready:
        return 0

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[PolicyRAG] PyMuPDF not installed. Run: pip install pymupdf")
        return 0

    try:
        doc   = fitz.open(pdf_path)
        text  = " ".join(page.get_text() for page in doc)
        doc.close()

        # Chunk into overlapping windows
        words  = text.split()
        chunks = []
        step   = chunk_size // 2   # 50% overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)

        if not chunks:
            return 0

        embeddings = _embed_model.encode(chunks, normalize_embeddings=True).tolist()
        ids        = [
            hashlib.md5(f"{payer_name}:{c[:50]}".encode()).hexdigest()[:16]
            for c in chunks
        ]
        metadatas  = [{"payer": payer_name, "flag": "POLICY_CHUNK", "source": pdf_path}
                      for _ in chunks]

        _policy_collection.add(
            ids=ids, embeddings=embeddings,
            documents=chunks, metadatas=metadatas,
        )
        print(f"[PolicyRAG] Loaded {len(chunks)} chunks from {pdf_path}")
        return len(chunks)

    except Exception as e:
        print(f"[PolicyRAG] Failed to load PDF: {e}")
        return 0


# ── Core retrieval ────────────────────────────────────────────────────────────

def _retrieve_policies(query: str, n_results: int = 3) -> list[dict]:
    """Retrieve most relevant policy rules for a query string."""
    if not _rag_ready or _policy_collection is None:
        return []
    try:
        query_emb = _embed_model.encode([query], normalize_embeddings=True).tolist()
        results   = _policy_collection.query(
            query_embeddings=query_emb,
            n_results=min(n_results, _policy_collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        policies = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 - dist, 3)   # cosine distance → similarity
            if similarity >= 0.35:             # minimum relevance threshold
                policies.append({
                    "rule_text":   doc,
                    "metadata":    meta,
                    "similarity":  similarity,
                })
        return policies
    except Exception as e:
        print(f"[PolicyRAG] Retrieval error: {e}")
        return []


# ── Code-level validation ─────────────────────────────────────────────────────

def validate_code(
    code: str,
    description: str,
    code_type: str,            # "ICD-10" or "CPT"
    icd_codes_in_claim: list[str] = None,
) -> dict:
    """
    Validate a single code against payer policies.

    Returns:
        {
          "code":        "96365",
          "flag":        "DIAGNOSIS_REQUIRED",
          "message":     "IV infusion requires active infection diagnosis (ICD A/B/J chapter)",
          "source":      "CMS LCD L36143",
          "payer":       "General",
          "severity":    "warning" | "info" | "error",
          "policy_text": "full policy rule text"
        }
    """
    if icd_codes_in_claim is None:
        icd_codes_in_claim = []

    # Build a rich query: code + description + context
    query = f"{code} {description} {code_type} payer policy coverage authorisation"

    policies = _retrieve_policies(query, n_results=4)
    if not policies:
        return _approved_result(code)

    # Find the most relevant policy that actually applies to this code
    for pol in policies:
        meta       = pol["metadata"]
        rule_text  = pol["rule_text"]
        pol_code   = meta.get("code", "")
        flag       = meta.get("flag", "APPROVED")

        # Check if this policy actually applies to this specific code
        if not _policy_applies(code, pol_code, flag, code_type):
            continue

        # For DIAGNOSIS_REQUIRED: check if supporting ICD codes are present
        if flag == "DIAGNOSIS_REQUIRED":
            req_icd    = meta.get("required_icd", [])
            req_prefix = meta.get("required_icd_prefix", "")
            req_chapter= meta.get("required_icd_chapter", [])

            if req_icd:
                codes_norm = [c.replace(".", "").upper() for c in icd_codes_in_claim]
                req_norm   = [c.replace(".", "").upper() for c in req_icd]
                if any(r in codes_norm for r in req_norm):
                    continue   # diagnosis present — policy satisfied
            if req_prefix:
                if any(c.upper().startswith(req_prefix.upper())
                       for c in icd_codes_in_claim):
                    continue
            if req_chapter:
                if any(c[0].upper() in [ch.upper() for ch in req_chapter]
                       for c in icd_codes_in_claim if c):
                    continue

        # Return the policy violation
        return {
            "code":        code,
            "flag":        flag,
            "message":     _build_message(flag, meta, code),
            "source":      meta.get("source", "Payer Policy"),
            "payer":       meta.get("payer", "General"),
            "severity":    _flag_severity(flag),
            "policy_text": rule_text[:300],
            "similarity":  pol["similarity"],
        }

    return _approved_result(code)


def _policy_applies(code: str, pol_code: str, flag: str, code_type: str) -> bool:
    """Check if a retrieved policy rule actually applies to this specific code."""
    code_upper    = code.upper().replace(".", "")
    pol_code_upper= pol_code.upper().replace(".", "")

    # Exact code match
    if code_upper == pol_code_upper:
        return True

    # ICD chapter match (e.g. pol_code="Z" matches all Z-codes)
    if len(pol_code_upper) == 1 and code_upper.startswith(pol_code_upper):
        return True

    # UNSPECIFIED rule — applies to codes ending in 9 or 90
    if pol_code_upper == "UNSPECIFIED":
        return code_upper.endswith("9") or code_upper.endswith("90")

    return False


def _build_message(flag: str, meta: dict, code: str) -> str:
    """Build a human-readable policy violation message."""
    condition = meta.get("condition", "")
    bundled   = meta.get("bundled_with", "")
    req_icd   = meta.get("required_icd", [])

    if flag == "PRIOR_AUTH_REQUIRED":
        msg = f"Prior authorisation required for {code}"
        if condition:
            msg += f" ({condition})"
    elif flag == "BUNDLED":
        msg = f"Code {code} is bundled — cannot bill separately"
        if bundled:
            msg += f" when {bundled} is also billed"
    elif flag == "DIAGNOSIS_REQUIRED":
        msg = f"Code {code} requires supporting diagnosis documentation"
        if req_icd:
            msg += f". Required ICD: {', '.join(req_icd)}"
        elif condition:
            msg += f". Condition: {condition}"
    elif flag == "FREQUENCY_LIMIT":
        msg = f"Frequency limit applies to {code}"
        if condition:
            msg += f": {condition}"
    elif flag == "EXCLUDED":
        msg = f"Code {code} may not be covered by this payer"
    else:
        msg = f"Code {code} — no policy issues found"

    return msg


def _flag_severity(flag: str) -> str:
    severity_map = {
        "PRIOR_AUTH_REQUIRED": "error",
        "BUNDLED":             "error",
        "EXCLUDED":            "error",
        "DIAGNOSIS_REQUIRED":  "warning",
        "FREQUENCY_LIMIT":     "warning",
        "APPROVED":            "info",
    }
    return severity_map.get(flag, "info")


def _approved_result(code: str) -> dict:
    return {
        "code":        code,
        "flag":        "APPROVED",
        "message":     f"No policy issues found for {code}",
        "source":      "Policy check",
        "payer":       "General",
        "severity":    "info",
        "policy_text": "",
        "similarity":  0.0,
    }


# ── Batch validation ──────────────────────────────────────────────────────────

def validate_all_codes(ai_codes: dict) -> dict:
    """
    Validate all AI-generated codes against payer policies.

    Args:
        ai_codes: Output from generate_icd_codes()

    Returns:
        {
          "policy_flags":   list of validation results per code,
          "errors":         codes with error-level flags (prior auth, bundled)
          "warnings":       codes with warning-level flags
          "summary":        { total, errors, warnings, approved }
          "rag_available":  bool
        }
    """
    if not _rag_ready:
        _init_rag()

    if not _rag_ready:
        return {
            "policy_flags":  [],
            "errors":        [],
            "warnings":      [],
            "summary":       {"total": 0, "errors": 0, "warnings": 0, "approved": 0},
            "rag_available": False,
            "rag_error":     _rag_error,
        }

    icd_codes_in_claim = [
        c["icd_code"] for c in ai_codes.get("icd_codes", [])
    ]

    flags    = []
    errors   = []
    warnings = []

    # Validate ICD codes
    for code_rec in ai_codes.get("icd_codes", []):
        result = validate_code(
            code=code_rec["icd_code"],
            description=code_rec.get("description", ""),
            code_type="ICD-10",
            icd_codes_in_claim=icd_codes_in_claim,
        )
        flags.append(result)
        if result["severity"] == "error":
            errors.append(result)
        elif result["severity"] == "warning":
            warnings.append(result)

    # Validate CPT codes
    for code_rec in ai_codes.get("cpt_codes", []):
        result = validate_code(
            code=code_rec["cpt_code"],
            description=code_rec.get("description", ""),
            code_type="CPT",
            icd_codes_in_claim=icd_codes_in_claim,
        )
        flags.append(result)
        if result["severity"] == "error":
            errors.append(result)
        elif result["severity"] == "warning":
            warnings.append(result)

    approved = len(flags) - len(errors) - len(warnings)

    return {
        "policy_flags":  flags,
        "errors":        errors,
        "warnings":      warnings,
        "summary": {
            "total":    len(flags),
            "errors":   len(errors),
            "warnings": len(warnings),
            "approved": approved,
        },
        "rag_available": True,
    }


def is_available() -> bool:
    if not _rag_ready and not _rag_error:
        _init_rag()
    return _rag_ready


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising Policy RAG...")
    _init_rag()
    print(f"Ready: {_rag_ready}")
    if _rag_ready:
        test_codes = {
            "icd_codes": [
                {"icd_code": "J18.9",  "description": "Pneumonia, unspecified"},
                {"icd_code": "N18.3",  "description": "Chronic kidney disease stage 3"},
                {"icd_code": "Z87.01", "description": "History of pneumonia"},
            ],
            "cpt_codes": [
                {"cpt_code": "96365", "description": "IV infusion"},
                {"cpt_code": "85025", "description": "Blood test CBC"},
                {"cpt_code": "93000", "description": "ECG routine"},
            ],
        }
        result = validate_all_codes(test_codes)
        print(f"\nSummary: {result['summary']}")
        print("\nFlags:")
        for f in result["policy_flags"]:
            icon = "✗" if f["severity"]=="error" else "⚠" if f["severity"]=="warning" else "✓"
            print(f"  {icon} [{f['code']:>8}] {f['flag']:<22} {f['message'][:60]}")