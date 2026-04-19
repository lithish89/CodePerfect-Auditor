"""
CodePerfect Auditor  v8  —  Fixed Release
------------------------------------------
✓ File upload fixed (PDF, Word, image, text)
✓ ICD/CPT code generation from clinical notes
✓ PDF report download enabled
✓ Audit logs hidden from insurers (hospital only)

Features:
  ✓ Role-based auth  (hospital | insurer)
  ✓ File upload      (PDF, Word, image OCR, text)
  ✓ Patient details  + insurer details saved per audit
  ✓ Insurance claim  raised and emailed to both
  ✓ Claim management (insurer portal — status updates)
  ✓ PDF report       download per audit
  ✓ Audit logs       HOSPITAL ONLY (insurers cannot see)


Endpoints changed:
  /logs → Hospital only (returns 403 for insurers)
  /claims → Available to both hospital & insurer
"""

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Header, Form
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.datastructures import Headers
from pydantic import BaseModel
from typing import Optional
import os, threading, json, io

# ─ Heavy agent modules lazily imported in _run_pipeline to avoid startup failures
# from app.agents.clinical_reader import extract_medical_entities
# from app.agents.coding_logic     import generate_icd_codes
# from app.agents.auditor_agent    import audit_codes

from app.database.audit_db  import (
    init_db, save_audit, get_all_audits, get_audit_by_id,
    get_audit_stats, update_audit_status, delete_audit,
)
from app.database.claims_db import (
    init_claims_db, save_patient as db_save_patient, get_patient_by_audit,
    raise_claim, get_claim, get_claims_for_insurer,
    get_claims_for_hospital, update_claim_status,
    mark_report_sent, get_all_claims,
)
from app.auth.auth import init_auth_db, login, create_user, get_current_user, get_user_by_email

# ── Startup ────────────────────────────────────────────────────────────────
init_db()
init_auth_db()
init_claims_db()

def _bg(fn):
    try: fn()
    except Exception: pass

threading.Thread(
    target=lambda: _bg(lambda: __import__('app.services.policy_rag', fromlist=['_init_rag'])._init_rag()),
    daemon=True
).start()

app = FastAPI(
    title="CodePerfect Auditor",
    description="AI-driven medical coding with claims management.",
    version="8.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Optional: Custom exception handler for validation errors (for debugging)
try:
    from fastapi.exceptions import RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        print(f"\n[ValidationError] {request.method} {request.url.path}")
        print(f"  Errors: {exc.errors()}")
        details = []
        for e in exc.errors():
            print(f"    - {e['loc']}: {e['msg']}")
            details.append({"loc": str(e["loc"]), "msg": e["msg"]})
        return {"detail": "Request validation failed", "errors": details}
except Exception as imp_err:
    print(f"[Warning] Could not setup RequestValidationError handler: {imp_err}")


# ── Request models ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username:     str
    password:     str
    role:         str        # hospital | insurer (NOT admin)
    full_name:    str = ""
    organization: str = ""
    email:        str = ""

class AnalyzeRequest(BaseModel):
    clinical_note:  str
    human_codes:    list = []
    explain:        bool = True
    check_policy:   bool = True

class ExplainRequest(BaseModel):
    clinical_note: str

class ICDSearchRequest(BaseModel):
    query: str
    top_k: int = 3

class CPTSearchRequest(BaseModel):
    procedure: str

class DirectAuditRequest(BaseModel):
    ai_codes:    dict
    human_codes: list

class StatusUpdate(BaseModel):
    status: str

class PatientRequest(BaseModel):
    audit_id:       str
    name:           str
    phone:          str = ""
    email:          str = ""
    dob:            str = ""
    gender:         str = ""
    address:        str = ""
    insurance_code: str = ""
    insurance_org:  str = ""
    insurer_email:  str = ""
    hospital_name:  str = ""

class ClaimRequest(BaseModel):
    audit_id:    str
    send_emails: bool = True

class ClaimStatusUpdate(BaseModel):
    status:  str
    remarks: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────

def _auth(authorization: str = "") -> Optional[dict]:
    return get_current_user(authorization)

def _require_auth(authorization: str) -> dict:
    user = _auth(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user

def _require_role(authorization: str, *roles: str) -> dict:
    user = _require_auth(authorization)
    if user.get("role") not in roles:
        raise HTTPException(status_code=403, detail=f"Access denied. Required role: {', '.join(roles)}")
    return user

def _run_explainability(entities, ai_codes):
    try:
        from app.services.explainability import explain_all_codes
        preferred = ai_codes.get("_preferred_terms", set())
        if not isinstance(preferred, set):
            preferred = set(preferred)
        return explain_all_codes(
            icd_codes=ai_codes.get("icd_codes", []),
            sentence_contexts=entities.get("sentence_contexts", {}),
            preferred_terms=preferred,
        )
    except Exception as e:
        print(f"[Explainability] {e}")
        return []

def _clean(ai_codes):
    return {k: v for k, v in ai_codes.items() if not k.startswith("_")}

def _run_pipeline(note: str, human_codes: list, explain=True, check_policy=True):
    # Lazy import of heavy ML modules to allow server startup without all dependencies
    from app.agents.clinical_reader import extract_medical_entities
    from app.agents.coding_logic     import generate_icd_codes
    from app.agents.auditor_agent    import audit_codes
    
    entities     = extract_medical_entities(note)
    ai_codes     = generate_icd_codes(entities)
    audit_result = audit_codes(ai_codes, human_codes) if human_codes else None
    explanations = _run_explainability(entities, ai_codes) if explain else []
    policy_result = None
    if check_policy:
        try:
            from app.services.policy_rag import validate_all_codes
            policy_result = validate_all_codes(ai_codes)
        except Exception as e:
            policy_result = {"rag_available": False, "rag_error": str(e)}
    audit_id = save_audit(
        clinical_note=note, entities=entities,
        ai_codes=ai_codes, audit_result=audit_result, human_codes=human_codes,
    )
    return {
        "audit_id":      audit_id,
        "entities":      entities,
        "ai_codes":      _clean(ai_codes),
        "manual_review": ai_codes.get("manual_review", []),
        "audit_result":  audit_result,
        "explanations":  explanations,
        "policy":        policy_result,
    }


# ── UI ─────────────────────────────────────────────────────────────────────

@app.get("/ui", include_in_schema=False)
def serve_ui():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ui.html not found.")
    return FileResponse(path, media_type="text/html")

@app.get("/", tags=["Health"])
def home():
    return {"service": "CodePerfect Auditor", "status": "running", "version": "8.0.0"}


# ── Auth ───────────────────────────────────────────────────────────────────

@app.post("/auth/login", tags=["Auth"])
def auth_login(req: LoginRequest):
    result = login(req.username, req.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return result

@app.post("/auth/register", tags=["Auth"])
def auth_register(req: RegisterRequest):
    """Register a new user (hospital or insurer role only)"""
    try:
        # Validate role
        valid_roles = {"hospital", "insurer"}
        if req.role not in valid_roles:
            raise HTTPException(status_code=400, detail=f"Role must be one of: {', '.join(valid_roles)}")
        
        # Validate required fields
        if not req.username or not req.username.strip():
            raise HTTPException(status_code=400, detail="Username is required")
        if not req.password or not req.password.strip():
            raise HTTPException(status_code=400, detail="Password is required")
        
        # Create user
        ok, msg = create_user(
            req.username.strip(), req.password, req.role,
            req.full_name or "", req.organization or "", req.email or ""
        )
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        
        return {"username": req.username, "role": req.role, "message": "Account created successfully."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Register Error] {str(e)}")
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")

@app.get("/auth/me", tags=["Auth"])
def auth_me(authorization: Optional[str] = Header(default=None)):
    user = _require_auth(authorization or "")
    return user


# ── File Upload (FIXED) ────────────────────────────────────────────────────

@app.post("/upload", tags=["Upload"])
async def upload_file(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    """
    ✓ FIXED: Upload PDF, Word, image, or text file
    Extract text via file_extractor and run full pipeline
    """
    print(f"\n[Upload START]")
    print(f"  Authorization header present: {bool(authorization)}")
    print(f"  File object: {file}")
    print(f"  File filename: {file.filename}")
    print(f"  Content-Type: {file.content_type}")
    
    try:
        # First, validate auth
        user = _require_auth(authorization or "")
        print(f"  Auth OK - User: {user.get('username')}")
    except HTTPException as auth_err:
        print(f"  Auth FAILED: {auth_err.detail}")
        raise
    
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided or no filename.")
        
        print(f"  Reading file bytes...")
        # Read file bytes
        file_bytes = await file.read()
        print(f"  File size: {len(file_bytes)} bytes")
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File is empty.")
        
        # Extract text using file_extractor
        from app.services.file_extractor import extract_text, is_supported
        
        if not is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: PDF, DOCX, TXT, PNG, JPG, JPEG"
            )
        
        print(f"  Extracting text...")
        result = extract_text(file_bytes, file.filename)
        print(f"  Extraction result: {result.get('success')}")
        
        # If server extraction fails, allow client-side fallback (especially for images)
        if not result.get("success"):
            error_msg = result.get('error', 'Unknown error')
            # For images, allow client-side extraction as fallback
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return {
                    "success": True,
                    "file": file.filename,
                    "extraction_method": "client-side-pending",
                    "char_count": 0,
                    "clinical_note": "",
                    "message": "Image uploaded. Please use the client-side extraction (browser will process via Claude) or paste text manually."
                }
            raise HTTPException(status_code=400, detail=f"Extraction failed: {error_msg}")
        
        clinical_note = result.get("text", "").strip()
        if not clinical_note or len(clinical_note) < 10:
            raise HTTPException(status_code=400, detail="Extracted text is too short or empty.")
        
        print(f"  Running pipeline...")
        # Run full pipeline
        pipeline_result = _run_pipeline(clinical_note, [], explain=True, check_policy=True)
        
        print(f"[Upload SUCCESS]")
        return {
            "success": True,
            "file": file.filename,
            "extraction_method": result.get("method"),
            "char_count": result.get("char_count"),
            "clinical_note": clinical_note,
            **pipeline_result,
        }
    
    except HTTPException:
        print(f"[Upload FAILED - HTTP Exception]")
        raise
    except Exception as e:
        print(f"[Upload ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
        
        # Extract text using file_extractor
        from app.services.file_extractor import extract_text, is_supported
        
        if not is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: PDF, DOCX, TXT, PNG, JPG, JPEG"
            )
        
        result = extract_text(file_bytes, file.filename)
        
        # If server extraction fails, allow client-side fallback (especially for images)
        if not result.get("success"):
            error_msg = result.get('error', 'Unknown error')
            # For images, allow client-side extraction as fallback
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return {
                    "success": True,
                    "file": file.filename,
                    "extraction_method": "client-side-pending",
                    "char_count": 0,
                    "clinical_note": "",
                    "message": "Image uploaded. Please use the client-side extraction (browser will process via Claude) or paste text manually."
                }
            raise HTTPException(status_code=400, detail=f"Extraction failed: {error_msg}")
        
        clinical_note = result.get("text", "").strip()
        if not clinical_note or len(clinical_note) < 10:
            raise HTTPException(status_code=400, detail="Extracted text is too short or empty.")
        
        # Run full pipeline
        pipeline_result = _run_pipeline(clinical_note, [], explain=True, check_policy=True)
        
        return {
            "success": True,
            "file": file.filename,
            "extraction_method": result.get("method"),
            "char_count": result.get("char_count"),
            "clinical_note": clinical_note,
            **pipeline_result,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Upload] Error: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


# ── Analysis ───────────────────────────────────────────────────────────────

@app.post("/analyze", tags=["Analysis"])
def analyze(
    req: AnalyzeRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Run full pipeline on clinical note."""
    _require_auth(authorization or "")
    
    if not req.clinical_note or len(req.clinical_note.strip()) < 10:
        raise HTTPException(status_code=400, detail="Clinical note is required and must be at least 10 characters.")
    
    return _run_pipeline(
        req.clinical_note,
        req.human_codes,
        explain=req.explain,
        check_policy=req.check_policy,
    )



@app.get("/download-report")
def download_legacy_report():
    file_path = "report.pdf"
    
    if not os.path.exists(file_path):
        return {"error": "Report not found"}
    
    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        filename="medical_report.pdf"
    )


    



@app.post("/explain", tags=["Analysis"])
def explain(req: ExplainRequest, authorization: Optional[str] = Header(default=None)):
    _require_auth(authorization or "")
    try:
        from app.services.explainability import explain_code
        return {"explanation": "Explainability endpoint working"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ICD & CPT Search ───────────────────────────────────────────────────────

@app.post("/codes/icd", tags=["Codes"])
def search_icd(req: ICDSearchRequest, authorization: Optional[str] = Header(default=None)):
    _require_auth(authorization or "")
    try:
        from app.services.icd_semantic_engine import search_icd
        results = search_icd(req.query, top_k=req.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/codes/cpt", tags=["Codes"])
def search_cpt(req: CPTSearchRequest, authorization: Optional[str] = Header(default=None)):
    _require_auth(authorization or "")
    try:
        from app.agents.coding_logic import _get_cpt_code
        cpt = _get_cpt_code(req.procedure)
        return {"result": cpt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Patient & Claims ───────────────────────────────────────────────────────

@app.post("/patients", tags=["Patients"])
def save_patient(req: PatientRequest, authorization: Optional[str] = Header(default=None)):
    _require_role(authorization or "", "hospital")

    audit = get_audit_by_id(req.audit_id)
    if not audit:
        raise HTTPException(status_code=404, detail="Audit not found. Save an audit first via /analyze.")

    patient_id = db_save_patient(
        audit_id=req.audit_id,
        name=req.name,
        phone=req.phone,
        email=req.email,
        dob=req.dob,
        gender=req.gender,
        address=req.address,
        insurance_code=req.insurance_code,
        insurance_org=req.insurance_org,
        insurer_email=req.insurer_email,
        hospital_name=req.hospital_name,
    )
    if not patient_id:
        raise HTTPException(status_code=400, detail="Failed to save patient details.")
    return {"patient_id": patient_id, "audit_id": req.audit_id, "message": "Patient details saved."}

@app.get("/patients/{audit_id}", tags=["Patients"])
def get_patient(audit_id: str, authorization: Optional[str] = Header(default=None)):
    _require_auth(authorization or "")
    patient = get_patient_by_audit(audit_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient


# ── Claims Management ──────────────────────────────────────────────────────

@app.post("/claims", tags=["Claims"])
def raise_claim_endpoint(
    req: ClaimRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Raise insurance claim — hospital only."""
    user = _require_role(authorization or "", "hospital")
    
    # Get audit
    audit = get_audit_by_id(req.audit_id)
    if not audit:
        raise HTTPException(status_code=404, detail="Audit not found.")
    
    patient = get_patient_by_audit(req.audit_id)
    if not patient:
        raise HTTPException(status_code=400, detail="Save patient details first via POST /patients.")
    
    # Parse codes from audit
    codes_by_cat = {}
    for c in audit.get("codes", []):
        cat = c.get("category","")
        if cat not in codes_by_cat:
            codes_by_cat[cat] = []
        codes_by_cat[cat].append(c)
    
    ai_codes_obj = {
        "icd_codes": [{"icd_code": c["code"], "description": c.get("description","")}
                      for c in codes_by_cat.get("ai_confirmed",[])],
        "cpt_codes": [{"cpt_code": c["code"], "description": c.get("description","")}
                      for c in codes_by_cat.get("ai_confirmed",[]) if c.get("code_type")=="CPT"],
    }
    
    audit_result = {
        "risk_level":    audit.get("risk_level",""),
        "revenue_risk":  audit.get("revenue_risk",""),
        "total_at_risk": audit.get("total_at_risk",""),
        "summary": {
            "matched":       audit.get("matched_count",0),
            "missing_count": audit.get("missing_count",0),
            "extra_count":   audit.get("extra_count",0),
        },
    }
    
    claim_id = raise_claim(
        audit_id=req.audit_id,
        patient=patient,
        audit_result=audit_result,
        ai_codes=ai_codes_obj,
        submitted_by=user.get("sub",""),
    )
    
    email_results = {}
    if req.send_emails:
        pdf_bytes = None
        try:
            try:
                from app.services.report_generator import generate_audit_report
            except ImportError:
                print("[ClaimSubmit] report_generator module not found, skipping PDF generation")
                generate_audit_report = None
            
            if generate_audit_report:
                audit["patient"] = patient
                pdf_bytes = generate_audit_report(audit)
        except Exception as e:
            print(f"[ClaimSubmit] PDF generation failed: {e}")
        
        from app.services.email_service import send_patient_report, send_insurer_claim
        
        if patient.get("email"):
            ok, msg = send_patient_report(
                patient_email=patient["email"],
                patient_name=patient["name"],
                claim_id=claim_id,
                audit_result=audit_result,
                pdf_bytes=pdf_bytes,
            )
            email_results["patient"] = {"sent": ok, "message": msg}
        
        if patient.get("insurer_email"):
            ok, msg = send_insurer_claim(
                insurer_email=patient["insurer_email"],
                insurer_org=patient.get("insurance_org","Insurer"),
                patient_name=patient["name"],
                claim_id=claim_id,
                audit_result=audit_result,
                ai_codes=ai_codes_obj,
                hospital_name=patient.get("hospital_name",""),
                insurance_code=patient.get("insurance_code",""),
                pdf_bytes=pdf_bytes,
            )
            email_results["insurer"] = {"sent": ok, "message": msg}
        
        if email_results:
            mark_report_sent(claim_id)
    
    return {
        "claim_id":     claim_id,
        "status":       "pending",
        "email_results": email_results,
        "message":      "Claim raised successfully.",
    }


@app.get("/claims", tags=["Claims"])
def list_claims(authorization: Optional[str] = Header(default=None)):
    """
    List claims — role-filtered:
      insurer → sees only claims addressed to their email
      hospital → sees only their submitted claims
    """
    user = _require_auth(authorization or "")
    role = user.get("role")
    email= user.get("email","")
    sub  = user.get("sub","")
    
    if role == "insurer":
        claims = get_claims_for_insurer(email)
    elif role == "hospital":
        claims = get_claims_for_hospital(sub)
    else:
        claims = []
    
    return {"total": len(claims), "claims": claims}


@app.get("/claims/{claim_id}", tags=["Claims"])
def get_claim_detail(claim_id: str, authorization: Optional[str] = Header(default=None)):
    _require_auth(authorization or "")
    claim = get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found.")
    return claim


@app.patch("/claims/{claim_id}/status", tags=["Claims"])
def update_status(
    claim_id: str,
    body: ClaimStatusUpdate,
    authorization: Optional[str] = Header(default=None),
):
    """Update claim status — insurer only."""
    user = _require_role(authorization or "", "insurer")
    valid = {"pending","requires_verification","issued","accepted","rejected"}
    if body.status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of: {valid}")
    
    ok = update_claim_status(claim_id, body.status, body.remarks, user.get("sub",""))
    if not ok:
        raise HTTPException(status_code=404, detail="Claim not found.")
    
    claim = get_claim(claim_id)
    email_result = {}
    if claim and claim.get("patient_email"):
        from app.services.email_service import send_status_update
        ok2, msg = send_status_update(
            patient_email=claim["patient_email"],
            patient_name=claim.get("patient_name","Patient"),
            claim_id=claim_id,
            new_status=body.status,
            remarks=body.remarks,
            insurer_org=user.get("org",""),
        )
        email_result = {"sent": ok2, "message": msg}
    
    return {
        "claim_id": claim_id,
        "status":   body.status,
        "remarks":  body.remarks,
        "email":    email_result,
    }


# ── Audit Logs (HOSPITAL ONLY) ─────────────────────────────────────────────

@app.get("/logs", tags=["Audit Logs"])
def list_audit_logs(
    limit:  int           = Query(default=20, ge=1, le=100),
    offset: int           = Query(default=0,  ge=0),
    risk:   Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
):
    """
    ✓ FIXED: Hospital only — insurers get 403
    """
    user = _require_role(authorization or "", "hospital")
    logs = get_all_audits(limit=limit, offset=offset, risk_filter=risk)
    return {"total": len(logs), "offset": offset, "limit": limit, "logs": logs}

@app.get("/logs/stats", tags=["Audit Logs"])
def audit_statistics(authorization: Optional[str] = Header(default=None)):
    """Hospital only."""
    user = _require_role(authorization or "", "hospital")
    return get_audit_stats()

@app.get("/logs/{audit_id}/report.pdf", tags=["Audit Logs"])
def download_report(audit_id: str, authorization: Optional[str] = Header(default=None)):
    """
    Generate and download PDF compliance report
    Hospital and Insurer can download
    """
    _require_auth(authorization or "")
    record = get_audit_by_id(audit_id)
    if not record:
        raise HTTPException(status_code=404, detail="Audit not found.")
    
    patient = get_patient_by_audit(audit_id)
    if not patient:
        raise HTTPException(status_code=400, detail="Patient details not found. Save patient details first via POST /patients.")
    
    record["patient"] = patient
    
    try:
        from app.services.report_generator import generate_audit_report
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"Report generation service not available: {str(e)}")
    
    try:
        pdf = generate_audit_report(record)
        return Response(
            content=pdf, media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="audit_{audit_id[:8]}.pdf"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report error: {str(e)}")

@app.get("/logs/{audit_id}", tags=["Audit Logs"])
def get_audit_log(audit_id: str, authorization: Optional[str] = Header(default=None)):
    """Hospital only."""
    user = _require_role(authorization or "", "hospital")
    r = get_audit_by_id(audit_id)
    if not r:
        raise HTTPException(status_code=404, detail="Audit not found.")
    patient = get_patient_by_audit(audit_id)
    if patient:
        r["patient"] = patient
    return r

@app.patch("/logs/{audit_id}/status", tags=["Audit Logs"])
def update_log_status(audit_id: str, body: StatusUpdate, authorization: Optional[str] = Header(default=None)):
    """Hospital only."""
    user = _require_role(authorization or "", "hospital")
    valid = {"pending","reviewed","approved","disputed"}
    if body.status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of: {valid}")
    if not update_audit_status(audit_id, body.status):
        raise HTTPException(status_code=404, detail="Audit not found.")
    return {"audit_id": audit_id, "status": body.status, "updated": True}

@app.delete("/logs/{audit_id}", tags=["Audit Logs"])
def delete_audit_log(audit_id: str, authorization: Optional[str] = Header(default=None)):
    """Hospital only."""
    user = _require_role(authorization or "", "hospital")
    if not delete_audit(audit_id):
        raise HTTPException(status_code=404, detail="Audit not found.")
    return {"audit_id": audit_id, "deleted": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)