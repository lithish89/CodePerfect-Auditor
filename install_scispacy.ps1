# ============================================================
#  Install scispaCy medical NER model for CodePerfect Auditor
#  Run from inside Code_perfect folder with venv active:
#    .\install_scispacy.ps1
# ============================================================

Write-Host ""
Write-Host "=== Installing scispaCy Medical NER ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Installing scispaCy base package..." -ForegroundColor Yellow
pip install scispacy

Write-Host ""
Write-Host "[2/3] Installing BC5CDR medical NER model..." -ForegroundColor Yellow
Write-Host "  (100MB download — takes 1-2 minutes)" -ForegroundColor Gray
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

Write-Host ""
Write-Host "[3/3] Verifying installation..." -ForegroundColor Yellow
$result = python -c @"
import spacy
nlp = spacy.load('en_ner_bc5cdr_md')
doc = nlp('Patient has pneumonia and poorly controlled hypertension. History of type 2 diabetes.')
print('Entities found:')
for e in doc.ents:
    print(f'  {e.text} -> {e.label_}')
print('scispaCy OK')
"@ 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $result -ForegroundColor Green
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " scispaCy installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host " Restart the server:" -ForegroundColor White
    Write-Host "   uvicorn main:app --reload" -ForegroundColor White
    Write-Host ""
    Write-Host " The NER method in API responses will now show:" -ForegroundColor Gray
    Write-Host '   "ner_method": "scispacy"' -ForegroundColor Gray
    Write-Host "============================================================" -ForegroundColor Cyan
} else {
    Write-Host "Installation may have failed. Error:" -ForegroundColor Red
    Write-Host $result -ForegroundColor Red
    Write-Host ""
    Write-Host "Try the fallback model instead:" -ForegroundColor Yellow
    Write-Host "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz" -ForegroundColor White
}

Read-Host "Press Enter to exit"