@echo off
REM ============================================================
REM  Install scispaCy medical NER model for CodePerfect Auditor
REM  Run from inside Code_perfect folder with venv active
REM ============================================================

echo.
echo === Installing scispaCy Medical NER ===
echo.

echo [1/3] Installing scispaCy base package...
pip install scispacy

echo.
echo [2/3] Installing BC5CDR medical NER model...
echo (This is a 100MB download — takes 1-2 minutes)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

echo.
echo [3/3] Verifying installation...
python -c "import spacy; nlp = spacy.load('en_ner_bc5cdr_md'); doc = nlp('Patient has pneumonia and hypertension'); print('Entities found:'); [print(' ', e.text, '-', e.label_) for e in doc.ents]; print('scispaCy OK')"

echo.
echo ============================================================
echo  Done. Restart the server:
echo    uvicorn main:app --reload
echo ============================================================
pause