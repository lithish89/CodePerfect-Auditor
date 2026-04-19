#!/usr/bin/env python
import os
import sys
import subprocess

os.chdir(r'c:\Users\lithi\Desktop\Code_perfect')
sys.path.insert(0, '.')

result = subprocess.run([
    sys.executable, '-m', 'uvicorn',
    'main:app', '--reload', '--host', '127.0.0.1', '--port', '8000'
], cwd=r'c:\Users\lithi\Desktop\Code_perfect')
sys.exit(result.returncode)
