#!/usr/bin/env python3
"""NewsBias Command-Line Interface.

Usage:
    python cli.py serve                       # Launch the web application
    python cli.py train tfidf --political     # Train TF-IDF political stance baseline
    python cli.py eval deberta-large --political --limit 50  # Evaluate DeBERTa on AllSides
    python cli.py eval roberta-babe --bias --limit 50        # Evaluate RoBERTa on BABE
"""

import sys
from pathlib import Path

# Add src to sys.path so it works out of the box
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from newsbias.cli import main

if __name__ == "__main__":
    main()
