#!/bin/bash
export GEMINI_API_KEY="TEST_KEY"
./venv/bin/python3 wiki_crawler.py \
  --start "Python (programming language)" \
  --target "C (programming language)" \
  --strategy best \
  --use-llm-hopping \
  --max-depth 2
