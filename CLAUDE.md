# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a personal learning repository for studying LLM (Large Language Model) principles. The learning plan covers:

- Stanford CS25: Introduction to Transformers (Andrej Karpathy)
- Building GPT from scratch in code (Andrej Karpathy)
- Deep Dive into LLMs like ChatGPT (Andrej Karpathy)
- Huggingface LLM Course: LLM architecture, fine-tuning, inference, deployment

## Repository State

This repository is in early setup — no code exists yet. As learning progresses, expect the following types of files to appear:

- Jupyter notebooks (`.ipynb`) for interactive experimentation
- Python scripts implementing concepts like attention mechanisms, transformers, GPT training loops
- Notes or markdown files summarizing key concepts

## Expected Development Environment

For Python/ML work in this repo, typical setup includes:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies when a requirements.txt or pyproject.toml appears
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook
# or
jupyter lab
```

Common libraries expected in this learning context: `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`.

## Jupyter Notebook (.ipynb) Rules

`.ipynb` files are JSON. **Always verify JSON validity before committing.** Run this after creating or editing any notebook:

```python
python3 -c "import json; json.load(open('your-notebook.ipynb'))"
```

Common mistake: unescaped double quotes inside markdown cell strings. Example fix:
- Wrong: `"（"我想找什么"）\n"`
- Right:  `"（\"我想找什么\"）\n"`
