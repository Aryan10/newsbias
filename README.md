# News Bias Detection using Machine Learning

A machine learning system for media bias detection, political stance classification, and neutral editorial rewriting.

The architecture combines specialized transformer heads (RoBERTa on BABE for sentence-level spin, DeBERTa on AllSides for political stance) alongside an isolated zero-shot LLM reading. An Arbiter Agent synthesizes these signals into an objective consensus verdict and generates an AP-style neutral rewrite with a detailed modification log.

---

## Quickstart: Web Application

### 1. Installation

Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra web --extra transformers --extra llm
```

Optional: Set your OpenAI/LLM API key in `.env` for zero-shot LLM readings:

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Run the Web Server

Launch the web app on `http://127.0.0.1:8000`:

```bash
python cli.py serve
```

For using our custom models with the web application, see releases.

---

## CLI Reference

The project provides a simplified `cli.py` for model training, evaluation, and serving:

### Web Server
```bash
python cli.py serve [--host 127.0.0.1] [--port 8000]
```

### Model Training
```bash
# Train TF-IDF baseline for political stance classification (AllSides)
python cli.py train tfidf --political

# Train TF-IDF baseline for sentence-level bias classification (BABE)
python cli.py train tfidf --bias
```

### Model Evaluation
```bash
# Evaluate trained TF-IDF baseline
python cli.py eval tfidf --political

# Evaluate DeBERTa-v3-large on AllSides political stance
python cli.py eval deberta-large --political --limit 50

# Evaluate RoBERTa on BABE sentence spin
python cli.py eval roberta-babe --bias --limit 50

# Evaluate zero-shot LLM reading
python cli.py eval gpt-5-mini --political --limit 20
```

