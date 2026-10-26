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

### Article input options

The compact newsfeed sits beside the editor on wide screens and becomes a single
horizontally scrollable row on narrower screens. Search is always available; section
and source filters open from the filter button. The editor and feed fill the initial
viewport without global page scrolling, with local scrolling for article text and
stories. Completed analysis automatically scrolls to and focuses the results below.
Select **Open article** to load a story in the editor; **Source** opens its reference
site separately. News API integration can supply the same provider-neutral story
metadata, including actual publication dates and images when available. See
[`docs/newsfeed-api-plan.md`](docs/newsfeed-api-plan.md) for the integration path.

The tabs above the editor provide four ways to prepare an article:

| Tab | Input | Workflow |
| --- | --- | --- |
| **Article** | A headline and article body | Edit directly and select **Analyze Bias**. |
| **Raw** | An entire news page copied with Ctrl+A / Ctrl+C | Paste, select **Extract Article**, then review in Article. |
| **URL** | A public HTTPS news article URL | Fetch and extract the article, then review in Article. |
| **Upload** | A `.txt` document, up to 1 MB | Upload and extract, then review in Article. |

Extraction uses the configured `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.
The agent selects original source blocks; the backend assembles their text without
paraphrasing or removing loaded language. Extraction does not automatically run bias
analysis. Tab drafts are retained, and completed imports replace the editable Article
fields and clear previous analysis/rewrite results.

TXT uploads support UTF-8 and BOM-marked UTF-16. Additional document/image readers can
be added under `src/newsbias/ingestion/readers/`. Raw input is limited to 120,000
characters; URLs have download-size, redirect, and timeout limits. If a website cannot
be fetched or needs JavaScript to expose its article, use copied text in Raw.

Source fetching and HTML metadata/paragraph parsing are adapted from the sister
DistillNews project. URL import tries **Trafilatura**, then **Newspaper4k** (the maintained
Newspaper3k successor), and finally **BeautifulSoup**. Empty/failed parsers and candidates
that the extraction agent cannot identify as an article advance to the next fallback.
Parsers work on the same fetched HTML rather than independently downloading the URL.
The fetcher supports gzip/deflate and detects HTML text encodings; size limits apply
to the decompressed content. Transient server errors receive one bounded retry.
The chosen parser appears in `extraction.method`.

Extraction selections are cached for one hour by content,
headline hint, prompt, and model; URL downloads expire after one minute so changed
pages can be refreshed. Analysis and rewrite caches still use the final article inputs.

### Language-aware web analysis

The web app detects the article body's language locally with Lingua. The current
RoBERTa/BABE and political-stance checkpoints are English-only, so they run only when
English is detected confidently. Non-English, uncertain, and substantial mixed-language
inputs use the original-language LLM reading and editorial judgment instead.

The response includes `language` and `analysis_method`. Skipped specialists have
`status: "skipped"`, a reason, and null scores/counts; the interface displays **Not
evaluated** instead of confidence percentages or neutral predictions. The LLM-only
path requires a valid LLM response and returns an explicit error when unavailable.
Rewrites preserve the original language. Explicit mock previews remain available for
offline testing. Cache versioning prevents older unsupported specialist results from
being reused after this routing change.

Future multilingual checkpoints can declare their own supported languages; this is
preferable to assuming that every BERT-family checkpoint supports the same languages.

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

---

## REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/bias/samples` | `GET` | Returns curated benchmark samples across academic datasets. |
| `/bias/analyze` | `POST` | Analyzes text for sentence spin, political stance, and arbiter synthesis. |
| `/bias/rewrite` | `POST` | Generates a neutralized AP-wire style rewrite with a before/after diff log. |
| `/bias/import/raw` | `POST` | Extracts an article from copied website text (`{"text": "..."}`). |
| `/bias/import/url` | `POST` | Fetches and extracts a public news page (`{"url": "https://..."}`). |
| `/bias/import/file` | `POST` | Extracts an article from a multipart upload in the `file` field. |
| `/` | `GET` | Serves the web interface. |

## Frontend development

```bash
npm ci --prefix frontend
npm test --prefix frontend
npm run build --prefix frontend
```
