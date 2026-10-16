# Repository redundancy audit and structure plan

Audit date: 2026-10-30. This document proposes a migration; it does not perform it.

## Applied low-risk cleanup (2026-10-30)

The audit below records the original layout. The following cleanup has now been applied:

- Removed the three root `prompts/*.prompt.j2` duplicates and the empty root prompt
  directory. Built-ins remain in `src/newsbias/prompts/`, with unchanged checksums.
- Removed `doc/.gitignore` and its empty directory, `data/processed/.gitkeep`, and the
  one-line unused `data/test.txt` placeholder.
- Cleared project Python bytecode, pytest caches, and Ruff caches.
- Documented the canonical template location and custom overrides in `README.md`.
- Verified 12 existing prompt/analyzer/segmenter tests, built a wheel, and checked all
  three built-in prompts from the installed wheel outside the checkout. Direct-file,
  explicit-search-path, and local-directory overrides also passed.

The existing loader already supports the single packaged template directory, so no
runtime code or packaging changes were needed. The broader resource/implementation
reorganization below remains a future migration plan.

## Summary

- **Three authored files are confirmed exact duplicates:** the three root-level
  prompt templates duplicate the three templates inside the Python package.
- There are **overlapping implementations** for dataset preparation, transformer
  inference, OpenAI access/JSON parsing, and command orchestration. They have
  different contracts and behavior, so they require consolidation rather than deletion.
- Python caches, dependency installations, frontend build output, and processed
  datasets are generated artifacts. Their presence is not duplicate application source.
- `src/newsbias/` is appropriate and should remain: it defines the importable Python
  package used by the entrypoints, tests, and TOML adapter paths.
- Keep one implementation per behavior and one authored copy per resource. Small
  compatibility entrypoints can delegate to that implementation without duplicating it.

## 1. Why `src/newsbias/`, rather than everything directly in `src/`?

The two directory names have different jobs:

```text
src/                 # Container for installable Python source
  newsbias/          # The actual Python package: import newsbias
    models/          # import newsbias.models
    datasets/        # import newsbias.datasets
```

`pyproject.toml` uses Hatchling and declares these entrypoints:

```text
newsbias           -> newsbias.cli:main
newsbias-benchmark -> newsbias.benchmark_cli:main
newsbias-remote    -> newsbias.remote.cli:main
```

The `src` layout separates importable application code from the repository's tests,
scripts, configurations, and frontend. It also helps expose packaging problems that
can be hidden when imports happen directly from the repository root.

Moving `cli.py`, `models/`, and `datasets/` directly into `src/` would require changing
package discovery, imports, entrypoints, and string-based adapter references. Generic
names such as `datasets` would also conflict with the installed Hugging Face library.
Putting an `__init__.py` in `src/` would instead create a package named `src`, not
preserve the existing `newsbias` namespace.

**Recommendation: keep `src/newsbias/`. Reduce duplicated responsibilities inside it.**

The root `cli.py` is a 22-line launch wrapper that adds `src` to `sys.path` and imports
`newsbias.cli.main`. It is not a second CLI implementation. Retain it to preserve the
documented `python cli.py ...` commands.

## 2. Confirmed duplicate authored files

SHA-256 comparison confirmed that each pair below is byte-for-byte identical:

| Redundant authored copy | Canonical copy to retain initially | SHA-256 prefix |
| --- | --- | --- |
| `prompts/bias_detection.prompt.j2` | `src/newsbias/prompts/bias_detection.prompt.j2` | `2a086f4ff318` |
| `prompts/bias_removal.prompt.j2` | `src/newsbias/prompts/bias_removal.prompt.j2` | `a9d7a4bbc265` |
| `prompts/zero_shot_bias.prompt.j2` | `src/newsbias/prompts/zero_shot_bias.prompt.j2` | `ca96aa7d5e3a` |

### Why both prompt directories currently exist

`src/newsbias/prompts.py:102-107` searches in this order:

1. `prompts/` relative to the current working directory.
2. `prompts/` under its inferred project root.
3. `prompts/` inside the Python package.
4. `../prompts/` relative to the current working directory.

This appears intended to support local/custom templates and an installed-package
fallback. The code establishes that mechanism; it does not establish who originally
created the copies or why they chose to duplicate them.

From the repository root, the outer copies normally win. Editing only the inner
copies can therefore have no effect on a local run. The copies can silently drift.

### Proposed resolution

- Store built-in templates once, under `src/newsbias/resources/prompts/`.
- Keep `newsbias.prompts` as the template-engine module and public import location.
- Resolve built-in resources using `importlib.resources`, supporting wheel installs.
- Preserve direct file loading and caller-provided `search_paths`.
- Preserve existing current-directory and parent-directory override lookup behavior
  for user-supplied templates; do not ship a second authored override directory.
- Preserve the inferred project-root override during migration if callers use it.
- Include templates explicitly in wheel/sdist packaging and test loading outside
  the repository before removing the three outer copies.

`prompts.py` and the prompt directory are not duplicate files: one is executable
Python; the other contains the text it loads. Renaming the resource directory makes
their roles clearer.

A checksum comparison of the application Python files, tests, TOML configs, and
frontend source/config files found no additional byte-identical file pairs.

## 3. Overlapping implementations that need consolidation

Paths in this table are relative to `src/newsbias/`, except `scripts/...`.

| Files / responsibilities | Evidence and important differences | Plan |
| --- | --- | --- |
| `datasets/political.py` and `datasets/all_sides.py` | Both load AllSides. The former returns records, defaults to an unpinned revision, accepts local input, and permits loose label mapping; the latter returns a cached `DatasetDict`, pins a revision, retains source columns, and validates a stricter schema. `datasets.__init__` exports the former; configs use the latter. | Share source/record utilities where semantics match; retain both public contracts and their defaults. |
| `datasets/bias.py:load_babe` and `datasets/babe.py` | Both load BABE, but delimiter handling, validation, IDs, hashes, fields, and output types differ. Both are used by code/tests. | Share CSV-reading mechanics with explicit policies; retain functional records and strict cached adapter behavior. |
| `datasets/political.py:create_random_splits`, `datasets/bias.py:create_stratified_splits`, `datasets/splitting.py:stratified_splits`, and remote `load_project_splits` | The first two repeat a Python `random.Random` grouping/shuffling algorithm; the others use two-stage scikit-learn stratification. These do not generally produce the same membership or row order. | One record-splitting helper parameterized by grouping key; one two-stage index-splitting helper. Keep named policies and wrappers. Bundle the shared helper for remote execution. |
| `data.py:text_hash` and `datasets/bias.py:hash_text` | Both hash text, but the first collapses whitespace; the second also lowercases and removes punctuation. Their equality rules differ. | Keep both named hash policies. Do not replace one with the other or invalidate IDs accidentally. |
| `models/political.py` and `models/huggingface.py` | Both perform transformer sliding-window mean pooling. Default checkpoints, generic label handling, rounding, model-loading lifecycle, and result metadata differ. | Extract a shared window-inference helper; preserve profile-specific defaults and result formatting. |
| `models/bias.py` and `models/huggingface.py` | Both run sequence classifiers, but the sentence predictor uses a Transformers pipeline with single-window truncation and binary-label interpretation. | Share loading/label utilities only where equivalent. Keep sentence and article inference policies distinct. |
| `models/llm.py:get_openai_client` and `models/openai.py:_create_client` | Both create OpenAI clients. Dotenv auto-loading, default-model handling, timeout, and SDK retry settings differ. | One client-construction helper with explicit options; preserve each caller's settings and precedence. |
| `models/llm.py:_extract_json`, `models/openai.py:_parse_response`, and `prompts.py:extract_json_payload` | All parse model JSON. Accepted formats, object validation, and fallback-versus-error behavior differ. | One parsing utility with compatibility modes; public wrappers retain existing return/error behavior. |
| Inline LLM prompts in `models/llm.py` and `cli.py`, plus `.prompt.j2` files | Several instructions are embedded in Python, but they describe different tasks/output schemas. They are not interchangeable copies of the arbiter templates. | Put each distinct instruction in one packaged template, preserving the exact rendered messages and request parameters. |
| `cli.py` model registries/config constants | `MODEL_REGISTRY` and `MODEL_ALIASES` repeat many checkpoint IDs. They have different aliases and execution metadata. | One registry with workflow profiles; preserve the union of aliases and each workflow's settings. |
| `cli.py:handle_merge_bias` and `scripts/merge_bias_datasets.py` | Both orchestrate merging. The script also accepts explicit paths, CSV output, stratified/LODO outputs, and an optional weak corpus. | One packaged merge implementation with profiles; retain the simplified command's output and the script's richer interface. |
| `cli.py`, `benchmark_cli.py`, and `remote/cli.py` | They overlap in execution, but expose different interfaces. The benchmark CLI rejects training-enabled configs; remote CLI dispatches arbitrary scripts and has more job options. | Extract shared execution helpers; preserve all three entrypoints, flags, defaults, validation, and output schemas. |
| `remote/scripts/run_transformer_kaggle.py` and local inference/evaluation helpers | The self-contained remote script repeats stratification, inference, and metrics. It also supports training and processes all windows in bounded batches, unlike local capped-window inference. | Stage shared dependency-light modules alongside a thin remote entrypoint. Preserve remote training, all-window processing, rounding, and legacy artifact fields. |

**No Python module in these groups is established as safely deletable dead code.**
Some functionality is dynamically imported from TOML or exposed publicly through
`__init__.py`; absence of a direct import would not be sufficient evidence of disuse.

## 4. Generated artifacts, clutter, and files to retain

| Path / pattern | Classification | Recommendation |
| --- | --- | --- |
| `**/__pycache__/**`, `**/*.pyc` | Regenerable bytecode. Includes stale caches for absent source files: `models/gpt5_mini.py`, `models/pretrained.py`, and both Python 3.12/3.14 bytecode. | Clean when desired; already ignored. Do not infer missing source files should be restored. |
| `.pytest_cache/`, `.ruff_cache/` | Tool caches. | Clean when desired; already ignored. |
| `.venv/` | Installed Python environment. | Keep local/ignored; recreate from `uv.lock` when needed. |
| `frontend/node_modules/` | Installed JavaScript dependencies. | Keep local/ignored; recreate with `npm ci`. Vendor duplicates here are not application-source redundancy. |
| `frontend/dist/` | Compiled frontend, currently served by `web/app.py:43-45,551-557`. | Rebuildable but required for the current UI. Rebuild/package before removing or relocating it. |
| `data/cache/` | Hugging Face download cache. | Keep local/ignored; avoid duplicate download-cache roots in future settings. |
| `data/processed/allsides/` | Fingerprinted prepared dataset cache. | Keep local/ignored; different fingerprints can represent different preprocessing, not redundant files. |
| `data/processed/merged_bias.json` and `merged_bias.csv` | Two export representations of merged records. Not confirmed byte duplicates or semantically compared. | Prefer JSON for internal consumption; generate CSV on demand if required. Preserve requested export formats. |
| `data/processed/merged_bias_{train,validation,test}.json` | Prepared stratified partitions used by `MergedBiasDataset`. | Retain until equivalent partitions can be regenerated and compared. |
| `data/processed/merged_bias_lodo_mbic_{train,validation,test}.json` | A different held-out-dataset evaluation protocol. | Retain; not redundant with random stratified partitions. |
| `data/processed/.gitkeep` | Directory placeholder inside an ignored directory. | Unnecessary for the current repo; output code creates directories. |
| `data/test.txt` | One-line `test` placeholder; no application reference found. | Cleanup candidate, rather than production data. |
| `data/raw/{babe,mbic,basil}/*.csv` | Input corpora. | Retain. Cached adapters and record loaders need these inputs. |
| `runs/<run-id>/`, `runs/benchmarks/<run-id>/`, `runs/models/` | Experiment evidence, benchmark outputs, and saved models. | Retain. Different run IDs do not establish duplication; deletions can remove reproducibility/history. |
| `src/newsbias/web/benchmark_cache.json` | Precomputed analysis responses loaded into the runtime cache at import. | Retain as an application resource; deleting it changes cache-hit results/latency and can trigger model/API calls. Move once to the resource directory. |
| `doc/.gitignore` | Contains only `*`; root `.gitignore` already ignores `doc/`. | Redundant ignore rule/unused documentation placeholder. Use tracked `docs/` for documentation. |
| `frontend/.gitignore` | `node_modules` overlaps the root ignore rule; `.DS_Store`, `dist-ssr`, and `*.local` add coverage. | Keep or consolidate deliberately; the whole file is not redundant. |
| `skills-lock.json` | Development-tool skill provenance, outside application runtime. | Retain if skill tooling is used. Lack of a runtime reference does not prove it is unwanted. |
| `uv.lock` and `frontend/package-lock.json` | Reproducibility for different dependency ecosystems. | Retain both. |
| Package `__init__.py` files and `scripts/__init__.py` | Package/public-export boundaries. Tests import `scripts.merge_bias_datasets`. | Retain needed boundaries. Similar names do not mean duplicate functionality. |
| `.env`, `.git/`, `.github/workflows/tests.yml` | Local configuration, repository metadata, and CI. | Retain their roles. |

The audit covers project-authored source/configuration, visible local artifact groups,
and application references. It does not claim to have compared every downloaded
dependency, Arrow shard, model weight, or historical run payload for content equality.

## 5. Recommended target layout

Keep the existing public package/module locations wherever they are already clear.
Extract shared internals only where there is an actual overlap or an oversized file.
All new Python subpackages include `__init__.py`.

```text
newsbias/
├── pyproject.toml                  # Packaging, extras, existing console commands
├── uv.lock
├── README.md
├── cli.py                          # Existing source-checkout launch wrapper
├── .gitignore
├── .github/workflows/tests.yml
├── skills-lock.json                # Tooling provenance, if still used
├── docs/
│   └── repository-structure-plan.md
├── configs/
│   ├── tfidf_allsides.toml
│   └── benchmark_politicalbiasbert_allsides.toml
├── src/newsbias/
│   ├── __init__.py
│   ├── __main__.py                 # Optional delegator: python -m newsbias
│   ├── cli.py                      # Parser + dispatch; existing public API
│   ├── benchmark_cli.py            # Existing benchmark interface, thin orchestration
│   ├── commands/
│   │   ├── training.py             # Train/eval handlers
│   │   ├── benchmarking.py         # Run/compare + local/remote orchestration
│   │   ├── analysis.py             # Article analysis handler
│   │   └── merge.py                # Canonical merge CLI/implementation
│   ├── registry.py                 # Alias definitions + workflow-specific profiles
│   ├── adapters.py                 # Dynamic adapter loading
│   ├── data.py                     # Shared dataset audits; stable public import
│   ├── evaluation.py               # Shared metrics; preserve output profiles
│   ├── experiment.py               # Training/experiment artifacts
│   ├── benchmark.py                # Inference-only benchmark artifacts
│   ├── cache.py                    # Benchmark artifact lookup, not dataset caching
│   ├── report.py                   # Terminal report formatting
│   ├── prompts.py                  # Public template engine, resource loading
│   ├── _json.py                    # Shared response parsing with explicit modes
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── all_sides.py            # Strict cached AllSides adapter
│   │   ├── babe.py                 # Strict cached BABE adapter
│   │   ├── political.py            # Existing record API + held-out-outlet policy
│   │   ├── bias.py                 # Existing record API + dataset-priority merge
│   │   ├── merged_bias.py          # Prepared merged-data adapter
│   │   ├── splitting.py            # Named record and DatasetDict split policies
│   │   ├── cache.py                # Atomic prepared-dataset cache
│   │   └── _records.py             # Shared mechanics, explicit normalization policies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── tfidf.py
│   │   ├── huggingface.py          # Dataset adapter
│   │   ├── political.py            # Record API, task-specific defaults/formatting
│   │   ├── bias.py                 # Sentence policy + mock API
│   │   ├── openai.py               # Dataset adapter + retries/concurrency
│   │   ├── llm.py                  # Sentence/article/joint task API
│   │   ├── _transformer.py         # Shared loading/window mechanics
│   │   └── _openai_client.py       # Shared client construction
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   └── segmenter.py
│   ├── remote/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cli.py                  # Arbitrary-job interface retained
│   │   ├── kaggle.py               # Bundles canonical shared modules with jobs
│   │   └── scripts/run_transformer_kaggle.py
│   ├── web/
│   │   ├── __init__.py
│   │   ├── app.py                  # App assembly + existing endpoint imports
│   │   ├── analysis.py             # Tribunal/rewrite services, if split from app
│   │   └── schemas.py              # Existing request contracts
│   └── resources/
│       ├── prompts/                # Only authored built-in template directory
│       │   ├── bias_detection.prompt.j2
│       │   ├── bias_removal.prompt.j2
│       │   ├── zero_shot_bias.prompt.j2
│       │   └── ...                 # Distinct extracted inline prompts, one each
│       ├── samples.json            # Curated samples extracted from app.py
│       ├── benchmark_cache.json    # Existing precomputed responses, unchanged
│       └── static/                 # Ignored/generated frontend build output
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── index.html
│   ├── vite.config.ts              # Build directly into resources/static/
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/{App.tsx,main.tsx,index.css}
├── scripts/
│   ├── __init__.py
│   └── merge_bias_datasets.py      # Compatibility delegator/re-exports
├── tests/                          # Existing tests; regroup only if useful
├── data/                           # Local/ignored; existing paths retained
│   ├── raw/{babe,mbic,basil}/
│   ├── cache/
│   └── processed/
└── runs/                           # Local/ignored; existing defaults retained
    ├── <experiment-run-id>/
    ├── benchmarks/<run-id>/
    └── models/
```

Generated `.venv/`, `node_modules/`, and tool caches are omitted from this authored
layout. The frontend source and its compiled static resources serve different roles.
Build directly into the final static directory so there is one build destination,
rather than maintaining both `frontend/dist/` and a manually copied backend build.

The target intentionally retains `experiment.py` and `benchmark.py`, both cache
modules, and both dataset/model interfaces: they implement genuinely different
behaviors. Shared helpers remove repeated mechanics, rather than forcing every
workflow through a behaviorally different implementation.

## 6. Migration sequence and functionality invariants

### Phase A — Establish the behavior baseline

1. Record current CLI parser behavior for all three console commands and root
   `cli.py`, including aliases, default output directories, and legacy `--config` mode.
2. Capture deterministic dataset IDs, labels, row order, split membership, both hash
   policies, merge-priority provenance, and held-out split behavior with fixed inputs.
3. Capture rendered system/user prompt messages and OpenAI request kwargs with fake
   clients; API model output is nondeterministic and is not a suitable equality test.
4. Capture API payloads, cached sample responses, rewrite fallback behavior, sentence
   offsets, mock predictions, and artifact/report formats.
5. Run the existing suite and lint checks with the dependencies needed by those tests.
   Record pre-existing failures before attributing a failure to restructuring.

### Phase B — Remove prompt duplication and centralize resources

1. Move the three canonical package templates into `resources/prompts/` without
   modifying their bytes. Update built-in lookup while retaining custom overrides.
2. Move `web/benchmark_cache.json` once into `resources/`. Preserve all keys/values
   and cache indexing behavior. Extract curated samples without changing their order.
3. Configure resource inclusion in Hatchling. Route Vite output directly to
   `resources/static/`, then resolve that directory through the resource loader.
4. Update the web asset mount and index-serving paths together. Preserve the current
   missing-build fallback response as well as the built UI behavior.
5. Verify source-checkout and installed-wheel execution outside the repo, then remove
   the three root prompt copies and obsolete `frontend/dist/` output.

The current web path walks four directory levels to locate `frontend/dist/`; an
installed wheel does not normally have that checkout layout. Resource consolidation
should address this deployment boundary as well as duplicate prompt maintenance.

### Phase C — Extract shared mechanics, preserving policies

1. Consolidate the identical record shuffle mechanics with configurable grouping
   keys. Preserve RNG call order, group iteration order, rounding, and error behavior.
2. Extract shared transformer mechanics while explicitly preserving model IDs,
   label mappings, precision/rounding, window limits, sentence truncation, and mocks.
   In particular, the pipeline's `premsa/...` default and CLI/web metadata mentioning
   `matous-volf/...` currently differ; changing checkpoints is a separate behavior change.
3. Consolidate OpenAI client creation/JSON mechanics using caller-specific options.
   Preserve dotenv precedence, retries, token settings, temperatures, fallback values,
   validation, and error handling. Move inline instructions without rewriting them.
4. Extract merge orchestration. Preserve the richer script's flags/export naming and
   current weak-corpus behavior, including its existing exclusion policy.
5. Centralize aliases without changing accepted aliases or either workflow's defaults.

### Phase D — Thin entrypoints and remote packaging

1. Move oversized CLI handlers into `commands/`; retain `main`, public helpers,
   entrypoint strings, existing TOML adapter paths, and compatibility imports.
2. Keep root `cli.py` and `scripts/merge_bias_datasets.py` as delegators to one
   implementation. Re-export functions used by callers/tests as necessary.
3. Extract web services without changing routes, schemas, cache semantics, headings,
   response keys, or fallback handling.
4. Bundle canonical shared modules with Kaggle jobs before replacing self-contained
   algorithms. Verify staging includes dependencies and the runner works without a
   source checkout. Generated upload bundles are deployment artifacts, not maintained
   duplicate source files.
5. Preserve distinct remote training/evaluation policies and legacy output schemas.
   Do not accidentally replace the remote all-window evaluator with a capped one.

### Phase E — Cleanup and completion checks

- Remove the redundant `doc/.gitignore` placeholder and unneeded processed `.gitkeep`.
  Remove `data/test.txt` if its placeholder status is confirmed by the project owner.
- Keep raw inputs, precomputed responses, saved models, and historical experiments.
- Keep `data/` and `runs/` defaults during this migration; path normalization or
  historical artifact archival can be addressed separately if desired.
- Update README build/install instructions and CI to exercise frontend packaging and
  dependencies required by prompt/web tests. Currently CI syncs only base/dev deps,
  but tests import FastAPI/Jinja2 and `prompts.py` imports PyYAML, which is not declared
  directly in `pyproject.toml`. Declare actual dependencies in appropriate extras and
  synchronize CI accordingly as part of packaging verification.
- Preserve provenance: source reorganization necessarily changes code hashes and
  new run identities. Keep historical artifacts and processor-version metadata;
  verify recomputed datasets match rather than rewriting old cache metadata.

Suggested checks when implementing the migration:

```bash
uv sync --extra web --extra llm --extra transformers
uv run ruff check .
uv run pytest
npm ci --prefix frontend
npm run build --prefix frontend
uv build
```

Additionally install the built wheel into an isolated environment outside the
checkout and check imports, all console-command help output, built-in prompt loads,
precomputed sample responses, and serving `/` plus `/assets/...`. Mock external
providers for equality checks. Test Kaggle bundle staging and local execution of the
staged entrypoint before a real remote smoke run.

Completion means the same user commands, adapter imports, input/output contracts,
deterministic data partitions, model settings, prompt messages, and cached behavior
are preserved, with one authored copy of each resource and one implementation of
each shared algorithm. Compatibility wrappers are delegation, not redundancy.
