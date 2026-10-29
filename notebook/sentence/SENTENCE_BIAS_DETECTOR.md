# NewsBias Sentence Model: Training Notebooks & Artifacts

## Model & Training Recipe

* **Backbone:** `jhu-clsp/ettin-encoder-400m` (ModernBERT-style encoder, 28 layers, width 1,024, 16 attention heads, 395,833,346 parameters).
* **Task:** Sentence linguistic bias classification (`0 = non-biased`, `1 = biased`).
* **Corpus:** 11,540 deduplicated sentences (BABE + BASIL + MBIC).
  * Train: 8,078 sentences
  * Validation: 1,154 sentences
  * Held-Out Test: 2,308 sentences
* **Hardware:** Google Colab Tesla T4 GPU (FP16 mixed precision, gradient checkpointing).
* **Batching:** Microbatch 4, accumulation 8 (effective batch size: 32).
* **Optimizer:** AdamW (encoder LR `1e-5`, head LR `1e-4`, weight decay 0.01, 6% linear warmup, linear decay).
* **Loss:** Source-aware weighted Cross-Entropy (BABE 2.0x, MBIC 1.0x, BASIL 0.5x, class weighting).

