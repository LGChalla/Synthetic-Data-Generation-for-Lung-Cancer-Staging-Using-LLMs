# StageCraft: Neuro-Symbolic Synthetic Clinical Data Generation for Lung Cancer TNM Staging

*What happens when the data you need to build a clinical AI system is exactly the data you can't have?*

Lung cancer kills more people than any other cancer. Survival swings from 80% at Stage I to under 20% at Stage IV — and three in four patients are diagnosed at the late stages. The models that could change that need structured, annotated clinical notes. Those notes are locked behind privacy law, annotation bottlenecks, and institutional walls that take years to navigate.

This project builds the data instead. Not as a shortcut — as a principled engineering commitment.

---

## The Pipeline

Four phases. Each one gates the next.

```
[Phase 1] Generate          →  RAG-grounded LLM synthesis across 639 runs
              ↓ G(x) gate: JSON schema + SNOMED CT + AJCC logic
[Phase 2] Audit             →  Shannon entropy per T / N / M — catches silent label collapse
              ↓ entropy gate: H_T, H_N ≥ 1.109 · H_M ≥ 0.554
[Phase 3] Benchmark         →  TSTR on real TCGA pathology reports (pathologist-assigned TNM)
[Phase 4] Fine-tune         →  QLoRA adapters on 162 certified records
```

No record reaches fine-tuning without passing both gates.

---

## What This Found

**The label-collapse failure.** All 128 ablation rows were seeded from a single patient — "65yo Male, T2N0M0." Every record passed schema validation, SNOMED coverage, and AJCC logic. The corpus looked clean. The adapter predicted T2N0M0 for everything. The failure was distributional, not structural, and nothing in the existing literature catches it.

**The fix:** a 32-cell TNM grid ({T1–T4} × {N0–N3} × {M0, M1}), round-robin seeded, with entropy gates enforced before training.

**The result on real clinical notes.** Evaluated train-on-synthetic, test-on-real against TCGA pathologist-assigned labels across three test sets — synthetic held-out (n=39), TCGA-Lung (n=737), and TCGA cross-tumor (n=3,161):

| Metric (real TCGA notes) | Zero-Shot Baseline | Adapter B |
|--------------------------|--------------------|-----------|
| N-stage accuracy — Cross-Tumor (n=3,161)  | 45.1% | **71.9%** (+26.8 pp) |
| Minority T4 accuracy — Cross-Tumor        | 61.0% | **82.7%** (+21.7 pp) |
| T-stage macro-F1 — Cross-Tumor            | 0.394 | **0.509** |
| N-stage accuracy — TCGA-Lung (n=737)      | 51.0% | **71.8%** (+20.8 pp) |
| Minority T4 accuracy — TCGA-Lung          | 68.4% | **81.6%** (+13.2 pp) |

Read the macro-F1, not the aggregate. TCGA reports state TNM explicitly, so the zero-shot baseline is already strong on the easy majority classes (T1/T2). Adapter B is the only system to clear the N-stage constant-classifier ceiling on real data — and it does so by deliberately trading a few points of majority-class T-stage accuracy for large minority-T4 and N-stage gains. Aggregate T-stage accuracy alone would misread that as a regression; macro-F1 and per-class accuracy are the honest metrics under this kind of class imbalance.

**The deeper finding — label diversity is necessary but not sufficient.** Adapter A′ — trained on the same 162 records with their (T, N, M) triples permuted across notes (a derangement, seed 42), so Shannon entropy is *identical* to Adapter B — doesn't just underperform. It collapses. It frequency-matches a single dominant class (T4: 0.978 accuracy) while every other T-stage falls below 6%, dragging T-stage macro-F1 from 0.509 down to 0.090 (a 5.7× gap) and T3 accuracy from 0.713 to 0.045. Same corpus, same entropy, same hyperparameters — the only thing destroyed is the correspondence between what a note describes and the label it carries. That correspondence, not label variety, is the operative variable for minority-class recovery.

---

## Repository

```
pipeline/           Four-phase pipeline (datagen → analysis → benchmark → finetune)
preprocessing/      TCGA pathology-report extraction and real-world TSTR setup
experiments/        Prompt engineering · Model comparison · Longitudinal generation
utils/              BioPortal SNOMED CT annotation
data/               Ablation design CSVs (128-cell full factorial · 11-point OFAT)
```

**Run order:** `phase1_datagen.py` → `phase2_analysis.py` → `tcga_prep.py` → `phase3_benchmark.py` → `phase4_finetuning.py`

---

## What Is Next

- **Colorectal and breast cancer** — TNM grids and staging rules exist; the architecture transfers
- **Demographic stratification** — 91.5% of generated records are Male; a demographic grid is the fix
- **MedCPT Cross-Encoder reranking** — single-chunk retrieval collapsed to one paper (80.5% of all retrievals); expanding the FAISS index to two PubMed chunks mitigated it, and cross-encoder reranking would harden retrieval diversity further
- **Multi-institutional evaluation** — MIMIC-III and eICU to test whether synthetic-to-real transfer holds at scale

---

*Paper: IEEE BIBM 2026*  
*All generated data is synthetic and not intended for clinical use, YET.*
