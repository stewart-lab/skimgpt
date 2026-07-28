# CPU-runnable multimodal models for dense scientific-figure transcription

Research note for [issue #123](https://github.com/stewart-lab/skimgpt/issues/123), part of the
[#120 map](https://github.com/stewart-lab/skimgpt/issues/120).
Written 2026-07-28.

---

## 1. Headline

**A CPU-only path exists and is narrower than it looks — but the decisive finding is not about
CPU at all. It is that dense structured transcription of figures is unsolved at every model scale,
including Gemini's.**

- **Top candidate: `Qwen3-VL-4B-Instruct`** (Apache 2.0, official GGUF, llama.cpp `libmtmd`).
  Best-in-class among sub-5B open models on the only realistic scientific-figure benchmark
  (CharXiv: DQ 76.2 / RQ 39.7), and one of only two families in this size class with a published
  CharXiv score at all.
- **Throughput: feasible, but only just, and only on modern hardware.** I measured Qwen3-VL-2B
  Q4_K_M end-to-end on a real PMC western-blot figure under llama.cpp on CPU:
  **280 s wall-clock on 8 threads ≈ 2,240 core-seconds/figure** — but on a 2015 Xeon that is
  6-20x slower than current server silicon. Normalised to a modern Xeon that is **~300
  core-seconds/figure**, which puts a full 40M-figure pass at **~33 days** of a realistic CHTC
  allocation. Cap the output at 300 tokens and it drops to ~19 days; step up to the 4B model and it
  roughly doubles to ~67 days. See
  [§6.2](#62-measured-qwen3-vl-2b-instruct-q4_k_m-on-llamacpp-cpu-only).
- **The dominant cost is token generation, not vision encoding** — ~78% of my wall-clock was
  decode. **Output length is therefore the biggest throughput lever there is**, which makes the
  map's open "dense-prompt design / token budget" question a throughput decision, not a quality
  one.
- **The quality gap depends entirely on which question you ask.**
  - *Reasoning about* figures: Gemini 3 Flash 80.3 CharXiv RQ vs Qwen3-VL-4B-Instruct 39.7. Gemini
    is at the human baseline (80.5); the open model is at half of it. A chasm.
  - *Densely serialising* figures: on OCRBench v2's Element Parsing sub-task — the only genuinely
    generative structured-transcription metric published anywhere — **Gemini 3 Pro Preview scores
    27.1 and GPT-5 scores 24.8**, against ~13-22 for open 7-8B models. Nobody is good. See
    [§4.3](#43-the-most-important-number-in-this-document-ocrbench-v2-element-parsing).
- **Nothing measures the biomedical half of the prompt.** No benchmark anywhere evaluates
  free-form description of western blots, gels, micrographs, or multi-panel composites. The closest
  is MMSci's caption task on Nature Communications figures, where **GPT-4o scores BLEU-2 ≈ 4.93**.
  A head-to-head on real PMC figures is required and is on the critical path.
- **If the CPU constraint binds, GPU changes the arithmetic by roughly an order of magnitude** —
  and CHTC's GPU Lab is ~62 GPUs, of which a "short" job may occupy up to 2/3.
  See [§8](#8-the-gpu-counterfactual).

---

## 2. What the target has to do

Verified against `origin/fulltext`, not re-derived.

| Aspect | Current implementation |
| --- | --- |
| Model | `gemini-3-flash-preview` (`skimgpt/image_analyzer.py`, default `model_name` arg) |
| Reasoning | `thinking_config=types.ThinkingConfig(thinking_level="high")` |
| Image fidelity | `media_resolution={"level": "media_resolution_high"}` on the image `Part` |
| Prompt | `get_transcription_prompt(hypothesis)`, `skimgpt/prompt_library.py:639` |
| Batch fallback | vLLM under HTCondor docker universe, `request_gpus: 1`, `request_cpus: 1`, `request_memory: 15GB`, `request_disk: 15GB`, `gpus_minimum_memory: 10GB`, `requirements: (CUDACapability >= 8.0)`, `+GPUJobLength: "short"` (`skimgpt/htcondor_helper.py:135`) |
| CPU viability of that fallback | Nil — `skimgpt/relevance_chtc.py` raises `RuntimeError` if `not torch.cuda.is_available()` before vLLM is constructed |

The transcription prompt asks for six things:

1. figure type identification,
2. all visible text, labels and legends,
3. visual patterns / trends / relationships,
4. **estimated values of data points** for charts,
5. **key observations** for blots and microscopy,
6. a detailed structured transcription capturing all scientific information.

Two properties of that prompt drive everything below:

- **It is free-form and instruction-following.** This immediately disqualifies the entire
  document-OCR-specialist family (see [§5.2](#52-the-ocr-specialist-trap)), which only responds
  to a handful of literal training prompts.
- **It demands long, dense output.** That is the throughput killer on CPU. Prefill of a big image
  is expensive, but generating 600-1000 tokens at CPU decode rates is worse.

**Scope as of the #124 resolution:** all of PMC — order 5-6M articles, order 30-50M figures, one
full pass plus re-population whenever the prompt or model changes. **No commercial model may be
used for population.** Gemini is a quality yardstick only.

---

## 3. Summary table

Parameter counts are `safetensors.total` from the Hugging Face API unless noted. Licenses are the
HF `cardData.license` field cross-checked against the model card text.

| Model | Params | License | Released | Official GGUF? | CharXiv DQ | CharXiv RQ | OCRBench | Follows a free-form prompt? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-VL-4B-Instruct** | 4.44B | Apache 2.0 | 2025-10-15 | ✅ Qwen + ggml-org | **76.2** | **39.7** | 881 | ✅ |
| **Qwen3-VL-4B-Thinking** | 4.44B | Apache 2.0 | 2025-10-15 | ✅ Qwen | **83.9** | **50.3** | 808 | ✅ |
| **Qwen3-VL-2B-Instruct** | 2.13B | Apache 2.0 | 2025-10-21 | ✅ Qwen + ggml-org | 62.3 | 26.8 | 858 | ✅ |
| **Qwen3-VL-8B-Instruct** | 8B | Apache 2.0 | 2025-10-15 | ✅ Qwen | 83.0 | 46.4 | 896 | ✅ |
| InternVL3.5-4B | 4B | Apache 2.0 | 2025-08 | community | 71.1 | 39.6 | 815 | ✅ |
| **granite-vision-3.3-2b** | 2.98B | Apache 2.0 | 2025-06-03 | ✅ IBM | not reported | not reported | 790 | ✅ |
| **gemma-4-E4B-it** | 4.5B effective (8B w/ embeddings) | Apache 2.0 | 2026-03-02 | ✅ ggml-org | not reported | not reported | not reported | ✅ |
| **gemma-4-E2B-it** | 2.3B effective (5.1B w/ embeddings) | Apache 2.0 | 2026-03-02 | ✅ ggml-org | not reported | not reported | not reported | ✅ |
| PaddleOCR-VL | 0.96B | Apache 2.0 | 2025-10-16 | ✅ Baidu | ✗ | ✗ | ✗ | ❌ fixed prompts |
| DeepSeek-OCR | 3.34B | MIT | 2025-10-17 | ✅ ggml-org | ✗ | ✗ | ✗ | ⚠️ two prompts only |
| dots.ocr | 3.04B (card says 1.7B — LLM only) | MIT | 2025-07-30 | ✅ ggml-org | ✗ | ✗ | ✗ | ❌ fixed prompts |
| Nanonets-OCR2-3B | 3.75B | ⚠️ **Qwen Research License** | 2025-10-13 | community only | not reported | not reported | not reported | ✅ (Qwen2.5-VL-3B SFT) |
| Florence-2-base/large | 0.23B / 0.77B | MIT | 2024-06 | ❌ no llama.cpp | ✗ | ✗ | ✗ | ❌ task tokens only |
| ChartGemma / TinyChart | ~2.9B / 3B | ⚠️ unreliable / absent | 2024 | ❌ none | 21.3 / 16.2 | 12.5 / 8.3 | — | chart-domain only |
| **Reference: Gemini 3 Flash** | — | proprietary | 2025-12 | — | — | **80.3** | — | ✅ |
| **Reference: human** | — | — | — | — | **92.1** | **80.5** | — | — |

CharXiv DQ/RQ = descriptive / reasoning accuracy. **DQ is the transcription-shaped half** and is
the column to weight most heavily. ✗ = the model architecturally cannot do the task, so no score
exists. "not reported" = the model could do it, but nobody has published the number. All
model-family numbers are self-reported by the model authors — see
[§4.7](#47-a-caveat-on-every-self-reported-number-above).

---

## 4. Quality evidence

### 4.1 CharXiv is the right proxy, and almost nothing else is

[CharXiv](https://charxiv.github.io/) ([arXiv:2406.18521](https://arxiv.org/abs/2406.18521)) is
2,323 charts **taken from arXiv papers** — i.e. real scientific figures with real multi-panel
layouts, real small-font axis labels, and real legend clutter. It splits into:

- **Descriptive (DQ)** — extract basic chart elements: labels, enumeration, counting, pattern
  recognition.
- **Reasoning (RQ)** — synthesise across the figure. This is the hard half.

Human baseline: **80.50% RQ, 92.10% DQ** (official leaderboard,
[princeton-nlp.github.io/CharXiv](https://princeton-nlp.github.io/CharXiv/)). The paper's headline
finding was that the best proprietary model at the time (GPT-4o) hit 47.1% RQ and the best open
model (InternVL Chat V1.5) hit 29.2%.

**The descriptive half is the part that matters most here.** CharXiv-DQ is built from 19 templated
question types — plot title, x/y axis label, leftmost/rightmost labelled tick, tick spacing, number
of lines, whether lines intersect, number of legend entries, **the names of the legend labels**,
colorbar range, general trend, total labelled ticks, **subplot layout ("n by m")**, number of
subplots. That is very nearly *dense transcription decomposed into independently gradeable
primitives*, including multi-panel decomposition. It is the closest published proxy that exists for
what `get_transcription_prompt()` asks for.

**Caveat on subject mix:** CharXiv is arXiv, and ~12.6% of it is `q-bio` — but that is
*quantitative* biology (models, phylogenies, population plots), **not** wet-lab blots or
micrographs. Notably, on the authors' own published per-question gradings, `q-bio` is consistently
at or near the bottom for models while humans find it easy (GPT-4o descriptive: 80.6 on q-bio vs
86.7 on cs; human: 93.1 on q-bio). A modest but real domain penalty, in the wrong direction.

Why it matters here: CharXiv is the *only* widely-reported benchmark whose images look like the
images this pipeline actually sees. ChartQA is synthetic-ish and saturated; DocVQA and InfoVQA are
business documents and infographics; AI2D is grade-school diagrams; OCRBench is text spotting.

The CharXiv authors built it precisely because the easier chart benchmarks flatter small open
models. From the abstract:

> We demonstrate that although open-source models can appear to outperform strong proprietary
> models on these benchmarks, a simple stress test with slightly different charts or questions can
> deteriorate performance by up to 34.5%.

**Keep that sentence in mind every time a model card quotes ChartQA 0.87 and nothing else.**

### 4.2 The numbers

Gemini 3 Flash, from the official
[Gemini 3 Flash Model Card (December 2025)](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf):

| Benchmark | Gemini 3 Flash (Thinking) |
| --- | --- |
| CharXiv Reasoning (no tools) | **80.3%** |
| MMMU-Pro | 81.2% |
| OmniDocBench 1.5 (edit distance, lower better) | 0.121 |
| ScreenSpot-Pro | 69.1% |

Qwen3-VL, from the official benchmark tables in the
[QwenLM/Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL) (the tables are published as images;
these values were read directly off
[`qwen3vl_2b_32b_vl_instruct.jpg`](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_2b_32b_vl_instruct.jpg)
and
[`qwen3vl_2b_32b_vl_thinking.jpg`](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_2b_32b_vl_thinking.jpg)):

**Instruct variants**

| Benchmark | 2B | 4B | 8B | 32B | Qwen2.5-VL-72B | GPT-5-Mini (Minimal) | Claude 4 Sonnet (no thinking) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CharXiv DQ | 62.3 | 76.2 | 83.0 | 90.5 | 87.4 | 78.6 | 87.8 |
| **CharXiv RQ** | **26.8** | **39.7** | **46.4** | **62.8** | 49.7 | 48.9 | 60.9 |
| OCRBench | 858 | 881 | 896 | 895 | 885 | 807 | 741 |
| OCRBench v2 (en/zh) | 56.3/53.0 | 63.7/57.6 | 65.4/61.2 | 67.4/59.2 | 61.5/63.7 | 45.7/41.0 | 44.6/36.2 |
| DocVQA test | 93.3 | 95.3 | 96.1 | 96.9 | 96.4 | 91.0 | 54.0 |
| InfoVQA test | 72.4 | 80.3 | 83.1 | 87.0 | 87.3 | 73.0 | 30.0 |
| AI2D test | 76.9 | 84.1 | 85.7 | 89.5 | 88.7 | 82.9 | 82.6 |
| MMMU val | 53.4 | 67.4 | 69.6 | 76.0 | 70.2 | 67.9 | 75.1 |
| MMLongBench-Doc | 31.6 | 43.5 | 47.9 | 55.4 | 42.1 | 39.6 | 52.6 |
| HRBench4K / 8K | 74.1/68.4 | 78.3/72.9 | 78.9/74.6 | 82.9/77.8 | 67.6/68.0 | 66.3/60.9 | 50.6/40.5 |

**Thinking variants**

| Benchmark | 2B | 4B | 8B | 32B |
| --- | --- | --- | --- | --- |
| CharXiv DQ | 70.1 | 83.9 | 85.9 | 90.2 |
| **CharXiv RQ** | **37.1** | **50.3** | **53.0** | **65.2** |
| OCRBench | 792 | 808 | 819 | 855 |
| OCRBench v2 (en/zh) | 56.4/51.9 | 61.8/55.8 | 63.9/59.2 | 68.4/62.1 |
| DocVQA test | 92.9 | 94.2 | 95.3 | 96.1 |
| MMMU val | 61.4 | 70.8 | 74.1 | 78.1 |

Two things to read off this:

1. **Thinking mode buys ~11 points of CharXiv RQ at 4B** (39.7 → 50.3) and ~14 points of DQ
   (76.2 → 83.9). It also costs several hundred extra generated tokens per image, which on CPU is
   exactly the thing you cannot afford. This is a direct quality/throughput trade you will have to
   make explicitly.
2. **OCRBench barely moves with scale** (858 → 896 from 2B to 8B) while **CharXiv RQ nearly
   doubles** (26.8 → 46.4). Raw text spotting is close to solved at 2B; *understanding the figure*
   is not. Do not let a good OCRBench number talk you into a small model.

### 4.3 The most important number in this document: OCRBench v2 "Element Parsing"

Everything above is short-answer QA. **OCRBench v2** ([arXiv:2501.00321](https://arxiv.org/abs/2501.00321),
[live leaderboard](https://99franklin.github.io/ocrbench_v2/)) contains one sub-capability that is
*genuinely generative structured transcription*: **Element Parsing** — table parsing, **chart
parsing**, document parsing and formula recognition, scored by **TEDS** (tree-edit distance over
the generated HTML/Markdown/JSON structure).

Scores on that column, from the official leaderboard:

| Model | OCRBench v2 EN avg | **Element Parsing (EN)** |
| --- | --- | --- |
| Qwen2.5-VL-7B | 41.8 | **13.1** |
| InternVL2.5-8B | 40.5 | 20.3 |
| MiniCPM-V-4.6 (1B) | 40.4 | 20.7 |
| InternVL3.5-8B | 46.0 | 21.5 |
| InternVL3-8B | 45.3 | 22.4 |
| LLaVA-OneVision-2-8B | 47.6 | 22.1 |
| **GPT-5** | 55.5 | **24.8** |
| **Gemini 3 Pro Preview** | 63.4 | **27.1** |
| Phi-4-multimodal (5.6B) | 38.1 | 38.7 |
| dots.mocr (3B) | 43.3 | 40.3 |

Read that table twice. **Dense structured transcription is not a "small model" problem — it is an
unsolved problem at every scale.** Models score 70-93 on Recognition and Extraction and collapse to
the 20s on Parsing. Gemini 3 Pro Preview manages 27.1. The only models above 38 are purpose-built
document parsers, and those are precisely the fixed-prompt models that cannot follow this pipeline's
prompt ([§5.2](#52-the-ocr-specialist-trap)).

This tempers the entire framing of the #120 map. The gap between Gemini and a 4B open model is
large on *reasoning about* figures (CharXiv RQ 80.3 vs 39.7), but on *densely serialising* a figure
into structured text, **nobody is good, and the frontier lead is much smaller.** If the corpus
target is "dense transcription" rather than "figure Q&A", the open-vs-closed gap may be narrower
than the CharXiv headline suggests — which is an argument *for* the project, and also a warning
that the ceiling is low in absolute terms regardless of which model you pick.

### 4.4 The one benchmark with real biomedical figures

**MMSci** ([arXiv:2407.04903](https://arxiv.org/abs/2407.04903),
[repo](https://github.com/Leezekun/MMSci)) — 131,393 articles / 742,273 figures from **Nature
Communications**, 72 disciplines, of which roughly 40 are biological/biomedical; biology figures are
~40-45% of the collection. It explicitly includes *"schematic diagrams, **microscopic images**, and
experimental data"* — i.e. real multi-panel wet-lab composites, the actual target distribution.

It has a **genuinely free-form caption-generation task** with reference captions averaging **~153
words**, scored with BLEU / ROUGE-L / METEOR / BERTScore / CIDEr / FActScore / G-Eval. And the
result is sobering:

| Model | BLEU-2 (figure only) | BLEU-2 (abstract-grounded) |
| --- | --- | --- |
| GPT-4o | **4.93** | 5.57 |
| Qwen2-VL-7B fine-tuned on MMSci | 19.13 | 21.77 |

**GPT-4o scores BLEU-2 ≈ 5 at writing a Nature-Communications-grade figure description.** A 7B model
*fine-tuned on the task* reaches 19. This is the single most relevant published datapoint in this
whole survey, and what it says is: **frontier general-purpose models cannot do dense biomedical
figure description well zero-shot, and domain fine-tuning is worth ~4x on this metric.**

On MMSci's multiple-choice matching tasks, for context: Qwen2-VL-7B averages 72.87, GPT-4o 79.57,
Claude 3.5 Sonnet 80.18 — and **PhD experts given a 1-minute limit average 69.51**, i.e. below the
frontier models. The MCQ format flatters everybody; the generative format flatters nobody.

### 4.5 Benchmarks that look relevant but are not

- **SciFIBench** ([arXiv:2405.08807](https://arxiv.org/abs/2405.08807)) — scientific figure↔caption
  matching, but it is **5-way multiple choice**, and two facts destroy it as a proxy: a plain
  **CLIP ViT-H-14-378 embedding scores 36.2**, beating every open LMM tested except one; and
  **text-only Gemini 1.5 Flash fed OCR output scores 49.5**, beating every open model by 10+ points.
  It measures embedding similarity and matching, not reading.
- **ChartQAPro** ([arXiv:2504.05506](https://arxiv.org/abs/2504.05506)) — genuinely much harder than
  ChartQA (Claude Sonnet 3.5: ChartQA 90.50 → CharXiv 60.20 → ChartQAPro 55.81), and small models
  crater on it (SmolVLM-2.3B 19.14, InternVL2.5-2B 17.81, ChartGemma-3B 6.84, human 85.02). But its
  answers average **1.18 tokens**, and its charts are journalism/business dashboards and
  infographics — no scientific-paper figures at all.
- **MicroVQA**, **μ-Bench**, **PMC-VQA**, **AI2D**, **ScienceQA** — all multiple choice, chance
  floors 20-25%. MicroVQA in particular is noted to be largely a *language* benchmark: small open
  LLMs only slightly underperform top models on it.
- **BiomedCLIP / BIOMEDICA / PMC-15M** — enormous PMC-derived corpora (BIOMEDICA is 24M image-text
  pairs from 6M PMC-OA articles), but evaluated **only** on zero-shot classification and image-text
  retrieval. Not generative at all.

### 4.6 The benchmark that does not exist

**There is no benchmark that measures dense free-form transcription of biomedical paper figures.**
Format audit of everything surveyed:

| Benchmark | Format | Answer length | Biomedical paper figures? |
| --- | --- | --- | --- |
| **CharXiv descriptive** | free-form short answer, 19 templates, GPT-4o-graded | 1-223 chars | No (arXiv q-bio only) |
| **OCRBench v2 Element Parsing** | **generative structured output**, TEDS | full HTML/MD/JSON | No |
| **MMSci caption generation** | **free-form long-form** | **~153 words** | **Yes** (Nature Comms) |
| CharXiv reasoning | free-form short answer | 1-58 chars | No |
| ChartQAPro | 76.5% free-form / 23.5% MCQ | **1.18 tokens** | No |
| ChartQA / DocVQA / InfoVQA / TextVQA | short answer | 1-3 tokens | No |
| SciFIBench | **5-way MCQ** | one letter | Partly (arXiv q-bio) |
| MicroVQA / μ-Bench / PMC-VQA-MC / AI2D / ScienceQA | **MCQ** | one letter | Yes, but MCQ |
| BiomedCLIP / BIOMEDICA | retrieval + classification | — | Yes, but not generative |
| OmniDocBench 1.5 | edit distance vs page linearization | full page | No |

So for the three things the prompt explicitly asks for that are *specific to this corpus* —
**western blot band descriptions, microscopy observations, and multi-panel decomposition** — there
is **no published free-form evaluation of any model, open or closed.** Not for Qwen3-VL, not for
Granite, not for Gemini. No western-blot lane/band readout task exists. No microscopy panel
description task with generative scoring exists. No multi-panel composite serialisation task
exists.

That is a finding, and it has a consequence: **the map cannot resolve the quality question from
published numbers alone. It needs a head-to-head run on real PMC figures.** That should be its own
ticket, and it is on the critical path.

If someone builds that eval, the design to copy is **CharXiv's descriptive template set** (19
gradeable transcription primitives, including subplot layout and legend label enumeration) applied
to **MMSci's figure pool** (real Nature Communications multi-panel biomedical figures with
expert-written ~153-word ground truth).

The nearest existing biomedical signal is `MedXPertQA MM`, reported on the
[gemma-4-E4B-it card](https://huggingface.co/google/gemma-4-E4B-it) — E4B scores 28.7%, Gemma 4 31B
scores 61.3%. It is medical-exam MCQ, not figure transcription, but a 2x spread between a 4B and a
31B model on medical imagery is a warning sign for small models in this domain.

### 4.7 A caveat on every self-reported number above

Model tech reports and official leaderboards disagree, sometimes badly. The clearest documented
case: **Qwen self-reports OCRBench v2 56.3 EN / 57.2 ZH for Qwen2.5-VL-7B; the official OCRBench v2
leaderboard measures the same model at 41.8 / 49.5** — a ~15-point gap.

Every Qwen3-VL number in §4.2 is **self-reported by Qwen**, because no independent evaluation of
Qwen3-VL on CharXiv has been published. The CharXiv leaderboard itself is frozen at 34 models from
mid-2024 and contains no model released after that point. **Treat the 39.7 CharXiv RQ for
Qwen3-VL-4B as an upper bound, not a measurement.** The Gemini 3 Flash 80.3 is likewise
self-reported by Google. The comparison is therefore self-report vs self-report, which is fair, but
neither number is independently replicated.

### 4.8 Cross-family comparison at similar sizes

For completeness, the sub-8B field on CharXiv (all from model tech reports; sources in the rows
above and [InternVL3.5 Table 4](https://arxiv.org/html/2508.18265v1),
[InternVL3 Table 3](https://arxiv.org/html/2504.10479v1),
[Qwen2.5-VL Table 5](https://arxiv.org/abs/2502.13923)):

| Model | CharXiv RQ | CharXiv DQ | OCRBench |
| --- | --- | --- | --- |
| InternVL3-1B | 21.0 | 47.1 | 790 |
| Qwen3-VL-2B-Instruct | 26.8 | 62.3 | 858 |
| InternVL3.5-1B | 26.9 | 60.6 | 795 |
| InternVL3-2B | 28.3 | 54.7 | 835 |
| Qwen2.5-VL-3B | 31.3 | 58.6 | 797 |
| InternVL3.5-2B | 31.6 | 65.0 | 836 |
| Phi-3 Vision (4B) | 31.6 | 60.5 | not reported |
| InternVL3.5-4B | 39.6 | 71.1 | 815 |
| **Qwen3-VL-4B-Instruct** | **39.7** | **76.2** | **881** |
| Qwen2.5-VL-7B | 42.5 | 73.9 | 864 |
| InternVL3.5-8B | 44.4 | 72.2 | 832 |
| **Qwen3-VL-8B-Instruct** | **46.4** | **83.0** | **896** |
| ChartGemma (3B) | 12.5 | 21.3 | not reported |
| TinyChart (3B) | 8.3 | 16.2 | not reported |

**Qwen3-VL-4B ties InternVL3.5-4B on reasoning (39.7 vs 39.6) but wins descriptive by 5 points
(76.2 vs 71.1) and OCRBench by 66 points.** Since descriptive is the transcription-shaped half,
Qwen3-VL-4B is the right pick within its size class. Note also that the 2024-era chart specialists
(ChartGemma 12.5, TinyChart 8.3) score at or **below** the GPT-4o random-guess floor of 10.8 —
further confirmation that they should be ignored.

---

## 5. Candidate detail

### 5.1 Qwen3-VL 2B / 4B — the recommendation

- **License: Apache 2.0**, verified via HF API `cardData.license` for both
  [2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and
  [4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). Clean for any use, commercial or academic.
- **Params:** 2,127,532,032 and 4,437,815,808 (HF API `safetensors.total`).
- **Released:** 4B on 2025-10-15, 2B on 2025-10-21 (per the
  [README news section](https://github.com/QwenLM/Qwen3-VL#news)); technical report
  [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) on 2025-11-27.
- **Quantized builds: first-party and maintained.**
  [`Qwen/Qwen3-VL-2B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF)
  ships `Q4_K_M` (1.107 GB), `Q8_0` (1.834 GB), `F16` (3.447 GB) plus separate `mmproj`
  projectors (`F16` 819 MB, `Q8_0` 445 MB). There is also
  [`ggml-org/Qwen3-VL-2B-Instruct-GGUF`](https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF)
  from the llama.cpp org itself, which is the strongest possible signal that the architecture is
  properly supported upstream. Downloads are healthy (Qwen's own 2B GGUF repo: 87.9k/30d; the 4B:
  36.3k/30d; unsloth's 4B mirror: 225k/30d), so these are live artifacts, not abandoned uploads.
- **Native dynamic resolution.** Qwen3-VL uses a NaViT-style encoder with 16px patches and 2x2
  spatial merging, so one visual token ≈ a 32x32 px region and the image is not force-squashed to
  a fixed grid. That is exactly the property you want for small axis-label text — and exactly the
  property that makes it expensive on CPU.
- **Weakness:** the 4B-Instruct CharXiv RQ of 39.7 is the best in class and still less than half
  the Gemini 3 Flash number.

### 5.2 The OCR-specialist trap

There is a large, actively-developed family of small "document OCR" VLMs — PaddleOCR-VL (0.96B),
dots.ocr (3.04B), DeepSeek-OCR (3.34B), granite-docling-258M, olmOCR-2 (7B), MinerU2.5, GLM-OCR,
HunyuanOCR. They are tempting: tiny, fast, Apache/MIT, with first-party GGUF and llama.cpp
support ([PaddleOCR-VL PR #18825](https://github.com/ggml-org/llama.cpp/pull/18825),
[DeepSeek-OCR PR #17400](https://github.com/ggml-org/llama.cpp/pull/17400),
[dots.OCR PR #17575](https://github.com/ggml-org/llama.cpp/pull/17575), all referenced from
[`docs/multimodal.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)).

**They cannot do this job.** They are trained on a fixed, tiny set of literal prompt strings —
PaddleOCR-VL's are `OCR:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`,
`Spotting:`, `Seal Recognition:`; dots.ocr's are `prompt_layout_all_en`, `prompt_layout_only_en`,
`prompt_ocr`, `prompt_grounding_ocr`. Hand one of them `get_transcription_prompt()` and it will
either ignore it or degrade toward its training prompt. `docs/multimodal.md` says this outright:

> **IMPORTANT** — OCR models are trained with specific prompt and input structure, please refer to
> these discussions for more info

Two pieces of hard evidence that this is a real capability gap, not just a prompting nuisance:

- On **OCRBench v2**, which mixes recognition with *understanding* and key-information extraction,
  the dedicated parsers collapse to ~16-20 (EN) while Qwen3-VL-2B scores 56.3. The
  [Qianfan-OCR report](https://arxiv.org/html/2603.13398v1) explains why: *"these benchmarks
  include not only pure OCR recognition but also understanding and key information extraction
  (KIE) tasks, where specialized OCR models generally underperform."*
- The same report finds two-stage OCR+LLM pipelines score **zero on CharXiv**.

Beyond that, most of them explicitly refuse non-text imagery. dots.ocr names **"picture parsing"**
as a known weakness on its own card. PaddleOCR-VL crops `image` regions and passes them through
undescribed. olmOCR deliberately *suppresses* figure description — its
[prompt](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/prompts/prompts.py) asks only
for one-line markdown alt-text, and the paper frames document-anchoring as a way to stop the model
"captioning images when not instructed to do so".

The only partial exception is **DeepSeek-OCR** (MIT, 3.34B,
[arXiv:2510.18234](https://arxiv.org/abs/2510.18234)), whose paper documents "deep parsing" for
charts (chart→HTML table), chemical structures (→SMILES), planar geometry, and *"dense captions for
natural images"*. But the model card documents only two prompt entry points
(`<image>\nFree OCR.` and `<image>\n<|grounding|>Convert the document to markdown.`), so that
behaviour is not reachable through an arbitrary prompt. Do not budget on it without testing.

**Conclusion: the OCR specialists are the wrong family.** They may be worth a *second*,
complementary pass over the surrounding page text and tables, but they cannot produce the artifact
this map is about.

### 5.3 IBM granite-vision-3.3-2b — the credible alternative

- **License: Apache 2.0** (verified in the
  [model card](https://huggingface.co/ibm-granite/granite-vision-3.3-2b/raw/main/README.md)).
- **Params:** 2,975,396,928. Created 2025-06-03, last modified **2026-04-02** — actively maintained.
  349k downloads/30d.
- **Architecture:** LlavaNext — SigLIP2 + 2-layer MLP connector + `granite-3.1-2b-instruct`.
- **Reported scores (from the card):** ChartQA **0.87**, DocVQA 0.91, TextVQA 0.80, AI2D 0.77,
  OCRBench 0.79.
- **Genuinely instruction-following** — it is a real chat VLM, not a fixed-task parser.
- **Official first-party GGUF** via IBM's [github.com/IBM/gguf](https://github.com/IBM/gguf)
  conversion pipeline. There is also a
  [`granite-vision-3.3-2b-chart2csv-preview`](https://huggingface.co/ibm-granite/granite-vision-3.3-2b-chart2csv-preview)
  finetune (Apache 2.0, 2026-01-29) and a newer
  [`granite-vision-4.1-4b`](https://huggingface.co/ibm-granite/granite-vision-4.1-4b)
  (Apache 2.0, 2026-04-29) with explicit `<chart2csv>` / `<chart2summary>` / `<chart2code>` tags.
- **Why it is second, not first:** **no CharXiv score is published for any Granite Vision model.**
  ChartQA 0.87 sounds excellent, but ChartQA is a saturated benchmark on synthetic-ish charts;
  Qwen3-VL doesn't even bother reporting it. Choosing Granite over Qwen3-VL means choosing a model
  whose performance on realistic scientific figures is *unmeasured*. If you pick it, measure it
  first.

### 5.4 Gemma 4 E2B / E4B — worth testing, unmeasured on charts

Released 2026-03-02 under **Apache 2.0** (verified — Gemma 4 dropped the custom Gemma Terms of Use;
the [license link](https://ai.google.dev/gemma/docs/gemma_4_license) resolves to Apache 2.0, though
a separate [prohibited use policy](https://ai.google.dev/gemma/prohibited_use_policy) and intended
use statement are still referenced). Sizes are quoted as *effective* parameters: E2B is 2.3B
effective / 5.1B with embeddings, E4B is 4.5B effective / 8B with embeddings, with a ~150M vision
encoder. Per-Layer Embeddings mean the embedding tables are large but only used for lookups — good
for on-device, but note the **full weight file is sized by the total, not the effective, count**,
which matters for a CHTC `request_memory` line.

Vision scores from the [E4B card](https://huggingface.co/google/gemma-4-E4B-it):

| | E2B | E4B | Gemma 4 31B | *Gemini 3 Flash* |
| --- | --- | --- | --- | --- |
| MMMU Pro | 44.2% | 52.6% | 76.9% | *81.2%* |
| OmniDocBench 1.5 (edit dist, ↓) | 0.290 | 0.181 | 0.131 | *0.121* |
| MATH-Vision | 52.4% | 59.5% | 85.6% | — |
| MedXPertQA MM | 23.5% | 28.7% | 61.3% | — |

The OmniDocBench row is the single cleanest apples-to-apples data point in this whole document,
because Gemini 3 Flash reports the same metric on the same benchmark version: **0.121 for Gemini 3
Flash vs 0.181 for Gemma 4 E4B**. Same order of magnitude — document linearization is close to a
solved problem at 4B. Compare that to the CharXiv RQ gap (80.3 vs 39.7), which is a factor of two.
**The gap is not in reading the text. The gap is in understanding the figure.**

Gemma 4 is in llama.cpp's curated pre-quantized list (`ggml-org/gemma-4-E2B-it-GGUF`,
`ggml-org/gemma-4-E4B-it-GGUF`), so support is first-class. **No CharXiv or ChartQA number is
published**, so like Granite it would need measuring.

### 5.5 Rejected, with reasons

| Model | Why not |
| --- | --- |
| **Nanonets-OCR2-3B** | Genuinely instruction-following (it is a Qwen2.5-VL-3B SFT, and retains VQA), and it does emit inline `<img>` descriptions. But the weights carry the **Qwen Research License**, inherited from Qwen2.5-VL-3B — the one Qwen2.5-VL size that is not Apache 2.0. The HF repo has *no* `license` field; maintainers confirm the research license in [discussion #2](https://huggingface.co/nanonets/Nanonets-OCR2-3B/discussions/2). Probably acceptable for academic lab use, but it is a real restriction and it would block any downstream redistribution of a derived corpus under a permissive licence. Flagging, not excluding. |
| **Florence-2** (MIT, 0.23B/0.77B) | Fixed task tokens (`<CAPTION>`, `<OCR>`, `<DENSE_REGION_CAPTION>`, …) with no free-form instruction following; [HF's own finetuning post](https://huggingface.co/blog/finetune-florence2) confirms VQA-style prompts don't work without finetuning. No chart training objective, no ChartQA number, and **no llama.cpp support** (absent from `docs/multimodal.md`). ONNX exports are good, but for the wrong model. |
| **olmOCR-2** (Apache 2.0) | 7B only — no smaller checkpoint exists — and deliberately suppresses figure description. GPU-oriented; no official GGUF. |
| **MonkeyOCR**, **Marker** | Weight licences restrict commercial use (MonkeyOCR: "academic research and non-commercial evaluation only"; Marker models: modified AI Pubs Open RAIL-M with a $5M revenue cap). Probably fine for a university lab, but both are also fixed-task parsers, so the licence is moot. |
| **ChartGemma, TinyChart, ChartLlama, OneChart, ChartMoE** | The 2024-era chart-specialist cohort is stale. ChartGemma's HF repo has been untouched since 2024-07-27 and its `mit` tag is legally unreliable (it is a PaliGemma derivative). TinyChart has no licence on its card at all. ChartLlama is 13B and non-commercial. **None has a llama.cpp, ONNX or OpenVINO build.** granite-vision-3.3-2b strictly dominates the whole cohort. |
| **Qwen2.5-VL-3B** | Qwen Research License, not Apache 2.0 — and superseded by Qwen3-VL-2B/4B on every metric. |

---

## 6. CPU throughput and the corpus arithmetic

### 6.1 The arithmetic framework

What matters at 30-50M figures is **core-seconds per figure**, not wall-clock seconds per figure —
a model that takes 30s on 8 cores costs the same as one that takes 60s on 4.

Let `S` = core-seconds per figure (wall-clock seconds x cores in the slot), `N` = number of figures.

```
total core-hours = N x S / 3600
```

CHTC states that
[most users can obtain **>100,000 core-hours in a single day**](https://chtc.cs.wisc.edu/uw-research-computing/scaling-htc)
across CHTC, campus pools and the OSPool. Using that as the sustained-throughput budget:

```
days for a full pass = N x S / 3600 / 100,000
```

At N = 40M figures (midpoint of the 30-50M estimate):

| Core-seconds per figure | Total core-hours | Days at 100k core-hr/day | Verdict |
| --- | --- | --- | --- |
| 30 | 0.33M | **3.3 days** | Comfortable |
| 100 | 1.11M | **11 days** | Comfortable |
| 300 | 3.33M | **33 days** | Tolerable for a one-time pass; painful to repeat |
| 1,000 | 11.1M | **111 days** | Re-population becomes a ~4-month project |
| 3,000 | 33.3M | **333 days** | Disqualified by arithmetic |
| 10,000 | 111M | **~3 years** | Disqualified |

**So the budget is roughly S ≤ 300 core-seconds per figure**, and comfortably under if you expect
to re-run. Everything below is about whether any open VLM hits that on CPU.

### 6.2 Measured: Qwen3-VL-2B-Instruct Q4_K_M on llama.cpp, CPU only

I ran this rather than estimate it. **Read the caveat in
[§6.2.4](#624-why-you-should-not-use-these-absolute-numbers) before quoting any number here.**

#### 6.2.1 Setup

| | |
| --- | --- |
| Engine | `llama-mtmd-cli`, conda-forge `llama.cpp` build **b10158** (`cpu_mkl`), published 2026-07-28 — i.e. **after** [PR #25781](https://github.com/ggml-org/llama.cpp/pull/25781), so the Qwen3-VL position-embedding fix is included |
| Model | `Qwen/Qwen3-VL-2B-Instruct-GGUF` → `Qwen3VL-2B-Instruct-Q4_K_M.gguf` (1.03 GiB) |
| Projector | `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` (445 MB) — the Q8_0, per [§6.3](#63-published-cpu-numbers-from-other-people) |
| Prompt | verbatim `get_transcription_prompt("")` from `skimgpt/prompt_library.py:639` |
| Image | a **real PMC figure** — [PMC3148254](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0023061.g001) fig 1, PLOS ONE, CC BY, **1975x1861 px**. Panel A is a gene-targeting diagram with small labels; panel B is a **western blot with 10 rotated lane labels and two antibody rows**. Exactly the target workload. |
| Settings | `-t 8 -c 8192 -n 600 --temp 0 --image-max-tokens 1024` |
| Hardware | **Intel Xeon E7-4830 v3 @ 2.10 GHz** (Haswell-EX, 2015; AVX2, **no AVX-512, no AMX**), 6 GB RAM available |

#### 6.2.2 Timings

| Phase | Measured |
| --- | --- |
| Model + projector load | ~4 s |
| **Vision encode** (`mtmd batch encoding done in …`) | **57,123 ms** |
| LLM prefill + 600-token decode | ~219 s |
| **Total wall-clock** | **280.2 s** |
| **Core-seconds per figure (280.2 x 8)** | **~2,240** |

Supporting `llama-bench` numbers on the same box and model (text-only):

| Threads | Prefill (pp512) | Decode (tg) |
| --- | --- | --- |
| 4 | 60.9 tok/s | **2.36 tok/s** |
| 8 | 110.4 tok/s | **4.36 tok/s** |
| 16 | 168.9 tok/s | **7.52 tok/s** |

**Decode scales roughly linearly with threads here and is the dominant cost: ~78% of the wall-clock
was token generation, not vision encoding.** That is the opposite of what the llama.cpp issue
tracker leads you to expect, and it is a direct consequence of this workload's long outputs. The
vision encode was 57 s — real, but only a fifth of the total.

#### 6.2.3 Quality observations, which matter as much as the timings

The output was **partly excellent and partly disqualifying**:

- **OCR of small labels was genuinely good.** It correctly read `Rnf31`, `LoxP`, `NeoR`, `SV40pA`,
  `DT cassette`, `5' genomic flank`, `3' genomic flank`, `Exon: 1-21`, `Base pairs`, and the
  antibody rows `anti-HOIP` / `anti-FLAG`. At 1024 visual tokens on a 3.7 Mpx figure, small-text
  legibility was not the failure mode.
- **It correctly identified the two-panel structure** (A: diagram, B: blot).
- **It hallucinated the western blot lane labels.** The real figure has 10 lanes
  (`A20.2J`, `HOIP-/+`, `HOIP-/- clone 1`, `HOIP-/- clone 2`, `A20.2J pMIP`, `A20.2J pMIP-HOIP`,
  `clone 1 pMIP`, `clone 1 pMIP-HOIP`, `clone 2 pMIP`, `clone 2 pMIP-HOIP`). The model emitted
  those and then invented `clone 3, clone 4, … clone 21`.
- **It then entered a repetition loop**, re-emitting the panel A and panel B label blocks
  verbatim until it hit the 600-token cap. Final output was ~1,610 characters / ~280 words, much of
  it duplicated.
- **It never described band intensities** — instruction 5 of the prompt ("if it's a blot or
  microscopy image, describe the key observations") produced only *"The blot shows protein bands
  for HOIP and FLAG."*

This is a single sample at 2B and must not be over-generalised. But it is a concrete instance of
the failure modes [§4.3](#43-the-most-important-number-in-this-document-ocrbench-v2-element-parsing)
predicts from the OCRBench v2 Parsing collapse: **the model can read, and cannot reliably
enumerate or structure.** Counting lanes is exactly the "enumeration" primitive CharXiv-DQ tests,
and Qwen3-VL-2B scores 62.3 there. The 4B (76.2) would be better; how much better is untested.

**Practical implication for the corpus: a repetition-loop guard and an output-length cap are not
optional.** Without them a fraction of 30-50M figures will burn the full token budget emitting
duplicated text.

#### 6.2.4 Why you should not use these absolute numbers

**This CPU is a 2015 Haswell-EX with no AVX-512 and no AMX, on a host with 6 GB of RAM, and it is
an outlier on the slow side.** Two independent cross-checks:

- Its vision-encode rate is ~1,024 tokens / 57.1 s ≈ **18 image tokens/s**, versus the
  **100-160 tokens/s** cluster across every published CPU measurement in
  [§6.3](#63-published-cpu-numbers-from-other-people) — **6-9x slow**.
- Its decode rate (4.36 tok/s at 8 threads for a 1.7B model) versus the CEUR dataset's Xeon
  Platinum 8480+ at 16 vCPU (100 tok/s for Gemma-3-1B, 80 for Phi-4-mini 3.8B) — **roughly 10-20x
  slow**.

So treat 2,240 core-seconds as a **pessimistic bound**, not the expected value. Normalising by the
more conservative 6-9x factor gives an estimate of **~250-370 core-seconds per figure on a modern
Xeon**, at these exact settings (2B model, 1024 visual tokens, 600 output tokens).

**That lands squarely on the boundary of the budget in [§6.1](#61-the-arithmetic-framework).**

| Scenario | Core-s/figure | Core-hours for 40M | Days at 100k core-hr/day |
| --- | --- | --- | --- |
| Measured, this 2015 Xeon | 2,240 | 24.9M | **249 days — disqualified** |
| Normalised to a modern Xeon, same settings | ~300 | 3.3M | **~33 days** |
| Modern Xeon, output capped at 300 tokens | ~170 | 1.9M | **~19 days** |
| Modern Xeon, 4B model instead of 2B | ~600 | 6.7M | **~67 days** |

Three conclusions fall out of that table, and they are the operationally important part of this
whole document:

1. **A one-time pass with Qwen3-VL-2B on modern CPU nodes is feasible — roughly a month.** Not
   comfortable, but feasible.
2. **Output length is the dominant lever, ahead of image resolution.** Halving the token budget
   nearly halves the cost. The dense-prompt design ticket in the #120 map ("what does 'very dense'
   mean concretely — target token budget per figure") is therefore **not a quality question, it is
   the throughput question**, and it should be resolved before anything is built.
3. **Stepping up to the 4B — which is the model with the defensible quality numbers — roughly
   doubles the bill to ~2 months.** That is the real cost of the +13 CharXiv DQ points between 2B
   and 4B.

And the sting: **OSPool is heterogeneous.** If a meaningful fraction of matched slots are
2015-era hardware like the box I measured on, the effective average lands well above the
"modern Xeon" row. A production submit should almost certainly add a microarchitecture requirement
(e.g. require `avx512f`) and accept matching fewer slots in exchange for predictable throughput.

### 6.3 Published CPU numbers from other people

**A warning before the table.** The most widely-circulated "CPU vision is slow" datapoint —
[llama.cpp #22582](https://github.com/ggml-org/llama.cpp/issues/22582), "82 seconds per image
slice" — **is not a CPU measurement.** The reporter's box is a **Ryzen 9 7950X plus a Radeon
7900 XTX** running `--gpu-layers 999`, and the "vision encoder runs on CPU in BF16" root cause was
the reporter's own LLM-generated speculation. ngxson closed it the same day: *"closing since the
diagnostic is sloppy AI generated."* The 82 s wall-clock is real; the attribution is not. It should
not be used for CPU planning, and it appears in a lot of secondary write-ups that should be.

Genuine CPU-only measurements, with what each does and does not state:

| Model / quant | Hardware | Image tokens | Measured | Source |
| --- | --- | --- | --- | --- |
| LFM2-VL-1.6B Q8_0 + Q8_0 mmproj | Intel i5-12500H, CPU-only build | 69 | encode **666 ms**; prefill 74.3 tok/s; decode **25.7 tok/s** | [#17290](https://github.com/ggml-org/llama.cpp/issues/17290) |
| SmolVLM2-500M Q8_0 + Q8_0 mmproj | Windows CPU-only (`ggml-cpu-haswell`) | 64 | encode **1,215 ms** | [#13704](https://github.com/ggml-org/llama.cpp/issues/13704) |
| MiniCPM-V-2.6 Q2_K | Apple Silicon, `-ngl 0` | 64 | encode **6,653 ms** (vs 1,121 ms on Metal — 5.9x) | [PR #12322](https://github.com/ggml-org/llama.cpp/pull/12322) |
| Qwen3.6-27B, **Q8_0** mmproj on CPU | 16 threads (`ik_llama.cpp` fork) | **2,646** | **encode 25,233 ms**; image processed in 27,282 ms | [ik PR #1788](https://github.com/ikawrakow/ik_llama.cpp/pull/1788) |
| Gemma-4-26B-A4B, **F32** mmproj on CPU | x86_64 Linux, default threads | 256 (80x80 image, padded to Gemma's 896 grid) | **encode 72,765 ms**, then crashed on an FA assert | [ik #1961](https://github.com/ikawrakow/ik_llama.cpp/issues/1961) |
| Phi-4-MM via ONNX Runtime GenAI, `-e cpu` | CPU model/threads **not stated** | 2,619 (1024x1024) | **TTFT 46.95 s**, prefill 57.0 tok/s, **decode 3.61 tok/s**, peak RSS 8,192 MB | [AMD RyzenAI-SW VLM README](https://github.com/amd/RyzenAI-SW/blob/main/LLM-examples/VLM/README.md) |
| Phi-4-MM via ORT-GenAI, `-e cpu` | same | 5,765 (2048x2048) | **TTFT 137.52 s**, decode 2.29 tok/s | same |

Two conclusions:

1. **Across every log-backed CPU measurement, vision-encode throughput clusters around 100-160
   image tokens/second.** For a PMC figure at ~2,000 visual tokens that is **13-25 seconds of
   encode alone**, before a single LLM token.
2. **Projector quantization is worth more than almost anything else you can tune.** Compare the
   two `ik_llama.cpp` rows: a **Q8_0** projector encodes 2,646 tokens in 25 s; an **F32** projector
   takes 73 s for **256** tokens. Always ship the Q8_0 `mmproj`.

There is also a structural reason CPU vision is slow that is not documented anywhere in llama.cpp's
docs: **the vision tower never gets BLAS or AMX.** `clip_ctx::clip_ctx` in `tools/mtmd/clip.cpp`
enumerates only `GGML_BACKEND_DEVICE_TYPE_{CPU,GPU,IGPU}` and never
`GGML_BACKEND_DEVICE_TYPE_ACCEL`, so ggml-blas (MKL/OpenBLAS) and AMX are structurally unavailable
to it — while `src/llama-context.cpp` *does* have an explicit "add ACCEL backends (such as BLAS)"
block for the language model. The vision encoder is multi-threaded and does get llamafile tinyBLAS
SIMD kernels, but on a modern Xeon with AMX you will see the LLM half accelerate and the vision
half not. Maintainers have deliberately deprioritised fixing this — ngxson,
[#17801](https://github.com/ggml-org/llama.cpp/issues/17801): *"libmtmd [is] also designed to work
on all sorts of embedded devices... so I prefer keeping the code simple and as backend-agnostic as
possible."*

Image **preprocessing** is likewise scalar C, not SIMD
([discussion #24478](https://github.com/ggml-org/llama.cpp/discussions/24478) measures a 67x
speedup for bicubic resize from swapping in `stb_image_resize2`, and reports a real **40-second
preprocessing delay before any encode starts** on a 2020-era Intel server CPU with image-heavy
prompts). **Resize your figures upstream in Pillow, not inside llama.cpp.**

For the LLM half, the best-controlled CPU dataset is Malakhov, ProfIT AI'25
([CEUR Vol-4164 paper 11](https://ceur-ws.org/Vol-4164/paper11.pdf)) — llama.cpp Q4_K, no GPU,
threads pinned to physical cores. It explicitly states *"Images were not provided during testing
(all evaluations were text-only)"*, so use it only for decode/prefill:

| Model | Xeon E5-2695 v2 (16 vCPU, no AVX2) | **Xeon Platinum 8480+ (16 vCPU)** | i7-13700H |
| --- | --- | --- | --- |
| Gemma-3-1B | 30 tok/s | **100 tok/s** | 70 tok/s |
| Phi-4-mini 3.8B | 15 | **80** | 40 |
| Qwen2.5-7B | 8 | **45** | 15 |

**The spread between an old Xeon and a current one is 5-6x.** That matters enormously for a
heterogeneous HTCondor pool, and it is the main reason my own measurement below should not be taken
at face value.

### 6.4 The two knobs that actually move the number

1. **`--image-max-tokens` / image resolution.** Qwen3-VL emits roughly one visual token per 32x32
   px region. The PLOS figure used in §6.2 is 1975x1861 px = 3.68 Mpx, which at native resolution
   is **~3,590 visual tokens**. llama.cpp caps this; you can raise the cap with
   `--image-max-tokens`. There is a real quality floor here — llama.cpp warns that *"Qwen-VL models
   require at minimum 1024 image tokens to function correctly on grounding tasks"* and suggests
   `--image-min-tokens 1024`. Raising the cap improves small-text legibility and costs prefill time
   roughly linearly (and attention cost quadratically). **This is the single biggest lever on both
   axes and it must be tuned empirically, not guessed.**
2. **Output length.** Dense transcription is 600-1000 tokens. CPU decode is memory-bandwidth bound
   at batch 1, so this dominates. **Continuous batching is the mitigation**: llama.cpp's server
   supports `-np` parallel slots, and at batch 8-16 the model weights are read once per batch step
   instead of once per token, which materially improves *aggregate* core-second efficiency even
   though per-request latency worsens. This argues for **fat CPU slots running a batched
   `llama-server`**, not the current 1-core-per-job pattern.

   **But see [§7.1](#71-llamacpp--gguf--the-only-mature-option) — `llama-server` has an unresolved
   Qwen-VL OCR accuracy regression that `llama-mtmd-cli` does not.** So the batching win is not free;
   it has to be validated against CLI output on real figures first.

   That is in direct tension with
   [OSPool's guidance](https://portal.osg-htc.org/documentation/htc_workloads/workload_planning/preparing-to-scale-up/)
   to *"start by requesting a single cpu … you will be able to achieve greater throughput with
   single-cpu jobs"* — which is about *matching more slots*, not about per-core efficiency. There
   is a genuine optimisation to do here: many thin slots match faster but batch worse; few fat
   slots batch better but queue longer. **Resolving that trade-off needs a real measurement at two
   or three slot sizes, and it should be its own ticket.**

### 6.5 What a CHTC CPU slot would need

Replacing the GPU stanza in `htcondor_helper.py:135`:

| Attribute | Current (GPU) | Proposed (CPU), Qwen3-VL-4B Q4_K_M |
| --- | --- | --- |
| `request_gpus` | `1` | *(remove)* |
| `gpus_minimum_memory` | `10GB` | *(remove)* |
| `requirements` | `(CUDACapability >= 8.0)` | *(remove; optionally require AVX-512 / a modern microarchitecture)* |
| `+WantGPULab` | `true` | *(remove)* |
| `+GPUJobLength` | `"short"` | *(remove)* |
| `request_cpus` | `1` | **`8`** — decode scaled near-linearly to 16 threads in my measurement (2.36 → 4.36 → 7.52 tok/s at 4/8/16), so more cores genuinely convert to throughput here rather than being wasted. 8 balances that against slot-matching. |
| `request_memory` | `15GB` | **`6GB`** for the 2B (Q4_K_M 1.1 GB + Q8_0 mmproj 0.45 GB + 8k KV + vision activations; my run fit in a 6 GB host). **`8GB`** for the 4B. 15 GB is over-provisioned and matches fewer slots. |
| `request_disk` | `15GB` | **`6GB`** (model files + input figure + output) |
| *(new)* `requirements` | — | consider **`(Microarch >= "x86_64-v4")`** or an explicit AVX-512 check — see [§6.2.4](#624-why-you-should-not-use-these-absolute-numbers); matching fewer but faster slots is very likely the right trade at this scale |

Note the model files must reach the execute node. At 30-50M figures across many jobs, pulling
~4 GB of GGUF per job from Hugging Face is not viable; the weights should be baked into the docker
image (the submit already uses `universe: docker` with `docker_pull_policy: missing`, so this is a
natural fit) or staged via CHTC's OSDF/Pelican.

---

## 7. Serving stacks

The ticket warned that several stacks support text well and vision poorly. That warning is
correct, and the gap is wider than it looks from the marketing.

### 7.1 llama.cpp / GGUF — the only mature option

**Verdict: this is the stack.** It is the only CPU runtime with first-party support for the models
that matter, an OpenAI-compatible server, and continuous batching that works with images.

Vision goes through `libmtmd`. From
[`docs/multimodal.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md), the
curated pre-quantized vision list covers Gemma 3, **Gemma 4 (E2B, E4B, 26B-A4B, 31B)**, SmolVLM /
SmolVLM2, Pixtral 12B, Qwen2-VL, Qwen2.5-VL, Mistral Small 3.1, InternVL 2.5 / 3, Llama 4 Scout and
Moondream2, plus an OCR-model section (PaddleOCR-VL, GLM-OCR, DeepSeek-OCR, Dots.OCR, HunyuanOCR).
**Qwen3-VL is not in that curated list but is fully supported** — both Qwen and ggml-org publish
Qwen3-VL GGUF, and there are merged Qwen3-VL-specific fixes in the tree.

Relevant flags, from
[`tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):

| Flag | Meaning |
| --- | --- |
| `-mm, --mmproj FILE` | the vision projector |
| `--mmproj-offload / --no-mmproj-offload` | GPU offload for the projector, default **enabled** |
| `--image-min-tokens N` | min visual tokens per image, "default: read from model" |
| `--image-max-tokens N` | max visual tokens per image |
| `-np, --parallel N` | server slots (default -1 = auto) |
| `-cb, --cont-batching` | continuous batching, **default enabled** |

Images are accepted through the OpenAI-compatible `/chat/completions` endpoint, so the existing
call shape in `image_analyzer.py` maps over with little change.

Qwen3-VL support merged in [PR #16780](https://github.com/ggml-org/llama.cpp/pull/16780)
(2025-10-30), which added deepstack visual-feature handling in `clip.cpp` and the IMROPE vision rope
type — full multimodal, not a text-only stub. Two useful operational details for HTCondor:
`/v1/chat/completions` accepts a **local file path** with `--media-path` and a `file://` prefix, so
you can skip base64 entirely; and `--mtmd-batch-max-tokens` (default 1024) caps image tokens per
encode batch.

**Maturity caveats — these are real:**

- **`llama-server` has an unresolved Qwen-VL OCR accuracy regression that `llama-cli` does not.**
  [#22785](https://github.com/ggml-org/llama.cpp/issues/22785) — "Server: Qwen-VL vision accuracy
  degraded in fine-grained OCR tasks since b8545, while llama-cli remains unaffected" — traced to
  the `mtmd_image_preprocessor_dyn_size` refactor, with suspected patch-slicing / mRoPE 2D
  position-ID misalignment. **Closed as not planned / stale, i.e. still present.**
  `llama-mtmd-cli` bypasses the new preprocessor and is unaffected. This directly complicates the
  "batched `llama-server`" throughput strategy in [§6.4](#64-the-two-knobs-that-actually-move-the-number):
  **you must A/B the server against the CLI on real figures before trusting it**, and be prepared
  to fall back to one-CLI-invocation-per-figure at a throughput cost.
- **The docs lag the code.** Both Qwen3-VL and Gemma 4 are absent from the `-hf` example list in
  `docs/multimodal.md` and from the conversion list in `tools/mtmd/README.md` despite being fully
  supported. That README also states the subsystem is *"under very heavy development, and breaking
  changes are expected."* Pin a commit.
- **There is no vision support in `llama-bench`.** `tools/llama-bench/llama-bench.cpp` contains no
  occurrence of `mmproj`, `mtmd`, `clip`, `image` or `vision`. There is no official benchmarking
  tool for the vision path and consequently no maintainer-published CPU vision performance table
  anywhere. You will be writing your own harness.
- **Qwen3-VL vision had a numerical fidelity bug until 2026-07-21.**
  [PR #25781](https://github.com/ggml-org/llama.cpp/pull/25781) ("mtmd: use align_corners for
  Qwen3-VL vision position-embedding interpolation") fixed an interpolation mismatch where
  llama.cpp used `align_corners=False` while the transformers reference uses `align_corners=True`.
  The error **scaled outward from the image centre, grew with image dimensions, and hit non-square
  images hardest** — which describes PMC figures exactly. It was merged one week before this note
  was written. **Any benchmark of Qwen3-VL under llama.cpp from before 2026-07-21 is suspect, and
  any build you deploy must be newer than that commit.** See also
  [#17594](https://github.com/ggml-org/llama.cpp/pull/17594) (clip nb calculation for Qwen3-VL,
  merged 2025-11-30).
- **Image preprocessing has diverged from the reference before.**
  [#16842](https://github.com/ggml-org/llama.cpp/issues/16842) was about implementing the correct
  `max_pixels` / `min_pixels` from Qwen's own preprocessor config (closed via PR #16878).
  [#17345](https://github.com/ggml-org/llama.cpp/issues/17345) reports
  `vl_high_resolution_images` not taking effect for Qwen3-VL-4B (closed stale, unconfirmed). The
  tree emits a warning that *"Qwen-VL models require at minimum 1024 image tokens to function
  correctly on grounding tasks"*.

**Practical consequence: pin a llama.cpp commit ≥ PR #25781, and validate output quality against
`transformers` on a sample of real figures before trusting the stack.** Do not treat GGUF as
lossless with respect to the published benchmark scores — those scores were measured with
`transformers`/vLLM, not llama.cpp.

### 7.2 OpenVINO GenAI — the serious second candidate, evaluate it in parallel

I initially expected to dismiss this. The evidence does not support dismissing it.

`openvino_genai.VLMPipeline` (OpenVINO 2026.2.1, 2026-06-17) supports, per the
[authoritative model table](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/):
InternVL2/2.5/3 (1B-14B), LLaVA-1.5/NeXT, nanoLLaVA, MiniCPM-V-2.6 / -o-2.6, Phi-3/3.5-vision,
Phi-4-multimodal, **Qwen2-VL / Qwen2.5-VL / Qwen3-VL (2B/4B/8B/32B)**, **Gemma-3 / 3n / 4**.
`optimum-intel` supports a superset via `OVModelForVisualCausalLM` (adds SmolVLM, Idefics3,
GOT-OCR2) but those are Python-only, not servable through the C++ pipeline or OVMS.
Not supported anywhere in the stack: **granite-vision**, Florence-2, Pixtral.

**The one clean CPU A/B in the public record favours it heavily.** From HF's own
[OpenVINO VLM blog](https://huggingface.co/blog/openvino-vlm) — SmolVLM2-256M-Video-Instruct, single
image, Intel Core Ultra 7 265K, 20 cores, 64 GB DDR5:

| Config | TTFT | E2E latency | Decode throughput |
| --- | --- | --- | --- |
| PyTorch fp32 | 5.150 s | 25.93 s | **0.72 tok/s** |
| OpenVINO fp32 | 0.420 s | 0.738 s | **47.24 tok/s** |
| OpenVINO INT8 WOQ | **0.247 s** | **0.482 s** | **63.93 tok/s** |

Treat the *direction* as solid and the *magnitude* as an upper bound: the PyTorch row is
implausibly bad (1.385 s/token for a 256M model on 20 cores suggests a threading pathology, not an
honest torch ceiling), it is a 256M model that will not extrapolate to 4B, and image resolution and
output token count are not stated.

**A quantization detail that matters for OCR fidelity, and is not on the model cards:** with
`optimum-cli export openvino --weight-format int4`, **INT4 is applied to the language model only**.
The vision encoder, vision merger and text embeddings fall back to INT8-symmetric. This is
hard-coded in
[`optimum/intel/openvino/quantization.py`](https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/quantization.py).
That default is exactly what you want here — **do not INT4 the vision tower.** Also: read
`openvino_config.json`, not the model card prose; for `OpenVINO/Qwen3-VL-8B-Instruct-int4-ov` the
card says "INT4_SYM" while the config says `sym: false`.

**Two concrete reasons to run this evaluation rather than skip it:**

1. `benchmark_vlm.py -d CPU` reports **"Embeddings preparation time" separately** — i.e. it gives
   you the vision-encoder cost as its own number, which is precisely the quantity nobody has
   published and which [§6.3](#63-published-cpu-numbers-from-other-people) shows dominates.
   llama.cpp has no equivalent tool at all.
2. It has two throughput levers llama.cpp lacks: **CDPruner visual-token pruning**
   (`--pruning_ratio`) and **prompt-lookup decoding extended to VLMs** in 2026.2. Prompt-lookup
   should help *a lot* on this specific workload, because dense transcription output largely copies
   text that is already in the image.

**Serving:** OpenVINO Model Server v2026.2.1 serves VLMs over an OpenAI-compatible chat API with
continuous batching, tested on 4th-6th gen Xeon Scalable. Note `--allowed_media_domains` blocks URL
image inputs by default; base64 or local paths sidestep it.

**Caveats:** several of the shiny document-OCR notebooks (`paddleocr_vl`, `deepseek-ocr`,
`hunyuan-ocr`) ship custom `ov_*_helper.py` conversion code and are notebook-grade, not
`VLMPipeline`-servable. There are no pre-converted IRs for sub-8B Qwen3-VL, so you would convert
yourself. And **Intel publishes no measured VLM benchmark data for Xeon CPUs at all** — the
official generative-AI performance tables state verbatim that they cover *"built-in GPUs"*, and the
underlying data files contain zero VLM entries.

### 7.2b OpenVINO vs llama.cpp — how to decide

Neither has a published CPU number for a ≥2B VLM on server-class x86. **That number does not exist
in public.** So this cannot be decided from the literature; it needs a bake-off. Recommendation:
run both on ~50 real PMC figures, measure encode and decode separately, and compare output quality
against a `transformers` reference. Budget a day. Given that [§6.1](#61-the-arithmetic-framework)
shows the whole go/no-go turns on core-seconds per figure, this is the highest-value experiment in
the entire map.

### 7.3 ONNX Runtime / Optimum / transformers.js — rule out, and the reason is structural

This is worse than "immature". Three independent blockers:

1. **Optimum cannot export any VLM to ONNX.** ONNX export moved to
   [huggingface/optimum-onnx](https://github.com/huggingface/optimum-onnx) (v0.1.0, 2025-12-23).
   Its [supported-architectures list](https://huggingface.co/docs/optimum-onnx/en/onnx/overview)
   contains **no** Qwen-VL, SmolVLM, Idefics3, LLaVA, PaliGemma, Florence-2, InternVL, Phi-3-vision,
   Gemma3-vision or granite-vision. There is no `ORTModelForImageTextToText`. In
   `model_configs.py` the string `"image-text-to-text"` appears exactly once — in a comment saying
   it is not supported. Feature requests for Qwen2.5-VL
   ([optimum#2376](https://github.com/huggingface/optimum/issues/2376)) and SmolVLM2
   ([optimum#2431](https://github.com/huggingface/optimum/issues/2431)) were closed **not planned**.
2. **ONNX Runtime GenAI never builds a vision tower.** In
   [`builder.py`](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py),
   *every* VLM branch — Qwen2_5_VL, Qwen3VL, Phi3V, Phi4MM, Gemma3, Mistral3 — prints
   *"WARNING: This is only generating the text component of the model."* Its README still lists
   multi-modal under *"On the roadmap"*. Qwen3-VL is not natively registered and only runs by
   masquerading as `qwen2_5_vl`, with a known mismatch on patch size (16x16 vs 14x14) —
   [issue #1989](https://github.com/microsoft/onnxruntime-genai/issues/1989), open since 2026-02-24
   with no maintainer reply.
3. **The community exports have real, unfixed correctness bugs.** Two concrete traps worth knowing
   even if you never use ONNX:
   - **Quantization is sometimes a silent no-op.** In `onnx-community/Qwen2-VL-2B-Instruct`,
     `vision_encoder.onnx_data` and `vision_encoder_q4.onnx_data` are **byte-identical**
     (2,661,085,184 bytes each). Same pattern for `embed_tokens` vs `_q4` across Florence-2,
     SmolVLM-256M and granite-docling. Diff the byte sizes; do not trust the filenames.
   - [granite-docling-258M-ONNX discussion #5](https://huggingface.co/onnx-community/granite-docling-258M-ONNX/discussions/5)
     (open, unanswered): the ONNX build emits *"3D-CAM camera with a lens and a tripod…"* where
     PyTorch emits *"A lake with trees in the background…"*. **A document-transcription model
     hallucinating in ONNX, with nobody debugging it.**

`transformers.js` has the broadest VLM list of any ONNX stack, but on a headless Linux slot it
falls back to onnxruntime-**web** WASM, which is strictly slower than native ORT. Wrong tool.

### 7.4 transformers on CPU — the fidelity reference, not the production path

`transformers` with `device_map="cpu"` is what the published benchmark numbers were measured with,
so it is the right thing to validate GGUF or OpenVINO output *against*. As a production stack at
30-50M figures it is not competitive — and note that **HF has effectively withdrawn its own CPU
inference guidance**: the `perf_infer_cpu` page is now one screen whose entire substance is a
pointer to ONNX Runtime and Optimum-Intel/OpenVINO, with no dtype guidance, no thread guidance and
no VLM mention.

Three practical traps if you use it as a reference:

- **Pass `device_map="cpu"` explicitly, never `"auto"`.** For multimodal models `_no_split_modules`
  must be defined or auto-sharding misbehaves
  ([transformers#29786](https://github.com/huggingface/transformers/issues/29786)), with related
  corruption bugs when vision towers are placed separately
  ([#35918](https://github.com/huggingface/transformers/issues/35918),
  [#38408](https://github.com/huggingface/transformers/issues/38408)).
- **Avoid fp16 on CPU entirely.** It is nominally supported since PyTorch 2.5, but without AMX-FP16
  it is *catastrophically* slow, not merely unaccelerated:
  [pytorch#146508](https://github.com/pytorch/pytorch/issues/146508) (still open) measures fp16
  matmul **137x slower** than fp32 on an AVX-512 CPU without AMX. For a heterogeneous HTCondor pool
  the safe default is **fp32**, opting into bf16 only after checking `/proc/cpuinfo` for
  `avx512_bf16` / `amx_bf16`.
- **SDPA has no fused CPU kernel** — the HF attention docs state it *"defaults to the PyTorch C++
  implementation for other backends"*. FlashAttention-2/3 are GPU-only.

Also inherited: [QwenLM/Qwen3-VL#1678](https://github.com/QwenLM/Qwen3-VL/issues/1678), a
single-threaded image resize in `qwen-vl-utils` that bottlenecks throughput.

### 7.5 Others — including two that are outright dead

| Stack | Status | Verdict |
| --- | --- | --- |
| **IPEX** (`intel-extension-for-pytorch`) | **ARCHIVED** 2026-03-30. README: *"THIS PROJECT IS ARCHIVED… This project has been identified as having known security issues."* Last CPU release v2.8.0+cpu (2025-08-12), five torch minors behind. Its CPU LLM table listed exactly three VLMs, none of them Qwen-VL. | **Do not use.** |
| **ipex-llm** | **ARCHIVED**, last push 2026-01-28. Intel-GPU/NPU-focused anyway. | **Do not use.** |
| **Ollama** | v0.32.5, very active; ships `qwen3-vl` 2b/4b/8b/30b/32b, `gemma4`, `glm-ocr`. | Works, but its API has **no vision/image-encode timing field at all** — encode is silently folded into prompt-eval. You cannot instrument the thing that dominates your cost. Prefer raw llama.cpp. |
| **llamafile** | **Not dead** — moved to [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile), v0.10.4 (2026-07-16). 0.10.x is a rebuild with llama.cpp as a submodule, so it inherits current libmtmd. | Genuinely attractive for HTCondor staging (single portable binary), but vision usage is undocumented. Worth a look for packaging only. |
| **vLLM** | The existing CHTC fallback; `relevance_chtc.py` raises `RuntimeError` before constructing `vllm.LLM` if CUDA is unavailable. A CPU backend exists but is an explicitly experimental/dev target. | GPU path only. |
| **MNN** | Very active; Qwen3-VL supported, x86 AVX2/AVX-512 rated "S". But the headline 8.6x speedup is **mobile ARM, text-only**, and there are no published x86 VLM benchmarks. | Unproven; not worth the integration risk. |
| **ExecuTorch** | Multimodal exists but is Android/iOS-oriented and requires AOT export to `.pte` — worst case for variable-resolution figures with per-image token counts. | Rule out. |
| **PowerInfer** | Its whole thesis is GPU-CPU hybrid (hot neurons on GPU). On a GPU-less slot the premise evaporates. No VLM support found. | Rule out. |
| **T-MAC**, **ratchet**, **nexa-sdk** | Dormant / WebGPU / Qualcomm-hardware respectively. | Rule out. |
| mistral.rs, candle | Active, have vision + CPU. CPU VLM performance **unverified**. | Not investigated further. |

---

## 8. The GPU counterfactual

If the CPU numbers do not clear the bar, here is what GPU changes.

CHTC's [GPU documentation](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs) lists roughly
**78 GPUs** in the HTC system, of which the **GPU Lab (~62 GPUs) is available to any CHTC user** —
a mix of A100-SXM4 40/80GB, L40, L40S, H100 and H200. Critically for this workload:

- **"Short" jobs may occupy up to 2/3 of the GPU Lab GPUs** — call it ~40 GPUs.
- **"Long" jobs are limited to 4 GPUs per user.**

The existing submit already sets `+GPUJobLength: "short"`, so the ~40-GPU ceiling is the operative
one, shared with every other CHTC user.

vLLM with continuous batching on an L40S or A100 running Qwen3-VL-4B should manage on the order of
**1-4 figures/second/GPU** at ~800 output tokens each — *I could not find a published measurement
of this exact configuration, so treat that range as an estimate, not a citation.* At 40M figures:

| Figures/s/GPU | GPUs | Wall-clock for 40M |
| --- | --- | --- |
| 1 | 40 | ~11.6 days |
| 2 | 40 | ~5.8 days |
| 2 | 10 | ~23 days |
| 4 | 40 | ~2.9 days |

**The GPU path also unlocks the bigger model.** Qwen3-VL-8B-Instruct (CharXiv RQ 46.4) or
32B-Instruct (62.8) are out of reach on CPU but routine on an 80GB A100. The 32B number — 62.8 RQ
— is the first open-weight score that is genuinely in the same conversation as Gemini 3 Flash's
80.3, and it beats Claude 4 Sonnet's 60.9 on that benchmark.

**So the real trade is not "CPU vs GPU cost". It is "CPU gets you CharXiv RQ ~40-50; GPU gets you
~63".** That is the sentence the map should decide on.

---

## 9. Recommendation

**Top candidate: `Qwen3-VL-4B-Instruct` (Apache 2.0), served via llama.cpp `libmtmd` with the
first-party `Q4_K_M` GGUF and the `F16` mmproj, in a batched `llama-server` on a multi-core CHTC
CPU slot.**

Reasons, in order:

1. It is the only sub-5B open model with a **published score on a realistic scientific-figure
   benchmark**, and it is the best in its class (CharXiv RQ 39.7 Instruct / 50.3 Thinking).
2. Apache 2.0 with no research-use rider — the derived corpus is unencumbered by the model licence.
3. **First-party GGUF from Qwen *and* a build from ggml-org itself.** No other candidate has both.
4. Native dynamic resolution, which is the property that keeps small axis-label text legible.
5. It is a real instruction-following VLM, so `get_transcription_prompt()` works unmodified.

**On the serving stack, hedge.** llama.cpp is the default, but
[§7.2](#72-openvino-genai--the-serious-second-candidate-evaluate-it-in-parallel) makes a real case
for **OpenVINO GenAI `VLMPipeline`** — it supports Qwen3-VL, its `benchmark_vlm.py` reports the
vision-encoder cost as its own number (llama.cpp gives you no such tool), and it has two levers
llama.cpp lacks (CDPruner visual-token pruning, and VLM prompt-lookup decoding, which should be
unusually effective on a task whose output largely copies text visible in the image). **Neither
stack has a published CPU number for a ≥2B VLM on server-class x86. Bake them off.**

**Runner-up models to benchmark against it, not to assume are worse:** `granite-vision-3.3-2b`
(Apache 2.0, official IBM GGUF, ChartQA 0.87 — but *not supported by OpenVINO*) and
`gemma-4-E4B-it` (Apache 2.0, OmniDocBench 1.5 0.181, supported by both stacks). Both are
unmeasured on CharXiv. Both are plausible. Also worth one afternoon:
**[`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR)** — **MIT**, **1.33B params** (HF API
`safetensors.total` = 1,325,258,240; note some write-ups say 0.9B), released 2026-01-30, **3.56M
downloads/30d**, CogViT encoder + GLM decoder, purpose-built for figures/tables/formulas, in
llama.cpp's OCR list and in Ollama. At 1.3B it is the model in this survey where CPU throughput is
least likely to be the blocker — the open question is purely whether it will follow a free-form
prompt or behave like the fixed-task parsers in [§5.2](#52-the-ocr-specialist-trap). That is
unpublished, and cheap to test.

**Four caveats, stated plainly:**

- **On reasoning, the gap is real and large.** 39.7 vs 80.3 CharXiv RQ is not "approaching" Gemini 3
  Flash. Gemini 3 Flash sits at the human baseline for this task; Qwen3-VL-4B-Instruct sits at
  half of it. Anything the corpus stores will carry that ceiling permanently — and re-transcribing
  30-50M figures later is a months-long job, not an afternoon. **The load-bearing assumption of
  the #120 map (that a dense unaware transcription is a superset of what the targeted prompt
  surfaces) is only testable against a model that can actually see the figure.** At CharXiv RQ 40,
  the superset property may fail for reasons that have nothing to do with hypothesis-awareness.
- **On transcription, the gap is much smaller — because everyone is bad.** OCRBench v2 Element
  Parsing puts Gemini 3 Pro Preview at 27.1 and GPT-5 at 24.8, against ~13-22 for open 7-8B
  models. If the deliverable is genuinely *dense serialisation* rather than *figure Q&A*, the
  open-model penalty is a handful of points off an already-low base, not a halving. **This cuts
  both ways: it weakens the "only Gemini is good enough" objection, and it weakens the premise
  that any model can produce the artifact the map wants.** Whoever writes the verdict should
  decide which of those two readings governs.
- **There is no published evidence for the biomedical half of the prompt.** Blots, gels,
  micrographs and multi-panel decomposition are unevaluated for every model here, open or closed.
  On the nearest proxy — MMSci caption generation over Nature Communications figures — **GPT-4o
  scores BLEU-2 ≈ 4.93** while a 7B model *fine-tuned for the task* reaches 19.13. That 4x gap from
  fine-tuning is worth noting: if this corpus matters enough, **fine-tuning a small open model on
  PMC figure-caption pairs may beat prompting a big one**, and Qwen3-VL-4B is a much better
  fine-tuning target than a proprietary API.
- **Thinking mode is a trap on CPU.** It buys ~11 CharXiv RQ points and ~8 DQ points, and costs
  several hundred extra generated tokens per figure — the single most expensive thing you can do on
  a CPU. If the measured budget is tight, Instruct mode is forced, and the quality numbers to plan
  around are DQ 76.2 / RQ 39.7, not 83.9 / 50.3.

### 9.1 The verdict on CPU, stated as plainly as the evidence allows

**CPU is viable, at roughly a month per full pass, with the 2B model on modern hardware.** It is
not comfortable and it is not obviously the right choice.

The measured arithmetic ([§6.2.4](#624-why-you-should-not-use-these-absolute-numbers)):

| Path | Model | Quality (CharXiv DQ / RQ) | Full pass over 40M figures |
| --- | --- | --- | --- |
| CPU, modern Xeon, 300-token cap | Qwen3-VL-2B | 62.3 / 26.8 | ~19 days |
| CPU, modern Xeon, 600-token cap | Qwen3-VL-2B | 62.3 / 26.8 | ~33 days |
| CPU, modern Xeon, 600-token cap | Qwen3-VL-4B | 76.2 / 39.7 | ~67 days |
| **GPU, ~40 CHTC "short" slots** | **Qwen3-VL-8B** | **83.0 / 46.4** | **~1-2 weeks** *(estimated, see §8)* |
| **GPU, ~40 CHTC "short" slots** | **Qwen3-VL-32B** | **90.5 / 62.8** | **weeks** *(estimated)* |

Read the top and bottom rows together. **The CPU path costs ~2 months to produce a corpus at
CharXiv DQ 76.2; the GPU path plausibly costs ~2 weeks to produce one at DQ 90.5.** Since the
corpus is a durable artifact that gets re-derived only at great expense, and since CHTC's GPU Lab
is real capacity that this project is entitled to use, **the CPU preference in the #120 map looks
like the wrong optimisation.** It was stated as a preference, not a requirement. This is the
evidence on which to revisit it.

The honest counter-argument: GPU slots are scarce and contended, a "short" job is capped at 2/3 of
the GPU Lab, and my figures/second/GPU estimate is unverified. A CPU pass is *schedulable* in a way
a 40-GPU campaign may not be. If the lab cannot realistically hold 40 GPUs for two weeks, the
~33-day CPU pass with the 2B is a genuine fallback — just accept CharXiv DQ 62.3 and plan the
re-population budget accordingly.

### 9.2 What I would do next, in order

1. **Build the eval before building the pipeline.** ~100 real PMC figures spanning charts, blots,
   micrographs and multi-panel composites, scored with CharXiv-descriptive-style templated
   primitives. Nothing in this document substitutes for it, and every downstream decision depends
   on it.
2. **Bake off llama.cpp vs OpenVINO GenAI** on the same 100 figures, measuring encode and decode
   separately. Settle the stack question with data, since the literature cannot.
3. **Settle the output-token budget**, because it is the dominant throughput lever
   ([§6.2.4](#624-why-you-should-not-use-these-absolute-numbers)).
4. **Then** decide CPU vs GPU, with §9.1's table populated by real numbers instead of my
   normalisations.

---

## 10. Confidence and gaps

**Measured by me on this repo's hardware, reproducible:** everything in
[§6.2](#62-measured-qwen3-vl-2b-instruct-q4_k_m-on-llamacpp-cpu-only) — the 57.1 s vision encode,
the 280.2 s end-to-end, the `llama-bench` thread-scaling table, and the quality observations. The
benchmark script and the exact model/projector/build versions are recorded there. **The absolute
values are from a 2015 Xeon and are 6-20x pessimistic; the 6-20x normalisation itself is my
inference from two independent published cross-checks, not a measurement.**

**High confidence (verified against primary sources, URLs inline):**

- All Qwen3-VL benchmark numbers — read directly off the official table images published in the
  QwenLM/Qwen3-VL README. Note that these tables are *images*, not text; a naive text fetch of that
  README returns nothing, and at least one automated summariser hallucinated a plausible-looking
  but entirely fabricated set of numbers when asked. Anyone re-checking this should read the JPEGs.
- Gemini 3 Flash numbers — read from the benchmark table image embedded in the official DeepMind
  model card PDF.
- All licences and parameter counts — HF API `cardData.license` and `safetensors.total`.
- CHTC capacity figures and GPU inventory — CHTC's own documentation.
- The repo-side facts in §2 — read from `origin/fulltext`.

**Could not verify:**

- **Gemini 3 Flash's `media_resolution_high` token cost.** I did not find Google documentation
  stating how many tokens `media_resolution_high` consumes per image for Gemini 3, so I cannot
  compare "effective input resolution" between the current pipeline and a Qwen3-VL configuration
  on an apples-to-apples basis.
- **vLLM figures/second/GPU for Qwen3-VL-4B.** The 1-4 figures/s/GPU range in §8 is my estimate
  from model size and output length, not a citation. It should be measured before anyone commits
  to a GPU plan; it is the number the whole GPU case rests on.
- **OCRBench v2 scores for PaddleOCR-VL / dots.ocr / DeepSeek-OCR** (~16-20 EN). These come from a
  third-party technical report table that I could not open directly. They are consistent with the
  Qianfan-OCR paper's stated explanation, and the qualitative conclusion (OCR specialists collapse
  on understanding tasks) is independently supported by that paper's CharXiv finding, but treat the
  specific integers as indicative.
- **Independent verification of any Qwen3-VL number.** Every CharXiv and OCRBench figure for
  Qwen3-VL is self-reported by Qwen. The CharXiv official leaderboard is frozen at 34 models from
  mid-2024 and contains nothing released since; the README changelog claims newer models were added
  but they were not. Where a self-report and an official leaderboard both exist for the *same*
  model, they have disagreed by ~15 points (Qwen2.5-VL-7B on OCRBench v2: 56.3 self-reported vs
  41.8 measured). **Discount the open-model numbers accordingly — and note that Gemini's are
  self-reported too.**
- **SPIQA per-model open-weight scores.** Not retrieved from a primary source; excluded rather than
  guessed.
- **Chart-to-code / chart-to-table generative benchmarks** (ChartMimic, Plot2Code, ChartX,
  ChartCoder). These are genuinely generative and arguably closer in spirit to dense transcription
  than any QA benchmark, but I did not verify small-model scores on them from primary sources.
  **If someone wants one more evidence axis before the verdict, this is the gap I would fill next.**
- **CharXiv scores for granite-vision-3.3-2b, granite-vision-4.1-4b, gemma-4-E2B/E4B.** Not
  published by anyone, as far as I can find. This is why they are runners-up rather than
  co-recommendations.
- **Any evaluation of any model on western blots, microscopy, gels, or multi-panel biomedical
  composites.** I could not find one. I believe this genuinely does not exist rather than that I
  failed to find it, but I cannot prove a negative.
- **Realistic sustained CHTC throughput for *this* group.** The 100,000 core-hours/day figure is
  CHTC's own published statement about what "most users" can get; it is not a guarantee and not
  specific to this lab's allocation or priority. Every day-count in §6.1 scales inversely with the
  real number.
