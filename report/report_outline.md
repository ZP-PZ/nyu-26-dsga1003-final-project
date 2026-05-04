# Report Content Outline

Working title: **Prompt-Conditioned Residual Re-aggregation for Lightweight Domain Adaptation**

Target format: maximum 8-page NeurIPS workshop style, excluding references and optional appendix.

## Core Message

The main result is not simply that prompt conditioning helps. The strongest story is:

> On a frozen Qwen3-0.6B backbone, residual-stream re-aggregation is a more effective lightweight adaptation mechanism than layerwise write-strength scaling, and prompt conditioning gives a small but consistent additional gain. The best model, prompt-conditioned residual-stream re-aggregation, reduces test perplexity from 16.81 to 15.23, a 9.41% relative reduction over the frozen base model, while using only a small learned controller on top of the frozen LM.

Secondary interpretation:

> MLP inspection suggests the prompt-conditioned re-aggregation controller is less collapsed and more domain-sensitive than the prompt-conditioned write-strength controller.

## Recommended Page Budget

- Abstract: 0.25 page
- Introduction: 0.75 page
- Related Work: 0.75 page
- Method: 1.75 pages
- Experimental Setup: 1.0 page
- Results: 1.5 pages
- Analysis: 1.0 page
- Limitations and Conclusion: 0.75 page

Keep figures/tables compact. Aim for one main results table, one small analysis table, and optionally one method diagram.

## 1. Abstract

Purpose: state the question, method family, setup, and best empirical result.

Must include:

- Frozen Qwen3-0.6B backbone.
- Mixed general/specialized continuation task: WikiText + medical abstracts.
- Compared static vs prompt-conditioned variants for two residual intervention families.
- Best result: prompt-conditioned residual-stream re-aggregation reaches test PPL 15.23 vs frozen base 16.81.
- One-sentence analysis: prompt-conditioned re-aggregation shows lower output collapse and stronger domain separation than write-strength conditioning.

Avoid:

- Overclaiming as a general PEFT result.
- Saying prompt conditioning alone is the dominant factor.

## 2. Introduction

Purpose: motivate prompt-conditioned lightweight adaptation.

Suggested flow:

1. Many PEFT methods learn static adapter parameters used identically for every prompt.
2. In mixed-domain settings, the useful adaptation may depend on the prompt.
3. This project asks whether a small prompt-conditioned controller can improve residual interventions while keeping the base LM frozen.
4. We test two intervention families:
   - layerwise write-strength scaling,
   - residual-stream re-aggregation.
5. Preview result: re-aggregation matters most; prompt conditioning adds consistent gains.

Contributions:

- Implement a controlled comparison of static and prompt-conditioned residual interventions on a frozen Qwen3-0.6B model.
- Introduce a prompt-conditioned residual-stream re-aggregation variant that predicts lower-triangular write weights.
- Evaluate overall and per-domain continuation loss/perplexity.
- Inspect controller outputs for collapse and domain sensitivity.

## 3. Related Work

Purpose: position the project without spending too much space.

Include short paragraphs on:

- Parameter-efficient fine-tuning: LoRA, BitFit, adapters, related lightweight tuning.
- Residual weighting / residual-stream intervention ideas, including the Kimi-inspired residual weighting motivation.
- Conditional or input-dependent adaptation: hypernetworks, prompt-conditioned adapters, routing/mixture-style adaptation.
- Domain adaptation for language models.

Tone:

- Emphasize that this project is an empirical course-scale study, not a full replacement for standard PEFT baselines.

## 4. Method

Purpose: define exactly what was trained and what stays frozen.

### 4.1 Problem Formulation

Define each example as fixed-length continuation:

- prompt: first 96 tokens.
- answer: next 96 tokens.
- loss: next-token cross-entropy only on answer tokens.
- base model: frozen Qwen3-0.6B.

State that trainable parameters belong only to small residual controllers.

### 4.2 Prompt Representation

For prompt-conditioned variants:

- Run the frozen base model on the prompt.
- Concatenate:
  - final prompt-token hidden state,
  - mean-pooled final hidden states across prompt tokens.
- Feed this representation to a small MLP with SiLU activations.

Mention identity-centered parameterization:

- write-strength scales: `scale = 1 + 0.1 * raw_output`.
- re-aggregation weights: `weight = 1 + 0.1 * raw_output`.

### 4.3 Write-Strength Scaling

Static variant:

- Learn one residual multiplier per decoder layer.
- Apply to each layer write: `x_l = x_{l-1} + a_l * write_l`.

Prompt-conditioned variant:

- MLP predicts one multiplier per layer from prompt representation.

Main role in paper:

- This is the simpler baseline intervention family.

### 4.4 Residual-Stream Re-aggregation

Static variant:

- For target layer `l`, collect all previous writes up to `l`.
- Rebuild the residual stream as embedding stream plus learned weighted sum of writes.
- Learn lower-triangular weights over `(target layer, source write)` pairs.
- For Qwen3-0.6B with 28 layers, this gives 406 weights.

Prompt-conditioned variant:

- MLP predicts the same lower-triangular weight set per prompt.

Important distinction:

- This method can alter how earlier computations are reused at later layers, not only scale each current layer write.

## 5. Experimental Setup

Purpose: make results reproducible and fair enough for the report.

### Data

Use `artifact/data/prepared/summary.json`:

- Corpora: WikiText-2 raw and `TimSchopf/medical_abstracts`.
- Prepared examples:
  - train: 20,368 examples,
  - validation: 2,561 examples,
  - test: 2,555 examples.
- Test source counts:
  - WikiText: 1,227,
  - medical: 1,328.
- Prompt/answer: 96/96 tokens.
- Seed: 42.

### Models Compared

Main table rows:

- Frozen base.
- Static write-strength.
- Prompt-conditioned write-strength.
- Static residual-stream re-aggregation.
- Prompt-conditioned residual-stream re-aggregation.

### Training Details

Include compact bullet/table:

- One epoch.
- CUDA bf16.
- SDPA attention.
- Effective batch size:
  - 32 for write-strength and static re-aggregation,
  - 30 for prompt-conditioned re-aggregation.
- Learning rates:
  - static variants: 2e-3,
  - prompt-conditioned variants: 1e-3.

### Metrics

Report:

- validation loss / perplexity,
- test loss / perplexity,
- per-source test loss/perplexity,
- MLP inspection metrics for prompt-conditioned variants.

## 6. Main Results

Purpose: make the empirical ranking unmistakable.

### Table 1: Overall Validation and Test Performance

Include columns:

| Model | Val Loss | Val PPL | Test Loss | Test PPL | Test PPL Reduction vs Frozen |
|---|---:|---:|---:|---:|---:|
| Frozen base | 2.8008 | 16.46 | 2.8222 | 16.81 | - |
| Static write-strength | 2.7442 | 15.55 | 2.7653 | 15.88 | 5.53% |
| Prompt-conditioned write-strength | 2.7415 | 15.51 | 2.7624 | 15.84 | 5.80% |
| Static re-aggregation | 2.7140 | 15.09 | 2.7330 | 15.38 | 8.53% |
| Prompt-conditioned re-aggregation | **2.7046** | **14.95** | **2.7233** | **15.23** | **9.41%** |

Key claims:

- All learned residual interventions outperform frozen base.
- Re-aggregation outperforms write-strength scaling.
- Prompt conditioning improves both intervention families, but the gain is modest relative to the architectural difference between scaling and re-aggregation.

### Table 2: Per-Domain Test Loss

Include columns:

| Model | WikiText Loss | Medical Loss |
|---|---:|---:|
| Frozen base | 3.2979 | 2.3827 |
| Static write-strength | 3.2237 | 2.3419 |
| Prompt-conditioned write-strength | 3.2201 | 2.3396 |
| Static re-aggregation | 3.1784 | 2.3215 |
| Prompt-conditioned re-aggregation | **3.1668** | **2.3136** |

Key claims:

- Improvements hold on both domains.
- WikiText has higher absolute loss, but both domains benefit.
- The best model improves test loss over frozen by 0.1311 on WikiText and 0.0691 on medical.

## 7. Controller Analysis

Purpose: answer the proposal's failure-mode question: did the prompt-conditioned controller collapse to near-static behavior?

### Table 3: MLP Output Inspection on Test Split

Include columns:

| Prompt-conditioned model | Collapse cosine | Domain-separation cosine gap | Mean variance | Mean domain difference |
|---|---:|---:|---:|---:|
| Write-strength | 0.9482 | 0.0561 | 0.1125 | 0.3712 |
| Re-aggregation | 0.8774 | 0.2105 | 0.9539 | 1.1606 |

Interpretation:

- Lower collapse cosine for re-aggregation means its predicted vectors vary more across prompts.
- Higher domain-separation gap suggests stronger domain-sensitive behavior.
- Higher variance/domain difference is expected partly because re-aggregation predicts a larger and richer lower-triangular weight vector, so avoid comparing raw magnitudes as the only evidence.
- Still, all four metrics consistently suggest re-aggregation uses prompt conditioning more actively.

Optional figure:

- Per-layer variance or per-layer domain-mean difference curves for the two prompt-conditioned models.
- If space is tight, omit the figure and keep only Table 3.

## 8. Discussion

Purpose: explain what the results mean and what they do not mean.

Main points:

- Residual-stream re-aggregation likely helps because it can control reuse of earlier layer writes, giving more expressive adaptation than independent per-layer scaling.
- Prompt conditioning helps, but the improvement is incremental in this experiment.
- The inspection results suggest prompt conditioning is not merely producing identical vectors for all prompts, especially for re-aggregation.
- The mixed-domain setting is useful for testing prompt-conditioned behavior, but this is still a small-scale continuation benchmark.

Careful wording:

- Say "suggests" and "in this setup"; avoid claims about all LMs or all PEFT methods.

## 9. Limitations

Purpose: be honest and preempt obvious reviewer questions.

Include:

- Only one base model size: Qwen3-0.6B.
- Only one training epoch and lightweight hyperparameter search.
- No direct LoRA/adapters benchmark in the final experiment.
- Medical abstracts dataset replaced PubMed for practicality.
- Continuation loss may not reflect downstream task performance.
- Prompt-conditioned re-aggregation has more trainable/output dimensions than write-strength, so some gains may come from expressivity rather than conditioning alone.
- MLP inspection uses 256 sampled examples per split.

## 10. Conclusion

Purpose: close with the concrete finding.

One-paragraph message:

- Lightweight residual interventions can improve a frozen LM on mixed-domain continuation.
- Re-aggregating the residual stream is more effective than simple write-strength scaling.
- Prompt conditioning provides consistent additional gains and appears more meaningful when paired with the richer re-aggregation mechanism.
- Best result: test PPL 15.23 vs 16.81 frozen base.

## Appendix Candidates

Use only if allowed outside 8 pages:

- Full training configs.
- Full validation/test per-domain metrics.
- Exact preprocessing details.
- Full per-layer MLP variance and domain-difference arrays.
- Script commands from `example_usuage.txt`.

## Writing Priorities

If time is limited, write in this order:

1. Main results table and result paragraph.
2. Method section with the two intervention families.
3. Experimental setup.
4. Controller analysis.
5. Introduction and conclusion.
6. Related work and limitations.

The report should keep returning to one clean conclusion: **the largest gain comes from residual-stream re-aggregation, and prompt conditioning is a useful but secondary improvement that becomes more visible with the richer re-aggregation controller.**
