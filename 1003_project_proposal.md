# DS 1003 Final Project Proposal

## Prompt-Conditioned Residual Weighting as Parameter-Efficient Fine-Tuning

**Authors:** Peng Zhao, Yu Gu, Jinghan Lei

## 1. Inspiration

We are inspired by several lightweight fine-tuning methods:

- **Attention Residuals:** Layer residual weighting adds expressiveness and may improve performance as a very lightweight architecture addition.
- **BitFit:** Fine-tuning the bias terms alone may improve performance.
- **SALT:** Rescaling and shifting the singular values of weight matrices may improve performance.

We are amazed by these lightweight fine-tuning methods and would like to extend them. In most existing methods, although the model is trained on some specific calibration text, the structures added onto the model are prompt-independent. For example, once fine-tuning is complete, the trained LoRA module is used as a static object regardless of the prompt.

This leads to our central question: what if the added structures could adapt to specific prompts during inference? One immediate use case is when the calibration text has a multi-modal distribution, such as a mixture of WikiText and math problems.

## 2. Question

Can prompt-conditioned residual weighting match or outperform static residual scaling and standard PEFT baselines under a similar parameter budget?

## 3. Method

Given a prompt-answer pair, a frozen LLM, and a small MLP, one training forward pass is defined as follows:

1. We encode the prompt using the frozen LLM and use the hidden state of the final prompt token as the fixed-length and assumed informative conditioning vector \(h\).
2. Pass \(h\) into a small MLP to generate residual scaling weights \((a_1, a_2, \ldots)\).
3. Apply the weights \((a_1, a_2, \ldots)\) to the LLM's layer residuals and generate the full predicted answer.
4. Compute the loss against the true answer and backpropagate only through the MLP.

The core idea is to make the fine-tuning mechanism prompt-dependent. In our case, the residual weighting is conditioned on the prompt through the hidden-state representation \(h\) and the small MLP, while the backbone LLM remains frozen.

## 4. Expected Result

After sufficient training of the MLP, we will benchmark the fine-tuned LLM against the following baselines:

- Our method: prompt-conditioned residual weighting,
- Static residual weighting,
- The baseline model,
- Possibly other PEFT methods, if including them turns out to be meaningful.

We will evaluate whether prompt-conditioned residual weighting provides measurable improvement over static residual weighting and the baseline model. In addition, we will analyze the learned residual weights on a test set. A possible failure mode is that the weights collapse to nearly static values, in which case the method would behave similarly to static residual scaling.

## 5. Discussion

The specific PEFT method chosen is not the central focus of this project. We use residual weighting mainly because it is probably the simplest option and was recently introduced by the KIMI team. Our main interest is in whether prompt-conditioning itself can make a meaningful difference.

For the LLM, we will use **Qwen3-0.6B**. For the calibration text, we will use **WikiText-2** and **PubMed abstracts**.
