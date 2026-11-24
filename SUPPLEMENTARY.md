# FedVARE — Supplementary Material

This file collects the material that did not fit in the 6-page ESANN paper. Section numbers refer to the paper.

1. [Algorithm](#1-algorithm)
2. [Variance–range lemma (Sec. 2.1)](#2-variancerange-lemma-sec-21)
3. [Reference-set construction details](#3-reference-set-construction-details)
4. [Datasets and splits](#4-datasets-and-splits)
5. [Models and common training settings](#5-models-and-common-training-settings)
6. [Per-method hyperparameters](#6-per-method-hyperparameters)
7. [Baseline details](#7-baseline-details)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [IID partitions](#9-iid-partitions)

---

## 1. Algorithm

**Algorithm 1 — FedVARE**

```
Require: w_0, R rounds, U local steps, proxy pool D_eval, lambda, eps > 0
 1: Select S_0 and obtain  w~_i <- LocalTrain(w_0, D_i, U),  i in S_0          (probe round)
 2: Compute V(z) for all z in D_eval and build the reference set D*_val         (once)
 3: Set w_{0,i} <- w~_i,  i in S_0                                              (round 0 reuses the probes)
 4: for t = 0, ..., R-1 do
 5:     if t > 0: select S_t and obtain w_{t,i} <- LocalTrain(w_t, D_i, U), i in S_t
 6:     L_{t,g} = mean loss of w_t on D*_val ;  L_{t,i} = mean loss of w_{t,i} on D*_val
 7:     s = 1 / (L + eps)  for the global model and every client model
 8:     q_{t,g} = s_{t,g} / Z_t ,  q_{t,i} = s_{t,i} / Z_t ,  Z_t = s_{t,g} + sum_j s_{t,j}
 9:     w_{t+1} = q_{t,g} w_t + sum_{i in S_t} q_{t,i} w_{t,i}
10: end for
11: return w_R
```

In cross-silo settings (AG News, all clients participate) the probes are the first-round client models.
In cross-device settings a separate warm-up round can provide them. The probe round adds no extra
training or communication: its client models are reused as the round-0 updates (line 3).

**Implementation note.** The released code uses `s = 1 / L` (no `eps`); the losses are strictly positive
in all runs.

## 2. Variance–range lemma (Sec. 2.1)

Let `l_i(z) = l(w~_i; z)` be the loss of probe model `i` on candidate `z`, with cross-client mean
`l̄(z)` and variance

```
V(z) = (1/|S_0|) * sum_{i in S_0} ( l_i(z) - l̄(z) )^2 ,
```

and let `R(z) = max_i l_i(z) - min_i l_i(z)` be the range of client losses.

**Lemma.** For every reference sample `z`,  `V(z) <= R(z)^2 / 4`, equivalently `R(z) >= 2 sqrt(V(z))`.

**Proof.** Let `X_z` be the random variable obtained by picking a client uniformly from `S_0` and setting
`X_z = l_i(z)`. Then `Var(X_z) = V(z)`, and the support of `X_z` lies in `[min_i l_i(z), max_i l_i(z)]`,
an interval of length `R(z)`. Popoviciu's inequality on variances gives `Var(X_z) <= R(z)^2 / 4`.
Substituting `Var(X_z) = V(z)` and rearranging proves the claim. ∎

**Interpretation.** A sample with large cross-client loss variance necessarily separates the best and the
worst client model by at least `2 sqrt(V(z))` in loss. Because the losses come from the conditional
distributions of the locally adapted models, this separation is an observable signal of the differences
induced by local fine-tuning, which is what the inverse-loss weights need.

## 3. Reference-set construction details

- **Sample loss.** Length-normalized token-level negative log-likelihood
  `l(w; (x, y)) = -(1/T) * sum_r log p_w(y_r | x, y_<r)`; normalizing by the target length makes losses
  comparable across examples. For the ELECTRA classifier this is the cross-entropy of the label.
- **Class-wise threshold.** Candidates are grouped by label. For class `c`, with `mu_c` and `sigma_c` the
  mean and standard deviation of `V` over the candidates of that class, a sample is a high-variance candidate if
  `V(z) > mu_c + lambda * sigma_c`. We use **lambda = 2** in all experiments. Thresholding per class stops
  classes with intrinsically higher losses from dominating the selection.
- **Budget.** The threshold sets the budget: `K_bal = max_c |{z in class c : V(z) > mu_c + lambda sigma_c}|`,
  and every class contributes its `K_bal` highest-variance samples, so `K = |C| * K_bal` and the set is
  class-balanced. A class with fewer than `K_bal` candidates is filled with its next-highest-variance samples.
  Using the maximum rather than the minimum keeps every class equally represented without discarding the most
  informative samples of the class with the most candidates.
- **Resulting sizes.** Across seeds, `K` is 100–200 on AG News and 50–100 on Sentiment140.
  In the selection ablation (Sec. 3.2) `K` is fixed to 100 for all strategies so that they are compared at an
  equal budget.
- **Ablation strategies (Sec. 3.2).** *High*: the `K` samples with the highest `V` (FedVARE). *Mid*: samples
  closest to the class mean of `V`. *Low*: the lowest `V`. *Random*: `K` random samples, three draws per seed,
  averaged per seed first. *Full*: the entire 8,000-sample pool (four seeds).
- The reference set is built once and reused in all later rounds. It is used only for aggregation, never for
  local training, tuning or final evaluation.

## 4. Datasets and splits

| Dataset      | Task                   | Train     | Proxy pool `D_eval` | Test  |
|--------------|------------------------|-----------|---------------------|-------|
| AG News      | 4-class topic          | 114,000   | 6,000               | 7,600 |
| Sentiment140 | binary sentiment       | 1,592,000 | 8,000               | 359   |

- AG News: `D_eval` is a stratified 5% split of the original training set; the standard test split is used.
- Sentiment140: `D_eval` is a stratified 0.5% split of the original training set. The test set is the manually
  annotated test set with the 139 neutral tweets removed (359 positive/negative tweets).
- `D_eval` is disjoint from the client data and the test set. Within a seed, all methods share the same pool.
- **Non-IID partitions.** Dirichlet label skew (FedNLP): `alpha = 0.1` on AG News, `alpha = 0.5` on
  Sentiment140, uniform label prior, balanced client sizes.

## 5. Models and common training settings

| Setting                        | AG News                               | Sentiment140                      |
|--------------------------------|---------------------------------------|-----------------------------------|
| Model                          | T5-small (seq2seq, label as text)     | ELECTRA-small discriminator + classification head |
| Fine-tuning                    | LoRA on q, k, v, o; r = 16, alpha = 32, dropout 0.05, no bias | full fine-tuning |
| Max. sequence length           | 256 input / 8 target tokens           | 96 tokens                         |
| Local batch size               | 64                                    | 512                               |
| Clients / per round            | 30 / 30 (all)                         | 100 / 10                          |
| Rounds                         | 50                                    | 100                               |
| Local epochs per round         | 1                                     | 1                                 |
| Dirichlet alpha (non-IID)      | 0.1                                   | 0.5                               |
| Gradient clipping (L2 norm)    | 1.0                                   | 1.0                               |
| LR schedule                    | cosine decay over rounds, from the initial LR to `eta_min` (table below) | same |
| Sophia Hessian refresh         | every k = 10 local steps (Gauss–Newton–Bartlett estimator) | same |
| Optimizer state                | each client keeps its optimizer state across the rounds it participates in | same |
| Seeds                          | 342365, 8541865, 785638984, 1486496, 2658 | same |

Hyperparameters were selected on server-side validation data only; the test set is never used for tuning,
early stopping or model selection.

## 6. Per-method hyperparameters

Values are taken from the released notebooks. "Adam" and "AdamW" use the PyTorch defaults
`betas = (0.9, 0.999)`, `eps = 1e-8` unless stated. Sophia is SophiaG (Liu et al., 2024).

### AG News (T5-small + LoRA)

| Method          | Local optimizer | Initial LR | `eta_min` | Weight decay | Other                                            |
|-----------------|-----------------|-----------:|----------:|-------------:|--------------------------------------------------|
| FedAvg-Adam     | Adam            | 6.0e-4     | 8e-7      | 0            |                                                  |
| FedAvg-Sophia   | SophiaG         | 3.5e-4     | 2e-7      | 0.1          | betas (0.96, 0.98999), rho = 0.01                |
| FedAdamW        | AdamW           | 4.5e-3     | 1e-8      | 1e-12        | correction weight gamma = 0.59 (Sec. 7)          |
| FedLBW          | SophiaG         | 2.8e-4     | 2e-7      | 0.1          | betas (0.96, 0.98999), rho = 0.01; proxy 1,000 (250 / class) |
| FedVARE-Adam    | Adam            | 3.4e-4     | 9e-7      | 0            | reference set 100–200 samples                    |
| FedVARE-Sophia  | SophiaG         | 3.5e-4     | 1e-7      | 0.1          | betas (0.965, 0.99), rho = 0.04; reference set 100–200 samples |

### Sentiment140 (ELECTRA-small)

| Method          | Local optimizer | Initial LR | `eta_min` | Weight decay | Other                                            |
|-----------------|-----------------|-----------:|----------:|-------------:|--------------------------------------------------|
| FedAvg-Adam     | Adam            | 1.3e-4     | 5e-8      | 3e-7         |                                                  |
| FedAvg-Sophia   | SophiaG         | 4.0e-5     | 1e-8      | 1e-7         | betas (0.965, 0.99), rho = 0.04                  |
| FedAdamW        | AdamW           | 2.3e-4     | 1e-8      | 5e-6         | correction weight gamma = 0.6 (Sec. 7)           |
| FedLBW          | SophiaG         | 9.0e-5     | 5e-8      | 5e-7         | betas (0.965, 0.99), rho = 0.04; proxy 1,000 (500 / class) |
| FedVARE-Adam    | Adam            | 3.7e-4     | 5e-7      | 0            | reference set 50–100 samples                     |
| FedVARE-Sophia  | SophiaG         | 9.0e-5     | 5e-8      | 5e-7         | betas (0.965, 0.99), rho = 0.04; reference set 50–100 samples |

**Selection ablation (Sec. 3.2):** FedVARE-Sophia settings with initial LR 1.1e-4, all reference sets of size 100.
**Held-out clients (Sec. 3.3):** clients 90–99 of the 100 never train; their local data are used only to evaluate
the global model after every round.

## 7. Baseline details

- **FedAvg-Adam / FedAvg-Sophia.** Local training with the stated optimizer; the server takes the plain average
  of the returned client models (client sizes are balanced, so this equals size-proportional FedAvg weighting).
- **FedAdamW** (Liu et al., AAAI 2026). Each client runs AdamW whose first moment is reset every round and
  whose second moment is initialized from the server-aggregated second moment. After every AdamW step, the
  client applies the global correction `x <- x - lr * gamma * Delta_G`, where
  `Delta_G = -(x^{r+1} - x^r) / (S K lr)` is the previous global update normalized by the number of
  participating clients `S` and local steps `K`; `gamma = 0.59` on AG News and `0.6` on Sentiment140.
  The server averages the client models and their AdamW second moments.
- **FedLBW** (Kundroo et al., 2026). Each client model is weighted by `1 / L_c` on a fixed class-balanced proxy
  set of 1,000 samples drawn from the same `D_eval` pool FedVARE selects from (fixed per seed); only the client
  models are averaged (no global-model term). The local optimizer and schedule are the same as for the
  corresponding FedAvg baseline.
- **Relation of FedVARE to FedLBW.** With `w̄_t` the inverse-loss average of the client models alone (the FedLBW
  update), the FedVARE update is `w_{t+1} = w_t + (1 - q_{t,g}) (w̄_t - w_t)`: an exponential moving average
  whose coefficient is recomputed every round from the reference losses. When all losses are similar,
  `q_{t,g} ≈ 1/(M+1)` for `M` participating clients, so the damping is mild.

## 8. Evaluation protocol

- **Time to target (Table 1).** The target `A*` is the final accuracy of FedAvg-Adam (mean over its last ten
  rounds): 85.24% on AG News and 82.07% on Sentiment140. A round counts only if the accuracy reaches `A*` and does
  not drop more than `delta = 2` points below it within the next `W = 5` rounds. Rounds are counted from 0; we
  report the median and range over the seeds that reach the target.
- **Stability (Table 2).** Standard deviation of the per-round global and held-out accuracy over rounds 50–99,
  and the rounds needed to reach 80% accuracy, counted once the 4-round moving average stays above it.
- **Generalization gap.** `G_t = A_t^global - (1/10) * sum_{i=1}^{10} A_{t,i}^held-out`. A small positive gap means
  similar performance on the global test set and on unseen clients.

## 9. IID partitions

Under IID partitions client updates are already aligned and almost all methods reach the target within a few
rounds, so FedVARE gives no gain. For reference, the highest sustained accuracy (accuracy held for `W = 5`
rounds within `delta = 2` points; mean ± std over five seeds) and the round at which each method reaches the
common sustained accuracy of the compared set, from an earlier evaluation without FedLBW:

| Method          | AG News acc. (%) | Round | Sentiment140 acc. (%) | Round |
|-----------------|-----------------:|------:|----------------------:|------:|
| FedAvg-Adam     | 91.58 ± 0.08     | 6     | 82.31 ± 1.99          | 33    |
| FedAvg-Sophia   | 92.03 ± 0.03     | 4     | 86.35 ± 2.01          | 1     |
| FedAdamW        | 88.17 ± 0.55     | 44    | 85.31 ± 0.27          | 1     |
| FedVARE-Adam    | 90.94 ± 0.15     | 8     | 86.07 ± 0.84          | 1     |
| FedVARE-Sophia  | 93.02 ± 0.05     | 4     | 86.42 ± 0.83          | 1     |

Common sustained accuracy: 88.17% (AG News), 82.31% (Sentiment140).
