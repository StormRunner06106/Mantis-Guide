# 04 — LBFGS Challenges (`ETHLBFGS` 1h / `BTCLBFGS` 6h)

The most structured probabilistic challenge. Miners submit a **17‑dimensional embedding** combining a 5‑bucket regime classifier and 12 quantile‑tail exceedance probabilities. Validators score the two parts independently and **blend 75/25**.

| Property | ETHLBFGS | BTCLBFGS |
|---|---|---|
| **Ticker** | `ETHLBFGS` (price_key `ETH`) | `BTCLBFGS` (price_key `BTC`) |
| **Loss function key** | `lbfgs` | `lbfgs` |
| **Embedding dim** | `17` | `17` |
| **Horizon (`blocks_ahead`)** | `300` blocks (1h) | `1800` blocks (6h) |
| **Challenge weight** | `3.5` | `2.875` |
| **Scorers** | `compute_lbfgs_salience()` and `compute_q_path_salience()` in `bucket_forecast.py` |

---

## 1. What you are predicting

Two things at the same time, packed into one 17‑vector per sample:

### Part A — 5‑bucket regime classifier (p[0..4])

Let `z = (r_h − 0) / σ_60(t)` where `r_h` is the endpoint log return over `H = blocks_ahead` and `σ_60(t)` is a rolling 60‑minute sigma. Bucket boundaries:

| Bucket | z range | Meaning |
|---|---|---|
| 0 | `z ≤ −2σ` | big down |
| 1 | `−2σ < z < −1σ` | moderate down |
| 2 | `−1σ ≤ z ≤ 1σ` | neutral |
| 3 | `1σ < z < 2σ` | moderate up |
| 4 | `z ≥ 2σ` | big up |

You submit `p = [p0, p1, p2, p3, p4]` — a valid probability distribution over these 5 buckets.

### Part B — Opposite‑move exceedance probabilities (q[5..17])

For the 4 *tail* buckets (0, 1, 3, 4 — not the center bucket 2), and at 3 threshold levels `m ∈ {0.5, 1.0, 2.0}`, you submit `P(|opposite excursion| ≥ m·σ | realized bucket = c)`.

- For negative buckets (`c = 0` or `c = 1`), "opposite" means an **UP** excursion: does `max(log P[u] − log P[t−1])` over the window reach `m·σ`?
- For positive buckets (`c = 3` or `c = 4`), "opposite" means a **DOWN** excursion: does `min(...)` reach `−m·σ`?

That gives `4 tail buckets × 3 thresholds = 12` numbers.

### Full layout (D = 17)

```
Index     Meaning
[0:5]     p[0..4]  — 5‑bucket regime probabilities (must sum to 1)
[5:8]     Q(c=0) at thresholds [0.5σ, 1.0σ, 2.0σ]   — opposite = UP
[8:11]    Q(c=1) at thresholds [0.5σ, 1.0σ, 2.0σ]   — opposite = UP
[11:14]   Q(c=3) at thresholds [0.5σ, 1.0σ, 2.0σ]   — opposite = DOWN
[14:17]   Q(c=4) at thresholds [0.5σ, 1.0σ, 2.0σ]   — opposite = DOWN
```

---

## 2. What you submit

```python
import numpy as np

p = np.array([0.05, 0.15, 0.60, 0.15, 0.05])           # must sum to 1
q = np.random.uniform(0.01, 0.99, 12).tolist()           # 12 probs in (0,1)

embeddings["ETHLBFGS"] = p.tolist() + q                  # length 17
embeddings["BTCLBFGS"] = p.tolist() + q
```

### Hard rules

- `p[0..4]` strictly in `(0, 1)` and sum to `1.0 ± 1e-6`.
- All `q[5..17]` strictly in `(0, 1)`.
- No NaNs/infs; validator will clamp to `(1e-6, 1−1e-6)` and renormalize `p` if needed, but submit clean values.

---

## 3. How validators score you — two independent paths

Validators run **two separate scorers** on the same matrix and blend the results **75% classifier + 25% Q‑path**.

### Path A — Classifier salience (p‑only) — 75% weight

Implemented in `compute_lbfgs_salience()` (`bucket_forecast.py`).

1. Extract only the `p[0..5]` slice from every miner’s embedding.
2. Compute the realized bucket labels `y ∈ {0,1,2,3,4}` from `(r_h, σ_60)`.
3. Represent each miner's submission by its **argmax bucket** at each sample (a categorical feature per miner).
4. Walk‑forward with daily steps and an embargo `max(60 bars, H_steps)`.
5. For each walk‑forward window, fit **5 per‑class L2 logistic regressions** (one `y_c = 1[label == c]` each) using **one‑hot encoded miner argmaxes** as features.
6. Per‑miner per‑class importance is `β_{j,c}²`. Summed across classes:
   \[
   \text{imp}_j^{\text{cls}} = \sum_{c=0}^{4} \beta_{j,c}^{2}
   \]
7. **Uniqueness penalty:** miners whose argmax prediction overlaps `> 85%` with a higher‑ranked miner are suppressed. This kills trivial copycats.
8. Vectorized balanced‑accuracy evaluation is used for time weighting of contribution across windows (exponential half‑life, see `_time_weights()` in `model.py`).

### Path B — Q‑path salience — 25% weight

Implemented in `compute_q_path_salience()`. 12 independent mini‑models, one per `(bucket c, threshold m)` pair:

1. For each `(c, m)`:
   - Compute the **realized opposite‑move label** using the realized bucket and actual min/max excursion vs `m·σ`.
   - Mask to samples where the realized bucket equals `c` (this is the **gating** — you only get scored on your Q(c,m) at those timestamps).
   - Fit a binary L2 logistic on `logit(miner_q_c_m)` as features.
2. Importance for miner `j` is the **average of `|β_j|` across the 12 sub‑models**.
3. Class‑weighted loss is used throughout (imbalanced positive rates, especially at `2σ`).

### Blending

```
s_final(j) = 0.75 * s_classifier(j) + 0.25 * s_q(j)
```

Both salience vectors are individually top‑K (K=50) renormalized with exponential rank decay before the blend, then the blended vector is renormalized again to sum to 1.

---

## 4. Why this design?

- **Classifier path is the main signal.** Regime classification is the bread‑and‑butter task; most information value lives in `p`.
- **Q‑path adds path‑dependent resolution.** A miner with a good `p` but no sense of opposite excursions gets only 75% credit. A miner that correctly prices `P(big pullback before a big rally)` can pick up the extra 25%.
- **Argmax, not soft probabilities, in the classifier:** this makes the uniqueness penalty straightforward and prevents cheap softmax‑smoothing tricks.
- **Per‑class `β²`:** every class contributes independently; a miner that is great at calling big drops but bad at neutral still gets recognized in its class.
- **Uniqueness penalty > 85% overlap:** the ultimate Sybil killer on top of the L2 split.

---

## 5. Timing and activation rules

- Minimum data: ~5 days of samples before LBFGS is scored at all. Until then, the challenge is **skipped** in the aggregation (not equal‑weighted out).
- Activation: a miner's first non‑zero 17‑vector starts their accrual; all‑zero rows before that are ignored for that miner.
- Update cadence: saliences recomputed every `WEIGHT_CALC_INTERVAL = 1000` blocks with walk‑forward step `samples_per_day` and embargo `max(60 bars, H_steps)`.

---

## 6. What gets you zero weight

- `p` that does not sum to 1 or contains 0/1 values.
- Argmax of `p` being constant over time (uniqueness penalty + low coefficient).
- Q values stuck at 0.5 — uninformative; the Q‑path contributes 0.
- Copying a top miner — >85% argmax overlap triggers the uniqueness penalty.

---

## 7. Tips for miners

- **Calibrate, don’t overconfident.** The 2σ buckets are rare (~2% each); if you hammer `p[0] = 0.3`, your classifier coefficient gets punished in out‑of‑sample fitting.
- **Diversify the argmax pattern.** Even small rebalances between adjacent buckets can push you out of the 85% overlap zone.
- **You can start with `p`‑only.** Leave all `q = 0.5` until you have a Q model; you forfeit the 25% Q weight but you still get 75%.
- **Use the same σ definition** as the validator (60‑min rolling sigma of 1‑step log returns).
- **ETHLBFGS (1h)** and **BTCLBFGS (6h)** are separate; solve them with separate models tuned to the horizon.

---

## 8. Pointers in the code

| What | Where |
|---|---|
| Dispatch entry | `model.py` → `multi_salience()` → `loss_type == "lbfgs"` |
| Classifier scorer | `bucket_forecast.py` → `compute_lbfgs_salience()` |
| Q‑path scorer | `bucket_forecast.py` → `compute_q_path_salience()` |
| Blending | `model.py`, the `s_cls / s_q / 0.75 / 0.25` block |
| Top‑K renorm | `_topk_renorm()` in `model.py` |
| Project‑level mining guide | `lbfgs_guide.md` in the repo root |
| Challenge specs | `config.CHALLENGES` entries `ETH-LBFGS`, `BTC-LBFGS-6H` |
