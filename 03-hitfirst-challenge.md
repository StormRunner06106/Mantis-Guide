# 03 — HITFIRST Challenge (`ETHHITFIRST`)

A three‑way barrier challenge: predict which of the two ±1σ barriers is hit **first** during the next 500 blocks, or whether **neither** is hit.

| Property | Value |
|---|---|
| **Ticker** | `ETHHITFIRST` (price_key: `ETH`) |
| **Loss function key** | `hitfirst` |
| **Embedding dim** | `3` |
| **Horizon (`blocks_ahead`)** | `500` blocks |
| **Challenge weight** | `2.5` |
| **Scorer** | `compute_hitfirst_salience()` in `hitfirst.py` |

---

## 1. What you are predicting

For each sample `t`, consider the log‑price path during the next 500 blocks. Define `σ` as the rolling sigma of 1‑step log returns at `t`.

- **Label 0 (up first):** price path hits `+σ` before it hits `-σ`.
- **Label 1 (down first):** price path hits `-σ` before it hits `+σ`.
- **Label 2 (neither):** within 500 blocks, neither barrier is hit.

You submit the **probability vector** for this 3‑way outcome.

---

## 2. What you submit

A 3‑dim vector **summing to 1** with all components strictly in `(0, 1)`:

```python
embeddings["ETHHITFIRST"] = [0.4, 0.35, 0.25]
#                            │     │     └── P(neither)
#                            │     └──────── P(down first)
#                            └────────────── P(up first)
```

### Hard rules

- Values must be strictly in `(0, 1)` — no hard 0s or 1s (log‑loss / logit would blow up).
- Must sum to 1.
- Validator will clip to `[EPS, 1-EPS]` and renormalize, but you should submit clean values.

---

## 3. How labels are built

In `hitfirst.py` → `compute_hitfirst_salience()`:

1. Compute `horizon_steps = round(blocks_ahead / sample_every) = 500 / 5 = 100` samples.
2. Compute the forward log return `r_h(t) = log(price[t + H]) − log(price[t])` and a rolling sigma `σ(t)` over a window ≥ 5 days of samples.
3. For each valid `t`:
   - Slice the path `seg = log_price[t+1 : t+1+H] − log_price[t]`.
   - `idx_up = first index where seg ≥ σ(t)`.
   - `idx_dn = first index where seg ≤ −σ(t)`.
   - Label = `0` if `idx_up < idx_dn`, `1` if `idx_dn < idx_up`, `2` if neither.

So the label is **path‑dependent**, not endpoint‑dependent. The first barrier touched wins, and "neither" is a real outcome that miners must price.

---

## 4. How validators score you

HITFIRST deviates from the binary recipe — there is **no walk‑forward, no meta‑model, and no feature selection**. It runs a single fit per direction on all valid samples.

### Step 1 — Transform miner probabilities to logit scores

Clip `X` to `(EPS, 1-EPS)`, renormalize rows, then take the logit of the up‑component and the down‑component:

- `up_scores[t, j] = logit(P_up(t, j))`
- `dn_scores[t, j] = logit(P_dn(t, j))`

Rows where miner `j` did not submit at all are zeroed out for that miner only.

### Step 2 — Two binary L2 logistic regressions

- **Up model:** regress `y_up = 1[label == 0]` on `up_scores`. Penalty `l2`, `C = 1.0`, `class_weight = "balanced"`, `lbfgs` solver.
- **Down model:** regress `y_dn = 1[label == 1]` on `dn_scores`. Same settings.

Both fits use **all valid samples at once** — no walk‑forward.

### Step 3 — Salience

Per‑miner importance is the sum of absolute coefficients from the two fits:

\[
\text{imp}_j = |\beta_j^{\text{up}}| + |\beta_j^{\text{down}}|
\]

Normalize across miners so the dict sums to 1.

---

## 5. Requirements for the scorer to run

- At least `max(MIN_REQUIRED_SAMPLES, 5 days * samples_per_day)` samples of price history.
- Both `y_up` and `y_dn` must contain both classes in the window.
- Embedding dim must be exactly 3.

If any of these fail, HITFIRST simply returns `{}` and the challenge is skipped in the aggregation for that cycle.

---

## 6. Why this design?

- **No walk‑forward needed:** the "which barrier first" label is already strongly path‑dependent and low‑autocorrelation, so in‑sample overfitting risk is small relative to the binary challenge.
- **Logit transform of probabilities:** using `logit(p)` as the regression feature is the correct link function for logistic regression. A miner submitting well‑calibrated probabilities gets a coefficient near 1; a miner submitting garbage gets a coefficient near 0.
- **Two independent fits:** up‑hit and down‑hit are different labels with different base rates, so a single 3‑class model would be dominated by the majority class. Two binary models with `class_weight="balanced"` avoids that.
- **Neither class is never regressed:** a miner can still get importance from accurately pricing the *existence* of a hit in either direction, without having to predict the "neither" class well.

---

## 7. What gets you zero weight

- Submitting degenerate probs (all three ≈ 0.33) — logit scores are near 0, coefficient near 0.
- Not submitting for enough samples — you never enter the regression.
- Submitting components outside `(0, 1)` — validator clips but extreme values carry very little signal after clipping.
- Submitting only for a short window shorter than the 5‑day minimum.

---

## 8. Tips for miners

- **Calibrate the "neither" bucket explicitly.** Many miners will underweight it. A 500‑block horizon with ±1σ barriers has a meaningful probability of no hit.
- **σ is computed on the validator side** over 60‑minute returns. Use the same definition locally when training.
- **Only up_scores and down_scores enter the regression.** `P(neither) = 1 − P_up − P_dn` is implicitly captured through the sum constraint, but the "neither" column is never directly used — do not waste capacity optimizing it.
- **Submit every sample.** A miner that only submits during hit events gets sparse coverage and low coefficient mass.

---

## 9. Pointers in the code

| What | Where |
|---|---|
| Dispatch entry | `model.py` → `multi_salience()` → `loss_type == "hitfirst"` |
| Scorer | `hitfirst.py` → `compute_hitfirst_salience()` |
| Label construction | same function, the `for t in range(len_r):` loop |
| Rolling sigma | `utils.py` → `rolling_std_fast()` |
| Logit transform | `utils.py` → `logit()` |
| Challenge spec | `config.CHALLENGES` entry `"ETH-HITFIRST-100M"` |
