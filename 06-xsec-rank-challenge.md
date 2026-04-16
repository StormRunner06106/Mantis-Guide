# 06 — XSEC‑RANK Challenge (`MULTIXSEC`, 33 assets, 4h)

A **cross‑sectional** ranking challenge. Don’t predict returns — predict **which assets will beat the cross‑sectional median** over the next 4 hours. Every sample is re‑used 33 times (once per asset), pooled into a single large binary classification.

| Property | Value |
|---|---|
| **Ticker** | `MULTIXSEC` |
| **Loss function key** | `xsec_rank` |
| **Embedding dim (per asset)** | `1` — a single score in `[-1, 1]` |
| **Assets** | 33 (same list as MULTI‑BREAKOUT) |
| **Horizon (`blocks_ahead`)** | `1200` blocks ≈ 4 hours |
| **Challenge weight** | `3.0` |
| **Scorer** | `compute_xsec_rank_salience()` in `xsec_rank.py` |

---

## 1. What you are predicting

For each timestep `t`, for each asset `a`:

\[
r^{(a)}_{t \to t+h} = \log P^{(a)}_{t+h} - \log P^{(a)}_t
\]

\[
y_{t, a} = \mathbf{1}\!\Big[\, r^{(a)}_{t \to t+h} \;>\; \text{median}_a\!\big(r_{t \to t+h}\big)\, \Big]
\]

That is, does asset `a` beat the median of all 33 assets' forward returns over the same window?

By construction, each timestep has exactly `~median_count` positive labels (16/17 out of 33), so the base rate is pinned to 50% — there is no majority class to exploit.

---

## 2. What you submit

A **dict keyed by asset**, each value a single float in `[-1, 1]`:

```python
from config import BREAKOUT_ASSETS

embeddings["MULTIXSEC"] = {
    asset: float(np.clip(your_score(asset), -1, 1))
    for asset in BREAKOUT_ASSETS
}
```

Larger positive → more confident the asset will be an outperformer. Larger negative → more confident it will underperform. Magnitude matters — your score is used directly as a **feature** in the logistic regression (after sanitation), not as a probability.

Missing assets default to `0.0` (neutral).

---

## 3. How labels are built

In `xsec_rank.py`:

1. For each sample `t`, compute forward log returns `r_{t→t+h}(a)` for every asset `a`.
2. Compute the **cross‑sectional median** `m_t = median_a(r_{t→t+h}(a))`.
3. For every `(t, a)` pair, label `y_{t,a} = 1[r(a) > m_t]`.
4. Pool all `(t, a)` pairs into one dataset → `T × 33` samples per miner. This is a huge multiplier over a single‑asset challenge.

Embargo / LAG: `max(LAG, ahead)` samples between train and validation to avoid leakage of the forward window.

---

## 4. How validators score you — walk‑forward meta‑model

Same recipe family as the binary challenge, but **over the pooled (t, a) dataset**:

### Step 1 — Assemble features

For each sample in the pooled dataset, each miner contributes **one feature**: their score for the asset at that sample. Matrix shape: `(T × N_assets, N_miners)`.

### Step 2 — Feature selection (top‑K by AUC)

For each miner `j` independently, compute the **univariate AUC** of their score column vs the pooled label. Keep the top `TOP_K = 20` miners whose AUC > 0.5.

(Lower K than the binary challenge because the meta‑model fit is heavier here — there are 33× more rows.)

### Step 3 — Walk‑forward segments

Split the timeline into walk‑forward segments. For each segment:

1. Fit an **L2 logistic meta‑model** on the pooled `(t, a)` training rows of the *selected* miners.
2. Evaluate AUC on the validation portion of the segment.
3. Per‑miner per‑segment importance:
   \[
   w_j^{(\text{seg})} = |\beta_j| \cdot \max\!\Big(\frac{\text{AUC}_{\text{meta}} - 0.5}{0.5},\; 0\Big)
   \]
   A meta‑model that is no better than random (AUC ≤ 0.5) contributes **zero** to all miners for that segment. A perfect meta‑model multiplies `|β_j|` by `1`.

### Step 4 — Recency‑weighted segment aggregation

Segments are aggregated with an exponential recency weight:

\[
w_i = \gamma^{n - 1 - i}, \quad \gamma = 0.5^{1/\text{HALFLIFE}}
\]

So the most recent segments dominate the aggregated importance. Salience is renormalized across miners to sum to 1.

---

## 5. Why this design?

- **Cross‑sectional label destroys the market beta.** Whether all crypto ran up or crashed, the label is about *relative* performance — a much cleaner signal.
- **Median split → 50% base rate.** Perfectly balanced classes, no class‑weighting gymnastics needed.
- **33× sample multiplier.** Same calendar time, 33× the training rows. The meta‑model converges much faster than in a single‑asset challenge.
- **AUC‑scaled coefficients.** Keeps overall salience mass honest: if the current market is unpredictable, all miners' importances are proportionally dampened.
- **Walk‑forward + recency weighting.** Markets drift; what worked 30 days ago matters less than what works today.

---

## 6. What gets you zero weight

- Submitting the same score across all assets → no cross‑sectional information → univariate AUC ≈ 0.5 → dropped.
- Submitting a constant per asset over time → no temporal information.
- Submitting values outside `[-1, 1]` → clipped, which may crush extreme predictions.
- Missing assets → those (t, a) rows are treated as 0 for you, which is neutral but takes up slots.

---

## 7. Tips for miners

- **Scale matters.** Because the score is a raw feature, not a probability, a miner with more decisive scores (closer to ±1) gets larger coefficients *if they are correct*. Confidence‑calibrate — submit `±0.8` for high conviction, `±0.2` for weak signals.
- **Focus on relative ranking, not absolute returns.** Your single‑asset predictor may be great at forecasting BTC direction, but for this challenge what matters is how your BTC score compares to your ADA score.
- **Cover all 33 assets.** Gaps reduce your effective sample count.
- **Update at the 60 s cadence.** Even though the label is 4h forward, the scorer still needs your prediction at every `sidx`.

---

## 8. Pointers in the code

| What | Where |
|---|---|
| Dispatch entry | `model.py` → `multi_salience()` → `loss_type == "xsec_rank"` |
| Scorer | `xsec_rank.py` → `compute_xsec_rank_salience()` |
| Label construction | same file, cross‑sectional median logic |
| Recency weights | `_time_weights()` in `model.py` |
| Challenge spec | `config.XSEC_RANK_CHALLENGE` |
| Asset list | `config.BREAKOUT_ASSETS` (reused) |
