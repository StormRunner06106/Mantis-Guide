# 07 — FUNDING‑XSEC Challenge (`FUNDINGXSEC`, 20 assets, 8h)

Same scoring recipe as `XSEC-RANK`, but on a **completely different dataset**: perpetual‑futures **funding rate changes**. The hardest challenge in the subnet — funding rate data is noisier, more autocorrelated, and available from multiple not‑always‑consistent sources (OKX / HyperLiquid / CoinGlass).

| Property | Value |
|---|---|
| **Ticker** | `FUNDINGXSEC` |
| **Loss function key** | `funding_xsec` |
| **Embedding dim (per asset)** | `1` — a single score in `[-1, 1]` |
| **Assets** | 20 |
| **Horizon (`blocks_ahead`)** | `2400` blocks ≈ 8 hours |
| **Challenge weight** | `4.0` (second highest) |
| **Scorer** | `compute_funding_xsec_salience()` in `funding_xsec.py` |

---

## 1. What you are predicting

For each timestep `t`, for each asset `a` (of the 20 funding‑tracked assets):

\[
\Delta f^{(a)}_t = f^{(a)}_{t+h} - f^{(a)}_{t}, \qquad h = 2400 \text{ blocks (8h)}
\]

\[
y_{t,a} = \mathbf{1}\!\Big[\, \Delta f^{(a)}_t \;>\; \text{median}_a(\Delta f)\, \Big]
\]

Does asset `a`'s funding rate **change** more than the cross‑sectional median change over the next 8h?

### Why the **change**, not the **level**?

Funding rate levels have extreme autocorrelation (`φ ≈ 0.97` per 8h period). Using levels would mean "the predictor just needs to read the current level" — trivially high AUC but zero actual signal. Taking the first difference kills autocorrelation. Subtracting the cross‑sectional median then kills the market‑wide funding regime (beta). What is left is **asset‑specific funding deviation** — the real alpha.

Base rate is exactly 50% by construction.

---

## 2. What you submit

A dict keyed by asset, each value a single float in `[-1, 1]`:

```python
from config import FUNDING_ASSETS

embeddings["FUNDINGXSEC"] = {
    asset: float(np.clip(your_model(asset), -1, 1))
    for asset in FUNDING_ASSETS
}
```

Positive = expect above‑median funding change. Magnitude matters (used directly as logistic regression feature).

Missing assets default to `0.0` (neutral).

### The 20 assets

```python
FUNDING_ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "SUI",
    "NEAR", "AAVE", "UNI", "LTC", "HBAR", "PEPE", "TRX", "SHIB", "TAO", "ONDO",
]
```

Fewer than the 33 in the MULTI‑BREAKOUT / XSEC‑RANK lists because not all assets have consistent perpetual funding feeds.

---

## 3. How labels are built

In `funding_xsec.py`:

1. Pull funding rates per asset (already ingested into `challenge_data` as the `price_data` JSON multi‑asset blob under ticker `FUNDINGXSEC`).
2. Compute `Δf^{(a)}_t = f^{(a)}_{t+h} - f^{(a)}_t` for horizon `h = 2400 blocks = 8h`.
3. Compute cross‑sectional median `m_t = median_a(Δf)`.
4. Label `y_{t,a} = 1[Δf^{(a)}_t > m_t]`. Pool all `(t, a)` pairs.

### Critical protections

- **Embargo**: `max(LAG, ahead)` samples between train and validation (`ahead = 2400/5 = 480 samples`). This is much larger than the binary challenge’s `LAG = 60` because the 8h forward window is longer.
- **Explicit `train_cutoff = val_start - ahead`:** prevents the 8h forward label from sliding into the training split.
- **Stale filter:** for each miner and each asset column, if the temporal standard deviation is `< 1e-4`, that column is **zeroed** before pooling. A miner that submits a constant for `ETH` but varies on `BTC` only contributes on `BTC`.

---

## 4. How validators score you

**Identical structure to `XSEC-RANK`** — see that doc for the walk‑forward details. The only differences in `funding_xsec.py`:

- Data source is `funding_rates` matrix, not `prices_multi`.
- Embargo is explicitly extended to `max(LAG, ahead)`.
- Stale filter on per‑asset columns before pooling.
- 20 assets → 20× sample multiplier (vs 33× in XSEC‑RANK).

Recipe:

1. Build `(T × 20, N_miners)` pooled feature matrix.
2. Feature‑select top‑K = 20 miners by univariate AUC.
3. For each walk‑forward segment: fit L2 logistic meta‑model; compute
   \[
   w_j^{(\text{seg})} = |\beta_j| \cdot \max\!\Big(\frac{\text{AUC}_{\text{meta}} - 0.5}{0.5},\; 0\Big)
   \]
4. Aggregate with exponential recency weighting.
5. Renormalize to sum to 1.

---

## 5. Why this design?

- **Funding rate is priced info, not price info.** It is driven by order‑book imbalance, perp premium decay, long/short liquidations — a different information set than spot returns. A good funding predictor is nearly decorrelated from a good price predictor, so high weight (`4.0`) is justified.
- **Δf not f:** breaks the autocorrelation trap (see §1 above).
- **Median subtraction:** removes market‑wide funding beta so a miner cannot coast on "everyone is positive funding today".
- **Stale filter:** specific to funding because some miners will naturally have nothing to say about some assets (low‑liquidity perps). Stale columns must be neutralized explicitly before pooling.
- **Extended embargo:** the 8h horizon is long enough that LAG=60 is insufficient; the `max(LAG, ahead)` rule is the safeguard.

---

## 6. What gets you zero weight

- Submitting a constant score per asset over time → stale filter zeros the column.
- Submitting the same score across assets → no cross‑sectional signal → univariate AUC ≈ 0.5.
- Submitting funding‑rate **levels** instead of a signed deviation score → the label is about changes, so level‑based miners underperform.
- Missing too many assets — fewer samples, more variance in AUC estimation, may fall below the `AUC > 0.5` gate.

---

## 7. Tips for miners

- **Don’t re‑use your price model.** Funding is driven by different dynamics (crowding, basis, liquidity). A dedicated model beats a shared one.
- **Useful features** (from the Miner Guide):
  - Current funding rate levels (mean‑reversion — extreme rates tend to normalize)
  - Open interest changes, long/short ratio shifts
  - Recent price momentum relative to peers
  - Liquidation volume and order‑book skew
  - Cross‑asset lead‑lag (BTC funding often leads alts by 1–2 settlement periods)
- **Cover all 20 assets.** The 20× multiplier is your biggest statistical edge.
- **Keep your submissions varied per asset.** The stale filter is ruthless (`std < 1e-4`).
- **This challenge is second‑highest weight (~17% of emissions).** Worth solving well.

---

## 8. Pointers in the code

| What | Where |
|---|---|
| Dispatch entry | `model.py` → `multi_salience()` → `loss_type == "funding_xsec"` |
| Scorer | `funding_xsec.py` → `compute_funding_xsec_salience()` |
| Label construction | same file, `Δf` + cross‑sectional median logic |
| Stale filter | same file, `temporal std < 1e-4` check |
| Embargo | same file, `train_cutoff = val_start - ahead` |
| Funding price service | `price_service.py` (OKX / HyperLiquid / CoinGlass ingestion) |
| Challenge spec | `config.FUNDING_XSEC_CHALLENGE` |
| Asset list | `config.FUNDING_ASSETS` |

---

## 9. Quick mental recap of all 7 challenges

You finished the tour. Here is a pocket reference of the full scoring taxonomy in MANTIS:

| Challenge | Dim | Horizon | Unit of observation | Label | Model stack |
|---|---:|---|---|---|---|
| Binary × 5 | 2 | 300b | time step | `r > 0` | per‑miner logistic → ElasticNet meta |
| HITFIRST | 3 | 500b | time step | which barrier hit first | 2 × L2 logistic (no meta) |
| LBFGS × 2 | 17 | 300/1800b | time step | 5‑bucket regime + Q exceedance | walk‑forward classifier (75%) + Q‑path (25%) |
| MULTI‑BREAKOUT | 2/asset | event | breakout event | continuation vs reversal | AUC gate → episode‑balanced L2 logistic |
| XSEC‑RANK | 1/asset | 1200b | (t, asset) pair | beats cross‑section median return | walk‑forward AUC‑scaled L2 meta |
| FUNDING‑XSEC | 1/asset | 2400b | (t, asset) pair | beats cross‑section median Δfunding | same as above + stale filter + big embargo |

Every challenge shares the same DNA: **coefficient‑based salience with Sybil‑resistant regularization**. What differs is *what* the label measures and *how* the features are arranged. Once you internalize that, the codebase becomes very easy to navigate.
