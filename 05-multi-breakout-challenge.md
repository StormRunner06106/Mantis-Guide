# 05 — MULTI‑BREAKOUT Challenge (`MULTIBREAKOUT`, 33 assets)

An **event‑driven** challenge (not a time‑series challenge). A state machine tracks rolling 4‑day ranges for 33 assets. When price breaches a barrier, a *breakout event* is triggered and the miner's prediction at that instant is locked in as the sample. Predict whether the breakout **continues** or **reverses**.

| Property | Value |
|---|---|
| **Ticker** | `MULTIBREAKOUT` |
| **Loss function key** | `range_breakout_multi` |
| **Embedding dim (per asset)** | `2` — `[P_continuation, P_reversal]` |
| **Assets** | 33 (see below) |
| **Trigger** | price breaches 25% of the rolling 4‑day range |
| **Challenge weight** | `5.0` (highest) |
| **Scorer** | `compute_multi_breakout_salience()` in `range_breakout.py` |

---

## 1. What you are predicting

At the **instant** a breakout event triggers for asset `a`:
- Predict whether the price continues in the breakout direction (`continuation`) or reverses back into the range (`reversal`), within some follow‑up window defined by the state machine.

Labels come out of a per‑asset state machine (`breakout_state` table in SQLite) — see `range_breakout.py` for the state logic.

**Key fact:** only the sample at the trigger matters for scoring, but you must keep submitting continuously so that your embedding happens to be *present* when the event fires.

---

## 2. What you submit

A **dict keyed by asset**, each value a 2‑element probability pair in `(0, 1)`:

```python
from config import BREAKOUT_ASSETS

embeddings["MULTIBREAKOUT"] = {
    asset: [float(np.clip(p_cont, 0.01, 0.99)),
            float(np.clip(1 - p_cont, 0.01, 0.99))]
    for asset in BREAKOUT_ASSETS
}
```

### Hard rules

- Keys are the 33 assets below; missing assets = neutral for that asset.
- Each value must be `[P_continuation, P_reversal]` with both components in `(0, 1)`.
- Validators clip to `(EPS, 1−EPS)` but submit clean values.

### The 33 assets

```python
BREAKOUT_ASSETS = [
    "BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "XMR",
    "LINK", "LEO", "HYPE", "XLM", "ZEC", "SUI", "LTC", "AVAX", "HBAR", "SHIB",
    "TON", "CRO", "DOT", "UNI", "MNT", "BGB", "TAO", "AAVE", "PEPE",
    "NEAR", "ICP", "ETC", "ONDO", "SKY",
]
```

---

## 3. How events trigger (state machine)

Implemented in `range_breakout.py`. For each asset `a`:

1. Maintain a rolling 4‑day price range. `range_lookback_blocks = 28800` (4 × 7200 blocks/day).
2. Compute `high − low` of that range. If the range width is below `min_range_pct = 1%` of price, **skip** — ranges that tight are noise.
3. Compute an upper and lower **barrier** at `25%` of the range width outside the range (or similarly — see `range_breakout.py` for exact anchoring).
4. When price crosses a barrier, emit a breakout event with metadata `(asset, trigger_sidx, direction, follow_up_window)`.
5. At the end of the follow‑up window, label:
   - `1` if the move continued beyond the barrier,
   - `0` if it reversed back into the range.

State is persisted per asset in the `breakout_state` SQLite table so the detector survives validator restarts.

---

## 4. How validators score you — two‑stage scoring

Events (not timesteps) are the unit of observation. Let `N_events` = total completed events across all assets.

### Stage 1 — Empirical AUC gate

Per miner, compute per‑miner **AUC of `P_continuation` vs realized label** across all their submissions on completed events.

A miner qualifies iff **all** of the following:

- AUC `> 0.5`
- At least **2 temporal episodes** submitted (a temporal episode ≈ a distinct time window where events were collected — prevents a miner from getting through on one lucky event)
- Prediction std `> 0.03` across their events (filter constants)

Miners that fail the gate are dropped.

### Stage 2 — L2 logistic with episode‑balanced weighting

- Build a feature matrix where each column is a qualifying miner's **z‑scored** `P_continuation` predictions across events.
- Compute **sample weights** so that each temporal episode gets equal total weight regardless of how many events it contains — this prevents a rare high‑event episode (e.g. a volatile weekend) from dominating training.
- Fit an L2 logistic regression `y = 1[continuation happened]` with those weights.
- Per‑miner importance:
  \[
  \text{imp}_j = |\beta_j|
  \]
- Normalize across miners to sum to 1.

---

## 5. Why this design?

- **Event‑driven, not time‑series:** most of the time nothing interesting is happening. Continuously scoring miners on "normal" samples would drown out the breakout signal. Restricting to events *multiplies* the effective signal.
- **AUC gate first:** cheap pre‑filter that removes garbage miners before they enter the regression. Saves compute and reduces multicollinearity.
- **Std gate (0.03):** prevents a miner who always submits 0.5 from being a "diversifier" that gets any weight.
- **Episode‑balanced weighting:** breakouts cluster in time. Without balancing, a single regime (e.g. a BTC trending week) would dominate the regression.
- **z‑scoring predictions:** normalizes miners with different decision thresholds onto a common axis.
- **Highest weight (5.0):** the subnet designer intentionally prices this challenge as the most important — breakouts are rare but information‑dense.

---

## 6. What gets you zero weight

- Submitting only constants or near‑constants (filtered by std gate).
- AUC ≤ 0.5 across completed events (filtered by AUC gate).
- Submitting only for some assets when events happen on the ones you missed.
- Not submitting at the right moment — your prediction at the *trigger sidx* is what is frozen in; you must submit every sample to be available when the event fires.

---

## 7. Tips for miners

- **Continuous submission is mandatory.** If your inference pipeline is batched and slow, you will miss events. Keep the round‑trip under the 60 s sample cadence.
- **Cover all 33 assets.** Missing half the assets halves your event count and cripples your AUC power.
- **Events are rare:** ~1–5 per asset per day. Expect to accumulate statistically meaningful AUC only after a few weeks.
- **Model the state machine yourself.** Since the trigger logic (barrier_pct = 25%, lookback = 4d) is fully specified, you can detect "imminent" breakouts locally and switch to a breakout‑specific sub‑model.
- **Don’t game by flipping near 0.5.** The std gate filters you; the L2 penalty crushes uninformative features.

---

## 8. Pointers in the code

| What | Where |
|---|---|
| Dispatch entry | `model.py` → `multi_salience()` → `loss_type == "range_breakout_multi"` |
| State machine | `range_breakout.py` → `RangeBreakoutTracker` / similar classes |
| Scorer | `range_breakout.py` → `compute_multi_breakout_salience()` |
| State persistence | `ledger.py` → `breakout_state` table |
| Challenge spec | `config.MULTI_BREAKOUT_CHALLENGE` |
| Asset list | `config.BREAKOUT_ASSETS` |
