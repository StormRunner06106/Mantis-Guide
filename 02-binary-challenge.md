# 02 — Binary Challenges (ETH / CADUSD / NZDUSD / CHFUSD / XAGUSD, 1h)

Five parallel binary‑direction challenges. The simplest and best entry point to understand the MANTIS scoring recipe, because all the more complex challenges are variations of this pattern.

| Property | Value |
|---|---|
| **Tickers** | `ETH`, `CADUSD`, `NZDUSD`, `CHFUSD`, `XAGUSD` |
| **Loss function key** | `binary` |
| **Embedding dim** | `2` |
| **Horizon (`blocks_ahead`)** | `300` blocks ≈ 1 hour |
| **Challenge weight** (each) | `1.0` (5 × 1.0 total) |
| **Scorer** | `salience_binary_prediction()` in `model.py` |

---

## 1. What you are predicting

**Direction of the next 1‑hour return** on that ticker:

\[
y_t = \mathbf{1}\![\, r_{t \to t+H} > 0 \,], \qquad H = 300 \text{ blocks}
\]

A price went up → label `1`. Went down → label `0`. Ties are treated as zero (via `RET_EPS = 0.0`).

Note: `y` must have both classes present in the window, otherwise the scorer returns no salience.

---

## 2. What you submit

A **2‑dimensional vector** per ticker, every sample. Each component must be in \([-1, 1]\).

```python
embeddings["ETH"] = [0.3, -0.1]        # your 2 features
embeddings["CADUSD"] = [-0.5, 0.2]
embeddings["NZDUSD"] = [0.0, 0.0]
embeddings["CHFUSD"] = [0.1, 0.1]
embeddings["XAGUSD"] = [-0.2, 0.4]
```

These are **features**, not probabilities. They are fed straight into a per‑miner L2 logistic regression. They do not need to sum to anything.

---

## 3. How validators score you

The full recipe is implemented in `salience_binary_prediction()` in `model.py`. It is the **template** for every other challenge in the repo.

### Step 1 — Assemble the matrix

- `X_flat` has shape `(T, H * D)` where `T` = samples, `H` = hotkeys, `D = 2`.
- Reshape to `X[t, j, :]` — row `t` sample, miner `j`’s 2‑dim feature vector.
- Labels `y = 1[r_{t→t+H} > 0]`.

A hard cap `MAX_INDEX_HISTORY = MAX_DAYS * INDICES_PER_DAY` is applied so the window is at most 60 days.
Minimum `T = 500` samples required.

### Step 2 — Feature selection (top‑K by AUC)

For each miner `j` independently:

1. Split `T` in half. Fit an L2 logistic regression on the first half `X[:T/2, j, :]`.
2. Score with `decision_function` on the second half.
3. Record that miner’s held‑out AUC.

Then **keep the top `TOP_K = 50`** miners by AUC whose AUC > 0.5. Everyone else is dropped for this challenge.

### Step 3 — Walk‑forward OOS base‑model predictions

Build OOS prediction column `X_oos[:, col(j)]` for each selected miner `j`:

1. Split the timeline into segments of length `CHUNK_SIZE = 4000`.
2. For each segment `[seg_start, seg_val_start, seg_val_end]`:
   - Fit a **per‑miner L2 logistic** on all data up to `seg_val_start - LAG` (the **embargo**).
   - Predict on `[seg_val_start, seg_val_end]`.
3. Store those out‑of‑sample predictions in `X_oos` (column = the miner).

Rows where no miner produced an OOS value are NaN‑filled and later ignored.

### Step 4 — Meta‑model (ElasticNet logistic)

A single **ElasticNet logistic** meta‑model is fit across all miners' OOS columns simultaneously:

- Penalty: `elasticnet`, `l1_ratio = 0.5`, `C = 1.0`, `saga` solver.
- Class weight: `balanced`.
- Sample weight: **exponential time decay** with half‑life `HALFLIFE_DAYS = 15` (recent samples weigh more).

### Step 5 — Salience

Per‑miner importance:

\[
\text{imp}_j = |\beta_j|
\]

Normalize across miners so `∑ imp_j = 1`, and that dict is the binary‑challenge salience for that ticker.

---

## 4. Why this recipe?

- **Feature selection first** → prevents one bad miner’s features from drowning out good ones in the meta‑model.
- **Walk‑forward OOS** → each miner’s meta‑model feature is a true out‑of‑sample prediction; leakage through the horizon is bounded by `LAG = 60`.
- **ElasticNet (L1 + L2)** → the L1 piece zeros out uninformative miners; the L2 piece splits coefficient mass among correlated/duplicated miners (the Sybil defense).
- **Recency weighting** → the market is non‑stationary; recent samples matter more.
- **\(|\beta_j|\) as salience** → a direct measure of "how much this miner’s prediction moves the meta‑model decision".

---

## 5. What gets you zero weight

- Submitting constants (e.g. `[0, 0]` forever) → feature‑selection AUC stays at 0.5 → dropped.
- Submitting random noise → AUC near 0.5 → dropped, or β near 0.
- Copying a top miner → correlated columns → L2 splits the coefficient mass; the clone cannot *add* weight.
- Missing the ticker entirely → zero contribution for that challenge.

---

## 6. Tips for miners

- Start with a simple logistic‑regression‑ready signal: e.g. a normalized momentum feature and a normalized volatility feature. Two *decorrelated* features are much better than two highly correlated ones.
- Clip your features to `[-1, 1]` (validators do not care about the scale, but the L2 penalty works better when the features are roughly standardized).
- Update every sample (`SAMPLE_EVERY = 5` blocks = 60 s). If your embedding does not change across samples, the scorer sees a nearly constant feature and your OOS AUC collapses.
- Hitting the **5 binary challenges** is a cheap way to cover ~22% of emissions in total.

---

## 7. Pointers in the code

| What | Where |
|---|---|
| Dispatch entry | `model.py` → `multi_salience()` → `loss_type == "binary"` |
| Scorer | `model.py` → `salience_binary_prediction()` |
| Feature selection | same function, `# --- Feature selection` block |
| Walk‑forward segments | `_build_oos_segments()` in `model.py` |
| Base logistic | `_fit_base_logistic()` in `model.py` |
| Meta ElasticNet | `_fit_meta_logistic_en()` in `model.py` |
| Challenge spec | `config.CHALLENGES` entries with `loss_func == "binary"` |
