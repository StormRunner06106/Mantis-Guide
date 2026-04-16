# MANTIS Guide

This folder is a learning guide for the **MANTIS** project (Bittensor Subnet 123). It explains what the project does, how it works end‑to‑end, and breaks down **each challenge** into its own focused document so you can study them one at a time.

> Think of this folder as a course. Read the docs in order the first time, then jump around as needed.

---

## What is MANTIS (in one paragraph)

MANTIS is a **distributed signal marketplace** running as Bittensor Subnet 123. Miners submit **encrypted embeddings** (predictions) for a set of financial forecasting **challenges** across crypto, forex, and commodities. Validators sample those payloads every few blocks, hold them encrypted until a **time‑lock** matures, decrypt them via the Drand beacon, and then periodically score each miner by how much their embeddings **help predict future returns**. Scores across all challenges are aggregated into a single weight vector and written on‑chain, which drives TAO emissions to the best miners.

Key design properties:

- **Commit–reveal with time‑lock encryption (Drand IBE + X25519 wrap)** — no one can peek at predictions before they mature.
- **Multi‑challenge aggregation** — miners are rewarded on a weighted mix of very different prediction tasks.
- **Coefficient‑based salience** — every scorer ends in an L2 (or ElasticNet) logistic regression and uses \(|\beta_j|\) of the coefficient for miner \(j\) as their contribution.
- **Sybil resistance** — L2 regularization splits coefficient mass across correlated/duplicate miners, so cloning a top miner does not multiply rewards.
- **Streaming training** — training data is iterated per challenge from SQLite rather than loaded into memory.

---

## How to read this guide

1. Start with [`01-architecture.md`](./01-architecture.md) for the big picture (how submissions, storage, scoring, and weight setting fit together).
2. Then go through each challenge doc in order. Each doc is **self‑contained**: it explains *what* the challenge predicts, *what you submit*, *how the label is built*, *how you are scored*, and *common pitfalls*.

### Challenge docs

| # | File | Challenge(s) | Why it matters |
|---|---|---|---|
| 02 | [`02-binary-challenge.md`](./02-binary-challenge.md) | `ETH`, `CADUSD`, `NZDUSD`, `CHFUSD`, `XAGUSD` (Binary, 1h) | Simplest challenge: binary direction prediction across 5 markets. Good entry point. |
| 03 | [`03-hitfirst-challenge.md`](./03-hitfirst-challenge.md) | `ETHHITFIRST` (3‑way barrier, 500 blocks) | Predict which barrier (±1σ) gets hit first or neither. Teaches path‑dependent labeling. |
| 04 | [`04-lbfgs-challenge.md`](./04-lbfgs-challenge.md) | `ETHLBFGS` (1h), `BTCLBFGS` (6h) | 17‑dim regime + quantile paths. The most structured probabilistic challenge. |
| 05 | [`05-multi-breakout-challenge.md`](./05-multi-breakout-challenge.md) | `MULTIBREAKOUT` (33 assets, event‑driven) | Event‑based challenge: predict if a range breakout continues or reverses. |
| 06 | [`06-xsec-rank-challenge.md`](./06-xsec-rank-challenge.md) | `MULTIXSEC` (33 assets, 4h) | Cross‑sectional return ranking — who beats the median? |
| 07 | [`07-funding-xsec-challenge.md`](./07-funding-xsec-challenge.md) | `FUNDINGXSEC` (20 assets, 8h) | Cross‑sectional *funding rate* change ranking. The hardest: non‑price data. |

### Challenge weight breakdown

Each challenge has a `weight` in `config.py`. Final weight share (roughly):

| Challenge | Raw weight | Share of emissions |
|---|---:|---:|
| MULTI‑BREAKOUT | 5.00 | ~22% |
| FUNDING‑XSEC  | 4.00 | ~17% |
| ETH‑LBFGS     | 3.50 | ~15% |
| XSEC‑RANK     | 3.00 | ~13% |
| BTC‑LBFGS‑6H  | 2.88 | ~12% |
| ETH‑HITFIRST  | 2.50 | ~11% |
| 5 × Binary    | 1.00 ea | ~22% total |

Higher weight = bigger share of emissions if you do well on that challenge.

---

## Repository map (files you will encounter)

| File | Role |
|---|---|
| `config.py` | Challenge definitions, network constants, Drand parameters. **Single source of truth** for ticker names, dims, horizons, and weights. |
| `validator.py` | Main validator loop: sample blocks, collect commits, schedule decryption, trigger scoring, set weights. |
| `ledger.py` | SQLite storage, submission validation, Drand cache, training‑data generator. |
| `cycle.py` | Per‑cycle miner payload download + commit URL validation. |
| `model.py` | `multi_salience()` — dispatches each ticker to its scorer and aggregates. |
| `bucket_forecast.py` | LBFGS classifier salience + Q‑path salience. |
| `hitfirst.py` | HITFIRST barrier‑hit scoring. |
| `range_breakout.py` | MULTI‑BREAKOUT state machine and scoring. |
| `xsec_rank.py` | XSEC‑RANK scoring. |
| `funding_xsec.py` | FUNDING‑XSEC scoring. |
| `generate_and_encrypt.py` | V2 payload builder (X25519 + Drand tlock). |
| `comms.py` | Subtensor / R2 I/O helpers. |
| `autoupdate.sh`, `install_reqs.sh` | Operator scripts for validator nodes. |
| `MINER_GUIDE.md`, `lbfgs_guide.md`, `whitepaper.md`, `README.md` | Original project docs. |

---

## Minimum vocabulary

- **Hotkey / UID** — miner identifier on Bittensor.
- **Embedding** — the vector a miner submits per challenge per sample.
- **Block** — Bittensor block (~12 s). `SAMPLE_EVERY = 5 blocks` → 60 s sample cadence.
- **Sample index (`sidx`)** — monotonically increasing integer used to align embeddings with the price observed at the same sample.
- **`blocks_ahead` / horizon `H`** — number of blocks in the future the challenge predicts.
- **`LAG`** — embargo (60 samples) between train and validation to prevent leakage.
- **Salience** — per‑miner contribution score, i.e. the normalized importance of miner \(j\)’s column in the regression.
- **EMA smoothing (α = 0.15)** — applied to the aggregated weight vector to reduce block‑to‑block variance.
- **Drand IBE** — identity‑based encryption under the Drand beacon. Time‑lock: payload can only be decrypted after the committed round.

---

## Suggested study order

1. `01-architecture.md` — get the map in your head.
2. `02-binary-challenge.md` — understand the core scoring recipe (feature selection → OOS base models → ElasticNet meta‑model → \(|\beta_j|\)).
3. `03-hitfirst-challenge.md` — see a *path‑dependent* label variant.
4. `04-lbfgs-challenge.md` — see a two‑path (classifier + quantile) scorer.
5. `05-multi-breakout-challenge.md` — see an **event‑driven** (not time‑series) challenge with an AUC gate.
6. `06-xsec-rank-challenge.md` — see a **cross‑sectional** (pooled) challenge.
7. `07-funding-xsec-challenge.md` — same recipe as (6) but on a *different* dataset (funding rates).

Once you have read all of them, re‑read `01-architecture.md`; it should click much faster the second time.
