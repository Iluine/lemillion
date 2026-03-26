# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

The sole purpose of this project is to win the EuroMillions jackpot (5+2). Only the jackpot matters — lower prize tiers are irrelevant.

## Build & Test Commands

```bash
cargo build --workspace               # compile all crates
cargo test --workspace                 # run all unit tests
cargo test -p lemillion-db             # run DB crate tests
cargo test -p lemillion-cli            # run CLI crate tests
cargo test -p lemillion-ensemble       # run ensemble crate tests
cargo test -p lemillion-ensemble -- features  # run feature tests only
cargo test -p lemillion-esn            # run ESN crate tests
cargo test -p lemillion-esn -- reservoir     # run reservoir tests only
cargo run -p lemillion-cli -- import   # import CSV into SQLite
cargo run -p lemillion-cli -- list --last 5   # show last N draws
cargo run -p lemillion-cli -- stats --window 20  # frequency & gap statistics
cargo run -p lemillion-cli -- predict --seed 42  # Dirichlet prediction
cargo run -p lemillion-cli -- predict --model ewma --alpha 0.9 --seed 42  # EWMA prediction
cargo run -p lemillion-ensemble -- calibrate                  # calibrate (8 ball windows, 5 star windows)
cargo run -p lemillion-ensemble -- calibrate --windows 20,50,100  # calibrate with custom ball windows
cargo run -p lemillion-ensemble -- calibrate --pool stars --star-windows 50,100,200,500  # calibrate stars only
cargo run -p lemillion-ensemble -- calibrate --pool balls         # calibrate balls only
cargo run -p lemillion-ensemble -- weights     # show ensemble weights
cargo run -p lemillion-ensemble -- predict --seed 42  # ensemble prediction (explicit seed)
cargo run -p lemillion-ensemble -- predict              # ensemble prediction (auto-seed YYYYMMDD)
cargo run -p lemillion-ensemble -- predict --oversample 50 --min-diff 3  # custom oversampling/diversity
cargo run -p lemillion-ensemble -- history --last 5   # show recent draws
cargo run -p lemillion-ensemble -- compare 3 15 27 38 44 2 9  # analyze a grid
cargo run -p lemillion-ensemble -- add-draw 26016 MARDI 2026-02-24 10 27 40 43 47 6 10  # add a draw manually
cargo run -p lemillion-ensemble -- backtest --last 10 --suggestions 5000  # backtest ensemble
cargo run -p lemillion-ensemble -- predict --jackpot-mode --suggestions 100  # jackpot mode (top-100 by P(5+2))
cargo run -p lemillion-ensemble -- predict --jackpot-mode --suggestions 5000 --jackpot 209000000  # jackpot mode with EV calc
cargo run -p lemillion-ensemble -- predict --jackpot-mode --suggestions 5000 --no-filter  # jackpot without structural filter
cargo run -p lemillion-ensemble -- predict --n-grids 6 --suggestions 5000             # few-grid mode (6 grilles, T forcée)
cargo run -p lemillion-ensemble -- predict --n-grids 3 --suggestions 10000            # few-grid mode agressif (3 grilles)
cargo run -p lemillion-ensemble -- backtest --last 10 --suggestions 50000 --jackpot-mode  # backtest jackpot mode
cargo run -p lemillion-ensemble -- backtest --last 20 --suggestions 5000 --n-grids 6   # backtest few-grid mode
cargo run -p lemillion-ensemble -- analyze              # non-randomness statistical tests
cargo run -p lemillion-ensemble -- rebuild              # rebuild DB in chronological order
cargo run -p lemillion-ensemble -- coverage --tickets 10 --jackpot 17000000  # coverage optimization
cargo run -p lemillion-ensemble -- research                                 # full bias research report
cargo run -p lemillion-ensemble -- research --tests physical               # physical tests only
cargo run -p lemillion-ensemble -- research --window 500                   # last 500 draws only
cargo run -p lemillion-ensemble -- benchmark --train 600 --seed 42         # benchmark physics models
cargo run -p lemillion-ensemble -- optimize --n-grids 3 --last 20 --suggestions 5000 --iterations 50  # BayesOpt hyperparams
cargo run -p lemillion-ensemble -- interactive          # interactive REPL mode
cargo run -p lemillion-esn -- train                          # train ESN with defaults
cargo run -p lemillion-esn -- train --reservoir-size 500 --save esn_best.json  # train with custom params
cargo run -p lemillion-esn -- gridsearch                     # parallel grid search (~34k configs)
cargo run -p lemillion-esn -- predict                        # predict from esn_best.json
cargo run -p lemillion-esn -- predict --ensemble 5           # multi-reservoir averaging (parallel)
```

## Architecture

**Cargo workspace** (resolver 3, LTO fat + codegen-units=1 in release) with 4 crates.

**Dependency graph:** `lemillion-db` ← `lemillion-cli`, `lemillion-esn` ← `lemillion-ensemble`

### lemillion-db (shared types + DB)

- `Draw` — full draw record (id, date, 5 balls [1-50], 2 stars [1-12], winners, prize, My Million, ball_order, star_order, cycle_number)
- `Pool` — enum `Balls | Stars` with `size()`, `pick_count()`, `numbers_from(draw)`
- `Suggestion` — struct `{ balls: [u8;5], stars: [u8;2], score: f64 }`
- `db.rs` — SQLite via `rusqlite` (bundled). Single `draws` table keyed by `draw_id`. DB at `./data/lemillion.db`
- Re-exports `rusqlite` as `pub use rusqlite`

### lemillion-cli (original CLI)

Data flow: **CSV → SQLite → analysis → terminal tables**. Parses FDJ CSV (`;`-delimited, French decimals). Preserves physical extraction order (ball_order, star_order) and cycle_number.

### lemillion-esn (Echo State Network)

Standalone ESN: sparse CSR reservoir (`sprs`), leaky integrator, dual ridge regression via `faer` Cholesky. Subcommands: `train`, `gridsearch` (34k configs, parallel via `rayon`), `predict` (with `--ensemble N`). Generated files: `esn_best.json`, `esn_gridsearch.json`.

### lemillion-ensemble (ensemble forecasting)

The main crate. ~44 active models behind `trait ForecastModel` — see `models/mod.rs:base_models()` for the current list.

**Key modules:**
- `models/` — Individual forecast models (each implements `ForecastModel`)
- `ensemble/` — Calibration, weight computation, consensus, meta-predictor, stacking
- `sampler.rs` — Grid generation (marginal, joint, jackpot enumeration), temperature, diversity
- `expected_value.rs` — EV computation with popularity anti-bias
- `coverage.rs` — Multi-ticket pair/triple coverage optimization
- `research/` — Bias detection (physical, mathematical, informational, DFA, RQA)
- `features/compute.rs` — 21 features per number for feature-based models

**Production pipeline:**
1. `calibrate` → walk-forward validation per model/window → saves `calibration.json` (weights, correlations, stacking, beta-transform, temperatures, neural scorer)
2. `predict` → loads `calibration.json` + optional `hyperparams.json` → applies meta-predictor + hedge + decorrelation → generates grids
3. `backtest` → walk-forward evaluation of the full pipeline on historical draws

**Ensemble combination:** Log-linear pool (geometric: `P(x) ∝ Π P_i(x)^w_i`). Weights from softmax of (model LL − uniform LL), with decorrelation penalty and family caps.

**Key architectural concepts:**
- `SamplingStrategy` — `Consecutive` (default), `Sparse { span_multiplier }` (wider temporal coverage), `FullHistory` (walk-forward trained)
- `CoherenceScorer` — scores grids on structural similarity to historical draws (sum, spread, pair/triplet co-occurrence)
- `StructuralFilter` — rejects candidates outside historical percentile bounds
- `BallStarConditioner` — conditional star probabilities given ball context (sum_bin × spread_bin)
- `JointConditionalModel` — sequential P(b1) × P(b2|b1) × ... for scoring complete grids
- Consensus map — 2D classification (prob × spread) → StrongPick/DivisivePick/StrongAvoid/Uncertain
- Family caps — TE family capped at 20%, Stresa family at 15% (iterative, applied after hedge)

**Machine physics (v4+):** The Stresa machine uses 3 central bars + 8 external bars (mod-8 symmetry for balls). The Pâquerette uses 4 blades (mod-4 for stars). All modular models use `mod4::modulus(pool)` for pool-aware symmetry.

## Adding a New Model

1. Create `lemillion-ensemble/src/models/your_model.rs`
2. Implement `ForecastModel` trait:
   - `name()` → unique string identifier
   - `predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64>` — returns probability distribution summing to 1.0 with `pool.size()` elements
   - `params()` → hyperparameter map for display
   - Optionally override `sampling_strategy()` (use `Sparse` for models needing wider history), `calibration_stride()` (>1 for expensive models), `is_stars_only()` (true for star-dedicated models)
3. Add `pub mod your_model;` in `models/mod.rs`
4. Register in `base_models()` via `Box::new(your_model::YourModel::default())`
5. Run `cargo test -p lemillion-ensemble` → `calibrate` → `backtest` to validate

Models returning near-uniform distributions will get ~0% weight and should be retired (moved out of `base_models()`, documented in `RETIRED_MODELS.md`).

## Critical Pitfalls

- **Accent matching** — Model names like "RényiTE" (with accent é) must be matched exactly everywhere. ASCII "RenyiTE" will silently fail family caps and other name-based lookups.
- **NaN propagation** — New models can produce NaN probabilities. Always sanitize output before returning from `predict()`. NaN in one model contaminates the entire ensemble via log-linear pool.
- **Star era filtering** — Star data before 2016-09-27 uses incompatible pool sizes (11 stars, not 12). Use `filter_star_era()` for any star-specific model. Omitting this silently corrupts star calibration.
- **Extraction order availability** — `ball_order`/`star_order` are `None` for older or manually-added draws. Models relying on extraction order must handle this (return uniform when insufficient data). These models appear to underperform during calibration if the window is too small to contain order data.
- **draws[0] = most recent** — This convention holds everywhere. Walk-forward calibration trains on `draws[t+1..t+1+window]`, tests on `draws[t]`. Reversing this causes future data leakage.
- **Backtest must match production** — Any transform applied in `cmd_predict` (family caps, hedge, temperature, meta-predictor) must also be applied in backtest functions, or improvement factors will be misleading.
- **Decorrelation trade-off** — Too aggressive decorrelation (low center, high sigma) helps few-grid mode but hurts jackpot mode. Current sweet spot: center=0.60, sigma=0.6, min_weight=0.10.
- **Brier score in calibration** — Never blend Brier score with log-likelihood for weight computation. The scaling mismatch destroys all signal.
- **Temperature for large jackpots** — For jackpots >100M€, T_balls is forced to 1.0 (no sharpening) because concentration is counterproductive. T_stars remains aggressive.

## Conventions

- Error handling: `anyhow::Result` everywhere, no `unwrap()` in production code
- Tests use `Connection::open_in_memory()` for DB tests
- `Draw.balls`/`stars` are always sorted ascending; `Draw.ball_order`/`star_order` preserve physical extraction order
- `Pool::numbers_from(draw)` to extract balls/stars from a Draw
- Display uses `comfy-table` with `UTF8_FULL` preset and `Cell::fg(Color)` for emphasis
- Progress bars via `indicatif` for calibrate, backtest, and grid search
- `--alpha` flag has different semantics per model (Dirichlet: prior strength, EWMA: decay factor)
- ESN `sparsity` = fraction of zeros in reservoir matrix
- ESN `leaking_rate`: 1.0 = full update (no memory), 0.0 = no update
