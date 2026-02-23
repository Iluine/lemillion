# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
cargo run -p lemillion-ensemble -- calibrate --windows 20,50,100  # calibrate ensemble models
cargo run -p lemillion-ensemble -- weights     # show ensemble weights
cargo run -p lemillion-ensemble -- predict --seed 42  # ensemble prediction (explicit seed)
cargo run -p lemillion-ensemble -- predict              # ensemble prediction (auto-seed YYYYMMDD)
cargo run -p lemillion-ensemble -- predict --oversample 50 --min-diff 3  # custom oversampling/diversity
cargo run -p lemillion-ensemble -- history --last 5   # show recent draws
cargo run -p lemillion-ensemble -- compare 3 15 27 38 44 2 9  # analyze a grid
cargo run -p lemillion-ensemble -- interactive              # interactive REPL mode
cargo run -p lemillion-esn -- train                          # train ESN with defaults
cargo run -p lemillion-esn -- train --reservoir-size 500 --save esn_best.json  # train with custom params
cargo run -p lemillion-esn -- gridsearch                     # parallel grid search (~34k configs)
cargo run -p lemillion-esn -- predict                        # predict from esn_best.json
cargo run -p lemillion-esn -- predict --ensemble 5           # multi-reservoir averaging (parallel)
```

## Architecture

**Cargo workspace** with 4 crates analyzing EuroMillions lottery draws via Bayesian and ML models.

### Workspace structure

```
lemillion/                          (workspace root)
  lemillion-db/                    (lib crate - shared types + DB)
    src/lib.rs, models.rs, db.rs
  lemillion-cli/                   (bin crate - original CLI)
    src/main.rs, import.rs, display.rs, analysis/{mod,dirichlet,ewma,sampler}.rs
  lemillion-esn/                   (bin+lib crate - Echo State Network)
    src/main.rs, lib.rs, config.rs, encoding.rs, reservoir.rs, training.rs
    src/linalg.rs, metrics.rs, gridsearch.rs, display.rs
  lemillion-ensemble/              (bin+lib crate - ensemble forecasting)
    src/main.rs, lib.rs, display.rs, sampler.rs, interactive.rs
    src/models/{mod,dirichlet,ewma,logistic,random_forest,markov,retard,hot_streak,esn}.rs
    src/features/{mod,compute}.rs
    src/ensemble/{mod,calibration,consensus}.rs
```

**Dependency graph:** `lemillion-db` ← `lemillion-cli`, `lemillion-esn` ← `lemillion-ensemble`

### lemillion-db (shared types)

- `Draw` — full draw record (id, date, 5 balls [1-50], 2 stars [1-12], winners, prize, My Million)
- `Pool` — enum `Balls | Stars` with `size()`, `pick_count()`, `numbers_from(draw)`
- `Suggestion` — struct `{ balls: [u8;5], stars: [u8;2], score: f64 }`
- `validate_draw` — ensures ranges and no duplicates
- `db.rs` — SQLite via `rusqlite` (bundled). Single `draws` table keyed by `draw_id`. `INSERT OR IGNORE` handles duplicates. DB at `./data/lemillion.db`

### lemillion-cli (original CLI)

Data flows: **CSV -> SQLite -> analysis -> terminal tables**.

- `import.rs` — Parses the FDJ CSV (`;`-delimited, French decimals, `flexible(true)`). Dates DD/MM/YYYY to YYYY-MM-DD.
- `analysis/` — Models operate on `Vec<([u8;5], [u8;2])>` (index 0 = most recent)
- `PredictionModel` — local `clap::ValueEnum` enum (`Dirichlet` | `Ewma`)

### lemillion-esn (Echo State Network)

Standalone ESN implementation with sparse reservoir, zero-alloc step, and dual ridge regression via `faer` Cholesky.

**Key types:**
- `EsnConfig` — all hyperparameters (reservoir_size, spectral_radius, sparsity, leaking_rate, ridge_lambda, input_scaling, encoding, washout, noise_amplitude, seed). Impl `Default + Clone + Serialize/Deserialize`.
- `Encoding` — enum `OneHot` (62-dim: 50 balls + 12 stars) | `Normalized` (7-dim: sorted values/max)
- `Reservoir` — sparse CSR matrix (`sprs`), leaky integrator: `h(t) = (1-a)*h(t-1) + a*tanh(W_in*x + W_res*h(t-1)) + noise`
- `TrainedEsn` — trained reservoir + readout weights `w_out_balls`, `w_out_stars`
- `DataSplit` — 80% train / 8% val / 12% test split (reverses draws to chronological internally)

**Key functions:**
- `train_and_evaluate(draws, config)` → `(TrainedEsn, EsnResult)` — full train/val/test pipeline
- `predict_next(esn, draws)` → `(ball_probs[50], star_probs[12])` — run reservoir on all draws, predict from last state
- `ridge_regression(H, Y, lambda)` — auto-selects primal (d×d) or dual (T×T) path via `faer` Cholesky
- `generate_grid()` — 34,560 hyperparameter configurations
- `run_grid_search()` — parallel eval via `rayon`, saves only if score improves

**CLI subcommands:** `train`, `gridsearch`, `predict` (with `--ensemble N` for multi-reservoir averaging)

**Generated files:** `esn_best.json` (best config), `esn_gridsearch.json` (full results)

### lemillion-ensemble (ensemble forecasting)

8 independent models behind `trait ForecastModel` (takes `&[Draw]`, returns `Vec<f64>` summing to 1.0):

1. **Dirichlet** — Dirichlet-Multinomial prior
2. **EWMA** — Exponentially Weighted Moving Average
3. **Logistic** — SGD with L2 regularization via ndarray, 14 features
4. **RandomForest** — 50 trees, depth 5, bootstrap, sqrt(n) features, Gini impurity
5. **Markov** — Transition matrices by ranges (5x10 balls, 3x4 stars) + frequency redistribution
6. **Retard** — `score = (gap/mean_gap)^gamma`, default gamma=1.5
7. **HotStreak** — Linear decreasing weights over K recent draws
8. **ESN** — Echo State Network wrapper; dynamically adjusts washout for small windows, uniform fallback on error

**Feature engineering** (`features/compute.rs`): 14 features per number (freq_5/10/20, retard, retard_norm, trend, mean_gap, std_gap, is_odd, decade, decade_density, day_of_week, recent_sum_norm, recent_even_count).

**Sampler** (`sampler.rs`):
- `date_seed()` — deterministic YYYYMMDD seed from local date (via chrono), used when no `--seed` provided
- `optimal_grid(ball_probs, star_probs)` — deterministic grid: top 5 balls + top 2 stars by ensemble probability, score = product of (prob/uniform)
- Oversampling: generates `count × oversample` candidates (default 20×), keeps top scores
- Diversity: greedy selection enforcing `min_ball_diff` (default 2) differing balls between any pair of suggestions
- Signature: `generate_suggestions_from_probs(ball_probs, star_probs, count, seed: u64, oversample, min_ball_diff)`

**Ensemble** (`ensemble/`):
- Walk-forward validation (NO future data leakage): train on `draws[t+1..t+1+window]`, test on `draws[t]`
- Stride sampling (~100 test points) for calibration performance
- Weights: `exp(best_ll - uniform_ll)`, normalized to sum to 1.0
- Calibration results saved/loaded as JSON (`calibration.json`)

**Consensus** (`ensemble/consensus.rs`):
- `build_consensus_map` — 2D classification (prob × median_spread) -> `StrongPick | DivisivePick | StrongAvoid | Uncertain`
- `consensus_score(balls, stars, ball_consensus, star_consensus)` — scores a grid against the consensus map: StrongPick=+2, DivisivePick=+1, Uncertain=0, StrongAvoid=-1. Range: -7 to +14

**Prediction pipeline** (`cmd_predict` in `main.rs`):
1. Load calibration weights (or uniform fallback)
2. Predict ball/star distributions via `EnsembleCombiner`
3. Display top-N distributions + consensus maps
4. Display optimal grid (deterministic, max probability)
5. Generate sampled suggestions, sort by consensus score (descending, stable on bayesian score), display with consensus column

**Interactive mode** (`interactive.rs`):
- REPL loop with menu: add draw, calibrate, predict, history, compare, weights, quit
- Commands by number (`1`-`7`), French name (`ajouter`), or alias (`add`, `q`, `cal`, `pred`, `hist`, `comp`)
- Each command prompts for parameters with defaults; errors are caught without exiting the loop

### Conventions

- Error handling: `anyhow::Result` everywhere, no `unwrap()` in production code
- Tests use `Connection::open_in_memory()` for DB tests
- The `--alpha` flag has different semantics per model (Dirichlet: prior strength, EWMA: decay factor 0<alpha<1)
- `draws[0]` = most recent draw in all contexts
- `Pool::numbers_from(draw)` to extract balls/stars from a Draw
- Display uses `comfy-table` with `UTF8_FULL` preset and `Cell::fg(Color)` for emphasis
- ESN `sparsity` = fraction of zeros in reservoir matrix
- ESN `leaking_rate`: 1.0 = full update (no memory), 0.0 = no update
