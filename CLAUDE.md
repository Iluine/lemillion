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
cargo run -p lemillion-ensemble -- calibrate                  # calibrate (7 default windows)
cargo run -p lemillion-ensemble -- calibrate --windows 20,50,100  # calibrate with custom windows
cargo run -p lemillion-ensemble -- weights     # show ensemble weights
cargo run -p lemillion-ensemble -- predict --seed 42  # ensemble prediction (explicit seed)
cargo run -p lemillion-ensemble -- predict              # ensemble prediction (auto-seed YYYYMMDD)
cargo run -p lemillion-ensemble -- predict --oversample 50 --min-diff 3  # custom oversampling/diversity
cargo run -p lemillion-ensemble -- history --last 5   # show recent draws
cargo run -p lemillion-ensemble -- compare 3 15 27 38 44 2 9  # analyze a grid
cargo run -p lemillion-ensemble -- add-draw 26016 MARDI 2026-02-24 10 27 40 43 47 6 10  # add a draw manually
cargo run -p lemillion-ensemble -- backtest --last 10 --suggestions 5000  # backtest ensemble
cargo run -p lemillion-ensemble -- analyze              # non-randomness statistical tests
cargo run -p lemillion-ensemble -- rebuild              # rebuild DB in chronological order
cargo run -p lemillion-ensemble -- interactive          # interactive REPL mode
cargo run -p lemillion-esn -- train                          # train ESN with defaults
cargo run -p lemillion-esn -- train --reservoir-size 500 --save esn_best.json  # train with custom params
cargo run -p lemillion-esn -- gridsearch                     # parallel grid search (~34k configs)
cargo run -p lemillion-esn -- predict                        # predict from esn_best.json
cargo run -p lemillion-esn -- predict --ensemble 5           # multi-reservoir averaging (parallel)
```

## Architecture

**Cargo workspace** (resolver 3) with 4 crates analyzing EuroMillions lottery draws via Bayesian and ML models.

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
    src/main.rs, lib.rs, display.rs, sampler.rs, interactive.rs, analysis.rs
    src/models/{mod,dirichlet,ewma,logistic,random_forest,markov,retard,hot_streak,esn,takens,spectral,ctw,nvar,nvar_memo,mixture}.rs
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
- `delete_draw(conn, draw_id)` — deletes a draw by ID, returns `bool`
- `fetch_last_draws_numbers(conn, limit)` — lighter query returning only `Vec<([u8;5], [u8;2])>`
- Re-exports `rusqlite` as `pub use rusqlite` (used by ensemble crate for `Connection` access)

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

14 independent models behind `trait ForecastModel` (takes `&[Draw]`, returns `Vec<f64>` summing to 1.0):

1. **Dirichlet** — Dirichlet-Multinomial prior
2. **EWMA** — Exponentially Weighted Moving Average
3. **Logistic** — SGD with L2 regularization via ndarray, 14 features
4. **RandomForest** — 50 trees, depth 5, bootstrap, sqrt(n) features, Gini impurity
5. **Markov** — Transition matrices by ranges (5x10 balls, 3x4 stars) + frequency redistribution
6. **Retard** — `score = (gap/mean_gap)^gamma`, default gamma=1.5
7. **HotStreak** — Linear decreasing weights over K recent draws
8. **ESN** — Echo State Network wrapper; dynamically adjusts washout for small windows, uniform fallback on error
9. **TakensKNN** — Phase-space reconstruction (Takens embedding theorem), K-nearest-neighbor in embedded space (k=5, tau=1, dim=3). Encodes draws as scalar vectors, weights successor draws by inverse distance. 70/30 mix with uniform
10. **Spectral** — FFT via `rustfft` on binary presence/absence series per number, identifies dominant harmonics, autocorrelation extrapolation (n_harmonics=5, smoothing=0.7, min 30 draws)
11. **CTW** — Context Tree Weighting: Bayesian universal predictor with Krichevsky-Trofimov estimator, depth-6 context tree over binary presence/absence series per number. Analytically computes predictive probability by traversing context path (O(D) per prediction). Theoretically optimal for finite-memory sources (depth=6, smoothing=0.5, min 10 draws)
12. **NVAR** — Nonlinear Vector Autoregression (Gauthier et al. 2021): deterministic replacement for ESN. Delay embedding (d=5) of summary statistics (sum, spread, parity, centroid, variance) + quadratic cross-products, ridge regression via `lemillion-esn::linalg::ridge_regression`. (poly_degree=2, ridge_lambda=1e-4, smoothing=0.6)
13. **NVAR-Memo** — Random Fourier Features (Rahimi & Recht 2007) approximating infinite-dimensional RBF kernel. Overparameterized (200 features > N samples) ridge regression for memorization/interpolation. Tests whether RBF kernel extrapolation beats uniform. (n_features=200, bandwidth=1.0, ridge_lambda=1e-6, delay=3, smoothing=0.5, seed=42)
14. **BME** — Bayesian Mixture of Experts: 6 lightweight experts (frequency, gap, parity, decade balance, sum target, co-occurrence) reweighted online via Hedge algorithm (multiplicative weights update). Proven regret bound: loss ≤ best_expert + sqrt(T×ln(K)). (learning_rate=0.1, smoothing=0.3)

**Feature engineering** (`features/compute.rs`): 18 features per number — freq_3, freq_5, freq_10, freq_20, retard, retard_norm, trend, mean_gap, std_gap, is_odd, decade, decade_density, day_of_week, recent_sum_norm, recent_even_count, pair_freq, gap_acceleration, low_half.

**Analysis** (`analysis.rs`): 5 non-randomness statistical tests — permutation entropy, runs test (Wald-Wolfowitz), auto-mutual information (AMI lag 1-20), correlation dimension (Grassberger-Procaccia, dims 3/5/7), Lempel-Ziv complexity. Each returns `AnalysisResult { test_name, value, expected_random, verdict: Signal|Neutral|Random, detail }`.

**Sampler** (`sampler.rs`):
- `date_seed()` — deterministic YYYYMMDD seed from local date (via chrono), used when no `--seed` provided
- `optimal_grid(ball_probs, star_probs)` — deterministic grid: top 5 balls + top 2 stars by ensemble probability, score = product of (prob/uniform)
- `generate_suggestions_from_probs(...)` — thin wrapper around `generate_suggestions_filtered`
- `generate_suggestions_filtered(...)` — oversampling + diversity with optional `StructuralFilter`
- `generate_suggestions_joint(...)` — **primary production path**. Two-phase: 50% marginal sampling scored with `CoherenceScorer`, 50% template recombination from top-20 historical draws. Score = `bayesian_score * (0.5 + coherence_score)`
- `StructuralFilter` — rejects candidates outside historical percentile bounds for ball sum, max consecutive run, odd count. Built via `StructuralFilter::from_history(draws, pool)`
- `CoherenceScorer` — computes historical sum/spread stats and pair co-occurrence frequencies
- `compute_bayesian_score(balls, stars, ball_probs, star_probs)` — standalone scoring function
- Diversity: greedy selection enforcing `min_ball_diff` (default 2) differing balls between any pair

**Ensemble** (`ensemble/`):
- Walk-forward validation (NO future data leakage): train on `draws[t+1..t+1+window]`, test on `draws[t]`
- Stride sampling (~100 test points) for calibration performance
- Weights: `exp(best_ll - uniform_ll)`, normalized to sum to 1.0
- Default windows: `20,30,40,50,60,80,100` (7 windows)
- Calibration results saved/loaded as JSON (`calibration.json`)

**Consensus** (`ensemble/consensus.rs`):
- `build_consensus_map` — 2D classification (prob × median_spread) -> `StrongPick | DivisivePick | StrongAvoid | Uncertain`
- `consensus_score(balls, stars, ball_consensus, star_consensus)` — scores a grid against the consensus map: StrongPick=+2, DivisivePick=+1, Uncertain=0, StrongAvoid=-1. Range: -7 to +14

**Prediction pipeline** (`cmd_predict` in `main.rs`):
1. Load calibration weights (or uniform fallback)
2. Predict ball/star distributions via `EnsembleCombiner`
3. Display top-N distributions + consensus maps
4. Display optimal grid (deterministic, max probability)
5. Generate sampled suggestions via `generate_suggestions_joint`, sort by consensus score (descending, stable on bayesian score), display with consensus column

**CLI subcommands:** `calibrate`, `weights`, `predict`, `history`, `compare`, `add-draw`, `fix-draw`, `rebuild`, `backtest`, `analyze`, `interactive`

**Interactive mode** (`interactive.rs`):
- REPL loop with menu: add draw, calibrate, predict, history, compare, weights, analyze, quit
- Commands by number (`1`-`8`), French name (`ajouter`, `analyser`), or alias (`add`, `q`, `cal`, `pred`, `hist`, `comp`, `ana`)
- Each command prompts for parameters with defaults; errors are caught without exiting the loop

### Conventions

- Error handling: `anyhow::Result` everywhere, no `unwrap()` in production code
- Tests use `Connection::open_in_memory()` for DB tests
- The `--alpha` flag has different semantics per model (Dirichlet: prior strength, EWMA: decay factor 0<alpha<1)
- `draws[0]` = most recent draw in all contexts
- `Pool::numbers_from(draw)` to extract balls/stars from a Draw
- Display uses `comfy-table` with `UTF8_FULL` preset and `Cell::fg(Color)` for emphasis
- Progress bars via `indicatif` for calibrate, backtest, and grid search
- ESN `sparsity` = fraction of zeros in reservoir matrix
- ESN `leaking_rate`: 1.0 = full update (no memory), 0.0 = no update
