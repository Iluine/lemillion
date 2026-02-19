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
```

## Architecture

**Cargo workspace** with 3 crates analyzing EuroMillions lottery draws via Bayesian and ML models.

### Workspace structure

```
lemillion/                          (workspace root)
  lemillion-db/                    (lib crate - shared types + DB)
    src/lib.rs, models.rs, db.rs
  lemillion-cli/                   (bin crate - original CLI)
    src/main.rs, import.rs, display.rs, analysis/{mod,dirichlet,ewma,sampler}.rs
  lemillion-ensemble/              (bin+lib crate - ensemble forecasting)
    src/main.rs, lib.rs, display.rs, sampler.rs
    src/models/{mod,dirichlet,ewma,logistic,random_forest,markov,retard,hot_streak}.rs
    src/features/{mod,compute}.rs
    src/ensemble/{mod,calibration,consensus}.rs
```

### lemillion-db (shared types)

- `Draw` — full draw record (id, date, 5 balls [1-50], 2 stars [1-12], winners, prize, My Million)
- `Pool` — enum `Balls | Stars` with `size()`, `pick_count()`, `numbers_from(draw)`
- `validate_draw` — ensures ranges and no duplicates
- `db.rs` — SQLite via `rusqlite` (bundled). Single `draws` table keyed by `draw_id`. `INSERT OR IGNORE` handles duplicates. DB at `./data/lemillion.db`

### lemillion-cli (original CLI)

Data flows: **CSV -> SQLite -> analysis -> terminal tables**.

- `import.rs` — Parses the FDJ CSV (`;`-delimited, French decimals, `flexible(true)`). Dates DD/MM/YYYY to YYYY-MM-DD.
- `analysis/` — Models operate on `Vec<([u8;5], [u8;2])>` (index 0 = most recent)
- `PredictionModel` — local `clap::ValueEnum` enum (`Dirichlet` | `Ewma`)

### lemillion-ensemble (ensemble forecasting)

7 independent models behind `trait ForecastModel` (takes `&[Draw]`, returns `Vec<f64>` summing to 1.0):

1. **Dirichlet** — Dirichlet-Multinomial prior
2. **EWMA** — Exponentially Weighted Moving Average
3. **Logistic** — SGD with L2 regularization via ndarray, 14 features
4. **RandomForest** — 50 trees, depth 5, bootstrap, sqrt(n) features, Gini impurity
5. **Markov** — Transition matrices by ranges (5x10 balls, 3x4 stars) + frequency redistribution
6. **Retard** — `score = (gap/mean_gap)^gamma`, default gamma=1.5
7. **HotStreak** — Linear decreasing weights over K recent draws

**Feature engineering** (`features/compute.rs`): 14 features per number (freq_5/10/20, retard, retard_norm, trend, mean_gap, std_gap, is_odd, decade, decade_density, day_of_week, recent_sum_norm, recent_even_count).

**Sampler** (`sampler.rs`):
- `date_seed()` — deterministic YYYYMMDD seed from local date (via chrono), used when no `--seed` provided
- Oversampling: generates `count × oversample` candidates (default 20×), keeps top scores
- Diversity: greedy selection enforcing `min_ball_diff` (default 2) differing balls between any pair of suggestions
- Signature: `generate_suggestions_from_probs(ball_probs, star_probs, count, seed: u64, oversample, min_ball_diff)`

**Ensemble** (`ensemble/`):
- Walk-forward validation (NO future data leakage): train on `draws[t+1..t+1+window]`, test on `draws[t]`
- Stride sampling (~100 test points) for calibration performance
- Models below uniform log-likelihood get weight 0
- Consensus map: 2D classification (prob x spread) -> StrongPick/DivisivePick/StrongAvoid/Uncertain
- Calibration results saved/loaded as JSON

### Conventions

- Error handling: `anyhow::Result` everywhere, no `unwrap()` in production code
- Tests use `Connection::open_in_memory()` for DB tests
- The `--alpha` flag has different semantics per model (Dirichlet: prior strength, EWMA: decay factor 0<alpha<1)
- `draws[0]` = most recent draw in all contexts
- `Pool::numbers_from(draw)` to extract balls/stars from a Draw
