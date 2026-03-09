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
cargo run -p lemillion-ensemble -- interactive          # interactive REPL mode
cargo run -p lemillion-esn -- train                          # train ESN with defaults
cargo run -p lemillion-esn -- train --reservoir-size 500 --save esn_best.json  # train with custom params
cargo run -p lemillion-esn -- gridsearch                     # parallel grid search (~34k configs)
cargo run -p lemillion-esn -- predict                        # predict from esn_best.json
cargo run -p lemillion-esn -- predict --ensemble 5           # multi-reservoir averaging (parallel)
```

## Architecture

**Cargo workspace** (resolver 3, edition 2024) with 4 crates analyzing EuroMillions lottery draws via Bayesian and ML models.

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
    src/main.rs, lib.rs, display.rs, sampler.rs, interactive.rs, analysis.rs, coverage.rs, expected_value.rs
    src/models/{mod,dirichlet,logistic,random_forest,markov,esn,spectral,ctw,mixture,transformer,tda,physics,mod4,mod4_profile,triplet,conditional,conditional_v2,gap_dynamics,joint,summary_predictor,star_specialist,stresa,transfer_entropy,star_pair,star_recency,context_knn,max_entropy,neural_scorer,jackpot_context,hmm,boltzmann,hawkes,bocpd,decade_persist,modular_balls,compression,star_momentum,spread,gap_model,unit_digit,delayed_mi,community,rqa_predictability,copula,wavelet,renewal,draw_order,tlr,particle_stresa,forbidden_patterns}.rs
    src/features/{mod,compute}.rs
    src/ensemble/{mod,calibration,consensus,meta,stacking}.rs
    src/research/{mod,physical,mathematical,informational,dfa,rqa}.rs
```

**Dependency graph:** `lemillion-db` ← `lemillion-cli`, `lemillion-esn` ← `lemillion-ensemble`

### lemillion-db (shared types)

- `Draw` — full draw record (id, date, 5 balls [1-50], 2 stars [1-12], winners, prize, My Million, ball_order, star_order, cycle_number)
- `Pool` — enum `Balls | Stars` with `size()`, `pick_count()`, `numbers_from(draw)`
- `Suggestion` — struct `{ balls: [u8;5], stars: [u8;2], score: f64 }`
- `validate_draw` — ensures ranges and no duplicates
- `db.rs` — SQLite via `rusqlite` (bundled). Single `draws` table keyed by `draw_id`. `INSERT OR IGNORE` handles duplicates. DB at `./data/lemillion.db`
- `delete_draw(conn, draw_id)` — deletes a draw by ID, returns `bool`
- `fetch_last_draws_numbers(conn, limit)` — lighter query returning only `Vec<([u8;5], [u8;2])>`
- Re-exports `rusqlite` as `pub use rusqlite` (used by ensemble crate for `Connection` access)

### lemillion-cli (original CLI)

Data flows: **CSV -> SQLite -> analysis -> terminal tables**.

- `import.rs` — Parses the FDJ CSV (`;`-delimited, French decimals, `flexible(true)`). Dates DD/MM/YYYY to YYYY-MM-DD. Preserves physical extraction order (ball_order, star_order) and cycle_number.
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

23 active models behind `trait ForecastModel` (takes `&[Draw]`, returns `Vec<f64>` summing to 1.0). Each model declares a `SamplingStrategy` (default: `Consecutive`) — models marked **(S)** use `Sparse { span_multiplier }` for wider temporal coverage during calibration. Models can also override `calibration_stride()` (default 1) to skip test points during calibration for expensive models. `SamplingStrategy` also supports `FullHistory` for walk-forward trained models.

**IMPORTANT (v4)**: The Stresa machine uses 3 central bars + 8 external bars (mod-8 symmetry for balls), and the Pâquerette uses 4 blades (mod-4 for stars). All modular models are pool-aware via `mod4::modulus(pool)`. Star data before 2016-09-27 uses incompatible pool sizes and is filtered via `filter_star_era()`.

**Active ensemble** (`base_models()` in `models/mod.rs`):
1. **Logistic** — SGD with L2 regularization via ndarray, 14 features
2. **Transformer** — Reservoir Transformer: self-attention à poids fixes (random frozen) + ridge regression readout. 2 couches, 4 têtes, d_model=32, masque causal, positional encoding sinusoïdal. (context_len=50, ridge_lambda=1e-3, smoothing=0.25, seed=42)
3. **TDA** **(S×3)** — Topological Data Analysis: homologie persistante H0 via Union-Find sur nuages de points 5D. (window_size=30, correlation_window=50, smoothing=0.25)
4. **Physics** **(S×4)** — Simulation de biais mécaniques: fréquences EWMA (α=0.15, v12), biais log-ratio avec shrinkage bayésien, drift temporel, lissage spatial gaussien par rangée de rack (σ=0.5, v12), CUSUM (threshold=2.0, v12). (prior_strength=5, drift_window=20, smoothing=0.25)
5. **StresaSGD** **(S×4)** — Simulateur physique Bayésien de la machine Stresa. Modèle génératif inverse avec row_bias (5 rangées rack), persistence inter-tirages et température. SGD par différences finies. (lr=0.02, reg=0.01, n_epochs=10, smoothing=0.30, star_smoothing=0.18)
6. **StresaChaos** **(S×4)** — Simulateur dynamique chaotique de la machine Stresa. Reconstruction espace des phases (Takens), analyse attracteur (Lyapunov local, RQA, UPO), prédiction multi-méthode fusionnée. Smoothing adaptatif basé sur la prédictibilité locale Lyapunov. (k_pilot=20, mod4_weight=0.90, smoothing=0.25)
7. **CondSummaryV2** — Naive Bayes factorisé: P(num|sum_bin) × P(num|spread_bin) × P(num|odd_count) / P(num)². (smoothing=0.35, n_bins=5)
8. **StarSpecialist** **(S×3)** — Modèle dédié étoiles: 4 micro-experts combinés via Hedge. Retourne uniforme pour Pool::Balls. (smoothing=0.35, learning_rate=0.15, min_draws=30)
9. **TransferEntropy** **(S×4)** — Paires causales TE(source→target) avec seuil vs baseline permutée (5 shuffles). Multi-lag scoring (v12): decay [1.0, 0.5, 0.25] for lags 1-3. Cross-pool TE(ball→star) pour les étoiles. calibration_stride=2. (alpha=2.0, te_threshold_factor=3.0, smoothing=0.30, n_top_sources=15)
10. **StarPair** **(S×3)** — Prédit les 66 paires d'étoiles via 3 experts Hedge. Expose `predict_pair_distribution()` pour scoring direct par paires. (smoothing=0.25, learning_rate=0.15, min_draws=50)
11. **ContextKNN** **(S×3)** — k-NN basé sur contexte 4D. Fonctionne pour boules ET étoiles. (k=15, smoothing=0.30, min_draws=30)
12. **MaxEntropy** — Maximum entropy distribution with bias tilts from detected non-random signals. (smoothing=0.25, mod_tilts coef=0.08)
13. **HMM** **(S×3)** — Hidden Markov Model with K=4/8 hidden states (pool-aware). Baum-Welch + forward algorithm. (n_states=4/8, max_iter=20, smoothing=0.35)
14. **Boltzmann** **(S×3)** — Markov Random Field for ball interactions. Mean-field approximation. (coupling_strength=0.3, smoothing=0.20)
15. **Hawkes** **(S×3)** — Self-exciting point process for pair co-occurrence temporal clustering. (decay=0.1, excitation=0.5, smoothing=0.20)
16. **BOCPD** **(S×4)** — Bayesian Online Changepoint Detection (Adams-MacKay). (expected_run_length=200, smoothing=0.35)
17. **DecadePersist** **(S×3)** — Matrice de transition entre profils de décades (5 rangées rack Stresa). 126 profils.
18. **ModularBalls** **(S×3)** — Teste 3 symétries (mod-3, mod-8, mod-24) et sélectionne la plus informative par KL-divergence.
19. **Compression** **(S×3)** — Mesure la compressibilité (deflate) de la séquence avec chaque candidat ajouté.
20. **TripletBoost** **(S×3)** — Triplet co-occurrence patterns avec pondération temporelle et seuil z>2. (smoothing=0.25)
21. **StarMomentum** **(S×3)** — DFA Hurst exponent pour détection momentum/mean-reversion sur fréquences étoiles. (smoothing=0.30)
22. **Spread** **(S×3)** — Clustering gaussien sur le spread (max-min) des tirages récents. (smoothing=0.25)
23. **DrawOrder** **(S×4)** — Exploite l'ordre d'extraction physique (brevet Stresa US6145836A). Matrice positionnelle complète N×5 avec poids par position basés sur l'entropie inverse. (ewma_alpha=0.05, smoothing=0.25, min_draws_with_order=50)

**Retired models** (modules still exist but excluded from `base_models()` — see `RETIRED_MODELS.md` for full details):
- v11: TLR, ParticleStresa, ForbiddenPatterns (BMA dilution — near-uniform models take weight from TransferEntropy)
- `predict_decorrelated()` uses same upside-only clamp as `predict()` (v12 fix — was bilateral [2,5])
- v9: Copula, Wavelet, Renewal (0% poids boules+étoiles)
- v7: RqaPredictability, UnitDigit, DelayedMI, Community, GapModel (0% poids boules+étoiles)
- v5: RandomForest, ModProfile, StresaSMC (corr 0.913 StresaChaos), GapDynamics, ModTrans (corr 0.975 ModularBalls)
- v4: CTW, Spectral, StarRecency, BME/Mixture
- v1-v3: Dirichlet, EWMA, Markov, Retard, HotStreak, ESN, TakensKNN, NVAR, NVAR-Memo, CondSummary (V1), Diffusion, JackpotContext

**Utility types** (not `ForecastModel`, used internally):
- **JointConditionalModel** (`joint.rs`) — Sequential joint conditional scorer: P(draw) = P(b1) × P(b2|b1) × ... × P(b5|b1..b4). Mirrors Stresa's physical process. Scores complete grids via `score_grid()`. Also `score_balls_with_confidence()` returning (log_score, confidence) for adaptive marginal/joint blend in jackpot mode [0.15, 0.40] based on confidence.
- **SummaryPredictor** (`summary_predictor.rs`) — Markov order-1 on summary states (sum_bin, spread_bin, odd_count). Used by sampler for adaptive filtering.
- **NeuralScorer** (`neural_scorer.rs`) — 3-layer MLP (62→32→16→1) ensemble of 5 networks. Scores complete grids for reranking. Saved/loaded as `neural_scorer.json`. Optional in jackpot mode (`--neural-rerank`).

**Feature engineering** (`features/compute.rs`): 21 features per number — freq_3, freq_5, freq_10, freq_20, retard, retard_norm, trend, mean_gap, std_gap, is_odd, decade, decade_density, day_of_week, recent_sum_norm, recent_even_count, pair_freq, gap_acceleration, low_half, mod4_class, mod4_class_freq, mod4_transition.

**Analysis** (`analysis.rs`): 5 non-randomness statistical tests — permutation entropy, runs test (Wald-Wolfowitz), auto-mutual information (AMI lag 1-20), correlation dimension (Grassberger-Procaccia, dims 3/5/7), Lempel-Ziv complexity. Each returns `AnalysisResult { test_name, value, expected_random, verdict: Signal|Neutral|Random, detail }`.

**Sampler** (`sampler.rs`):
- `date_seed()` — deterministic YYYYMMDD seed from local date (via chrono), used when no `--seed` provided
- `optimal_grid(ball_probs, star_probs)` — deterministic grid: top 5 balls + top 2 stars by ensemble probability, score = product of (prob/uniform)
- `generate_suggestions_from_probs(...)` — thin wrapper around `generate_suggestions_filtered`
- `generate_suggestions_filtered(...)` — oversampling + diversity with optional `StructuralFilter`
- `generate_suggestions_joint(...)` — **primary production path**. Two-phase: 50% marginal sampling scored with `CoherenceScorer`, 50% template recombination from top-20 historical draws. Score = `bayesian_score * (0.5 + coherence_score)`
- `StructuralFilter` — rejects candidates outside historical percentile bounds for ball sum, max consecutive run, odd count, spread (max-min). Built via `StructuralFilter::from_history(draws, pool)`
- `CoherenceScorer` — computes historical sum/spread stats, pair and triplet co-occurrence frequencies. Weights: 0.35*sum + 0.25*spread + 0.25*pair + 0.15*triplet
- `compute_bayesian_score(balls, stars, ball_probs, star_probs)` — standalone scoring function
- Diversity: greedy selection enforcing `min_ball_diff` (default 2) differing balls between any pair
- `generate_suggestions_jackpot(ball_probs, star_probs, count, filter, coherence, joint_model, star_pair_probs, excluded_balls, conditioner, neural_scorer)` — **jackpot mode**: exhaustive enumeration of top-N combinations by P(5+2). Adaptive K (balls/stars subset, K_balls minimum 25), 5 nested loops, min-heap for large enumerations. Ball scoring blends 70% marginal + 30% joint conditional (via `JointConditionalModel.score_balls()`). Optional `star_pair_probs` for pair-aware star scoring (from StarPairModel). Optional `excluded_balls` for K-reduction via consensus exclusion (disabled for jackpots >100M). Optional `conditioner` for ball→star conditional scoring. Returns `JackpotResult { suggestions, total_jackpot_probability, enumeration_size, filtered_size, improvement_factor }`
- `BallStarConditioner` — models conditional star pair probabilities given ball context (sum_bin × spread_bin). v12: 3×3=9 contexts (removed odd_bin for 3x more data per context). Built from history with adaptive tercile bins (v9), produces 66 pair probability tables per context bin. Adaptive blend via `adaptive_blend()`: `obs/(obs+20)` per context (0 with few obs, ~0.78 with many).
- `conviction_temperature(verdict)` — adaptive temperature: HighConviction→0.10, MediumConviction→0.20, LowConviction→0.25
- `conviction_temperature_split(conviction)` — separate ball/star temperatures. v12: more aggressive defaults (balls: 0.12/0.10/0.05, stars: 0.35/0.30/0.18). For jackpots >100M: T_balls forced to 1.0 (no sharpening); T_stars remains aggressive.
- `few_grid_temperature(n_grids)` — forced temperatures for few-grid mode (3-10 grilles). N≤3: (0.55, 0.25), 4-6: (0.60, 0.30), 7-10: (0.65, 0.30). Overrides skill/conviction.
- `select_optimal_n_grids(candidates, n_grids, max_common_balls, max_common_stars)` — greedy selection of N grids maximizing P(5+2) with diversity constraints. v10: Liu-Teo geometric mean overlap bonus (1 common ball optimal = 1.25x).
- `optimal_subset_k(ball_probs, n_grids)` — (v11) computes optimal subset size K for jackpot enumeration. Currently returns k=50 for typical distributions (no-op).
- `cross_window_stability(results)` — (v11) Gaussian penalty on LL variance across calibration windows. sigma=0.5 (very mild).

**Expected Value** (`expected_value.rs`):
- `PopularityModel` — models player number selection biases (birthday bias, lucky numbers, recency)
- `compute_ev(balls, stars, ball_probs, star_probs, popularity, jackpot)` — computes expected value per euro
- `ScoredSuggestion` — extends `Suggestion` with `bayesian_score`, `anti_popularity`, `ev_per_euro`

**Coverage** (`coverage.rs`):
- `optimize_coverage(ball_probs, star_probs, draws, n_tickets, jackpot, seed)` — generates a set of tickets optimizing pair/triple coverage across the ticket set
- `CoverageStats` — tracks ball pair, star pair, and ball triple coverage metrics

**Ensemble** (`ensemble/`):
- Walk-forward validation (NO future data leakage): train on `draws[t+1..t+1+window]`, test on `draws[t]`
- Stride sampling (~100 test points) for calibration performance
- Weights: `exp(best_ll - uniform_ll)`, normalized to sum to 1.0
- Default ball windows: `20,30,50,80,100,150,200,300` (8 windows)
- Default star windows: `50,100,200,300,500` (5 windows) — longer for sparser star patterns
- `--pool balls|stars|both` for partial recalibration; `--star-windows` to override star windows
- `SamplingStrategy::Sparse` models are calibrated on both consecutive AND sparse strategies, keeping the best
- `CalibrationResult` includes `sparse: bool` field; `ModelCalibration` includes `best_sparse: bool`
- Calibration results saved/loaded as JSON (`calibration.json`) — backward-compatible via `#[serde(default)]`

**Consensus** (`ensemble/consensus.rs`):
- `build_consensus_map` — 2D classification (prob × median_spread) -> `StrongPick | DivisivePick | StrongAvoid | Uncertain`
- `consensus_score(balls, stars, ball_consensus, star_consensus)` — scores a grid against the consensus map: StrongPick=+2, DivisivePick=+1, Uncertain=0, StrongAvoid=-1. Range: -7 to +14
- `compute_exclusion_set(consensus, threshold, max_excluded)` — returns StrongAvoid numbers with consensus_value < threshold, sorted most negative first, capped at max_excluded. Used for K-reduction in jackpot enumeration

**Log-linear pool** (`ensemble/mod.rs`):
- Geometric combination: `P(x) ∝ Π P_i(x)^w_i` (log-space computation with max-subtraction)
- Replaces linear pool. Optimal when models have partially independent information sources.
- Log-ratio clamp (v11): upside-only asymmetric. `max_positive_log_ratio = 4 + 6*(1-h_ratio)` → [4.0, 10.0]. Downside unclamped for full concentration. Replaces v10's inopérant absolute clamp.

**Agreement boost** (`ensemble/mod.rs`):
- `predict_with_agreement_boost(draws, pool, strength)` — boosts numbers that are both FAVORED (above uniform) AND UNANIMOUS (low spread). `agreement = deviation × spread_agreement`

**Online Hedge** (`ensemble/mod.rs`):
- `compute_hedge_weights(models, draws, ball_w, star_w, n_recent, eta)` — multiplicative weight update over N recent draws: `w[m] *= exp(-η * loss_m)`

**Meta-predictor** (`ensemble/meta.rs`):
- `RegimeFeatures` — 7D context features: sum_norm, spread_norm, mod4_cosine, recent_entropy, day_of_week (MARDI=0/VENDREDI=1), gap_compression, rqa_determinism
- `MetaPredictor::train(draws, detailed_ll, lambda)` — ridge regression from context features to per-model LL adjustments
- `MetaPredictor::weight_adjustments(features)` — returns per-model multiplicative weight adjustments for current context

**Stacking** (`ensemble/stacking.rs`):
- `StackingWeights` — per-number elastic net regression: model weights `[pool_size][n_models]`, context weights `[pool_size][6]`, bias `[pool_size]`
- Walk-forward data collection (~200 points), trains separate balls/stars stacking layers
- LASSO (L1) + L2 regularization via coordinate descent — automatically eliminates irrelevant models
- Context features = `RegimeFeatures` (6D)

**Redundancy detection** (`ensemble/calibration.rs`):
- `collect_detailed_ll(model, draws, window, pool, strategy)` — per-draw LL collection
- `detect_redundancy(detailed_ll, threshold)` — Pearson correlation between model LL series, flags pairs > threshold
- `compute_decorrelated_weights` — continuous Gaussian penalty (center=0.50, σ=0.20): smooth penalization instead of binary threshold

**Prediction pipeline** (`cmd_predict` in `main.rs`):
1. Load calibration weights (or uniform fallback)
2. Predict ball/star distributions via `EnsembleCombiner`
3. Display top-N distributions + consensus maps
4. Compute conviction score on RAW distributions (before temperature)
5. Apply temperature: `--n-grids` overrides with `few_grid_temperature()`, else explicit `--temperature`, else adaptive split via `conviction_temperature_split()`. For jackpot >100M: T_balls=1.0, T_stars from conviction.
6. Display optimal grid (deterministic, max probability)
7. If `--jackpot-mode` or `--n-grids`: multi-perspective enumeration (3 weight profiles: Principal, Star-variant, Exploratoire). All candidates pooled and selected via `select_optimal_n_grids()` with diversity constraints. `--n-grids N` activates few-grid mode with forced temperature. K-reduction disabled for jackpots >100M. `--no-filter` disables structural filter. `--jackpot <amount>` sets jackpot for EV calculation.
8. Else (default EV mode): generate sampled suggestions via `generate_suggestions_ev`, sort by EV descending, display with consensus column

**Research** (`research/`):
Bias detection module organized in 3 axes. Each test returns `TestResult { test_name, category, statistic, p_value, effect_size, verdict: Significant|Marginal|NotSignificant, detail }`.

- **Physical** (`research/physical.rs`): 4 analyses based on Stresa machine knowledge
  1. **Rack position** — chi² by row (decades) and column (units) for balls/stars + windowed drift detection (w=100,200,500)
  2. **Trap bias** — per-number frequency deviation with Bonferroni correction (50 tests), chi² global, temporal drift across eras
  3. **Co-occurrence** — pair chi² (1225 pairs), triplet excess detection with z-scores
  4. **Ball/star independence** — chi² on contingency table + mutual information between ball sum and star sum

- **Mathematical** (`research/mathematical.rs`): 4 analyses in transformed spaces
  1. **Modular analysis** — chi² per residue class for mod 2,3,4,5,7,10 + inter-draw cosine correlation (mod 4 priority)
  2. **Sum/spread** — ball sum vs theoretical distribution (z-test), spread (max-min), odd count vs hypergeometric H(50,25,5)
  3. **Co-occurrence graph** — 50-node graph, decade modularity vs Monte Carlo baseline (1000 permutations)
  4. **Gap analysis** — K-S test vs geometric distribution, trend regression, lag-1 autocorrelation

- **Informational** (`research/informational.rs`): 4 information-theoretic analyses
  1. **Conditional entropy** — H(t+1|t) via summary encoding, information gain ratio
  2. **Transfer entropy** — TE(i→j) for top-15 ball pairs + ball→star cross-pool, with FDR threshold
  3. **Compression** — deflate ratio for 4 encodings (raw, mod4, decades, gaps) vs random baseline
  4. **Delayed MI** — MI at lags 1,2,3,5,7,10,20,50 with shuffled surrogate baseline

- **DFA** (`research/dfa.rs`): Detrended Fluctuation Analysis
  1. **Hurst Exponent (Balls)** — average H across 50 ball frequency series with z-test vs 0.5
  2. **Hurst Exponent (Stars)** — average H across 12 star frequency series with z-test vs 0.5
  3. **Persistence Classification** — determines momentum vs mean-reversion strategy

- **RQA** (`research/rqa.rs`): Recurrence Quantification Analysis
  1. **RR** (Recurrence Rate), **DET** (Determinism), **L_max**, **ENTR** (Entropy), **LAM** (Laminarity)
  2. Monte Carlo baseline (100 surrogates) for p-values and effect sizes
  3. Takens embedding (dim=3, delay=1) on ball frequency series

**CLI subcommands:** `calibrate`, `weights`, `predict`, `history`, `compare`, `add-draw`, `fix-draw`, `rebuild`, `backtest`, `analyze`, `coverage`, `research`, `benchmark`, `interactive`

**Interactive mode** (`interactive.rs`):
- REPL loop with menu: add draw, calibrate, predict, history, compare, weights, analyze, coverage, research, quit
- Commands by number (`1`-`10`), French name (`ajouter`, `analyser`, `couverture`, `recherche`), or alias (`add`, `q`, `cal`, `pred`, `hist`, `comp`, `ana`, `cov`, `res`)
- Each command prompts for parameters with defaults; errors are caught without exiting the loop

### Conventions

- Error handling: `anyhow::Result` everywhere, no `unwrap()` in production code
- Tests use `Connection::open_in_memory()` for DB tests
- The `--alpha` flag has different semantics per model (Dirichlet: prior strength, EWMA: decay factor 0<alpha<1)
- `draws[0]` = most recent draw in all contexts
- `Draw.balls`/`stars` are always sorted ascending; `Draw.ball_order`/`star_order` preserve physical extraction order (None for manually-added draws)
- `Pool::numbers_from(draw)` to extract balls/stars from a Draw
- Display uses `comfy-table` with `UTF8_FULL` preset and `Cell::fg(Color)` for emphasis
- Progress bars via `indicatif` for calibrate, backtest, and grid search
- ESN `sparsity` = fraction of zeros in reservoir matrix
- ESN `leaking_rate`: 1.0 = full update (no memory), 0.0 = no update
