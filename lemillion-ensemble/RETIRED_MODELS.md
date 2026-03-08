# Retired Models Registry

Journal de traçabilité de tous les modèles testés et retirés de l'ensemble actif.

---

## RandomForest — Retiré le 2026-03-08 (v5)
- **Skill boules** : 0 (poids 0%)
- **Skill étoiles** : 0 (poids 0%)
- **Raison** : Aucun signal détecté ni pour les boules ni pour les étoiles. 100 arbres, profondeur 3.
- **Corrélation** : Faible corrélation avec les autres modèles
- **Condition de réintégration** : Si un feature engineering enrichi améliore le skill au-dessus de 0
- **Module** : `models/random_forest.rs`

## ModProfile (Mod4Profile) — Retiré le 2026-03-08 (v5)
- **Skill boules** : 0 (poids 0%)
- **Skill étoiles** : 0 (poids 0%)
- **Raison** : Matrice de transition sur 792 profils modulaires boules — trop d'états pour les données disponibles (~635 tirages). Aucun signal.
- **Corrélation** : Modéré avec ModTrans
- **Condition de réintégration** : Si le nombre de tirages dépasse 2000+ pour mieux estimer les transitions
- **Module** : `models/mod4_profile.rs`

## StresaSMC — Retiré le 2026-03-08 (v5)
- **Skill boules** : 0 (poids 0%)
- **Skill étoiles** : 2.2%
- **Raison** : Corrélation 0.913 avec StresaChaos (12.8% poids étoiles). Redondant — StresaChaos domine sur les deux pools.
- **Corrélation** : 0.913 avec StresaChaos
- **Condition de réintégration** : Si StresaChaos est retiré ou si les deux divergent significativement
- **Module** : `models/stresa.rs` (StresaSmcModel)

## GapDynamics — Retiré le 2026-03-08 (v5)
- **Skill boules** : 0 (poids 0%)
- **Skill étoiles** : 0 (poids 0%)
- **Raison** : Fonction de hasard empirique + autocorrélation. Aucun signal détecté malgré les corrections v4.
- **Corrélation** : Faible
- **Condition de réintégration** : Si la recherche détecte un signal de gap-compression plus fort
- **Module** : `models/gap_dynamics.rs`

## ModTrans (Mod4Transition) — Retiré le 2026-03-08 (v5)
- **Skill boules** : poids 3.54%
- **Skill étoiles** : poids modéré
- **Raison** : Corrélation 0.975 avec ModularBalls (6.81% poids). Quasi-identique, ModularBalls est plus performant car il teste 3 symétries (mod-3/8/24) et sélectionne la meilleure.
- **Corrélation** : 0.975 avec ModularBalls
- **Condition de réintégration** : Si ModularBalls est retiré
- **Module** : `models/mod4.rs` (Mod4TransitionModel)

---

## Modèles retirés lors des audits précédents

### CTW — Retiré v4 (2026-03-08)
- **Raison** : Corrélation 0.94 avec ModTrans. Context Tree Weighting redondant.
- **Module** : `models/ctw.rs`

### Spectral — Retiré v4 (2026-03-08)
- **Raison** : Aucune périodicité réelle détectée dans les données EuroMillions.
- **Module** : `models/spectral.rs`

### StarRecency — Retiré v4 (2026-03-08)
- **Raison** : Subsumé par StarSpecialist qui inclut un expert EWMA multi-échelle plus sophistiqué.
- **Module** : `models/star_recency.rs`

### BME/Mixture — Retiré v4 (2026-03-08)
- **Raison** : Redondant avec le mécanisme Hedge de l'ensemble lui-même.
- **Module** : `models/mixture.rs`

### Dirichlet — Retiré v1-v3
- **Raison** : Prior conjugué trop simple, dominé par les modèles fréquentistes.
- **Module** : `models/dirichlet.rs`

### EWMA — Retiré v1-v3
- **Raison** : Signal insuffisant, subsumé par les modèles basés sur les fréquences.
- **Module** : (dans lemillion-cli)

### Markov — Retiré v1-v3
- **Raison** : Trop d'états pour les données disponibles.
- **Module** : `models/markov.rs`

### ESN (Echo State Network) — Retiré v1-v3
- **Raison** : Réservoir aléatoire sans signal suffisant pour le lottery.
- **Module** : `models/esn.rs`

### CondSummary (V1) — Retiré v1-v3
- **Raison** : ~4800 états pour ~634 tirages. Remplacé par CondSummaryV2 (Naive Bayes factorisé).
- **Module** : `models/conditional.rs`

### JackpotContext — Retiré v1-v3
- **Raison** : Modèle basé sur le montant du jackpot. Signal négatif en calibration.
- **Module** : `models/jackpot_context.rs`

### Diffusion — Retiré v1-v3
- **Raison** : Modèle de diffusion sans signal.

### Retard / HotStreak — Retiré v1-v3
- **Raison** : Heuristiques simples dominées par des modèles plus sophistiqués.

### TakensKNN / NVAR / NVAR-Memo — Retiré v1-v3
- **Raison** : Embedding de Takens et NVAR sans signal suffisant.
