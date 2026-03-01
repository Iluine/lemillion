# Backtest : Poids Calibres vs Poids Uniformes

Date : 2026-02-25

## Parametres

- Suggestions par tirage : 5000
- Oversample : 20
- Modeles : 14 (Dirichlet, EWMA, Logistic, RandomForest, Markov, Retard, HotStreak, ESN, TakensKNN, Spectral, CTW, NVAR, NVAR-Memo, BME)
- Calibration : walk-forward, windows 20,30,40,50,60,80,100
- Uniforme : 1/14 par modele (fallback sans calibration)

## Resultats sur 10 tirages

| Metrique | Calibre | Uniforme |
|---|---|---|
| Score moyen | 1.0665 | 1.0350 |
| Score min | 0.6628 | 0.4048 |
| Score max | 1.5992 | 1.6688 |
| Percentile moyen | 3.4% | 0.0% |
| Bits d'info moy. | -0.616 | -1.142 |
| Hit rate boules | 0.60/5 | 0.30/5 |
| Hit rate etoiles | 0.20/2 | 0.50/2 |
| Tirages > 50e pct | 0/10 | 0/10 |
| Rang estime | ~135M / 140M | ~140M / 140M |

## Resultats sur 50 tirages

| Metrique | Calibre | Uniforme |
|---|---|---|
| Score moyen | 1.0340 | 1.0365 |
| Score min | 0.5138 | 0.2140 |
| Score max | 1.7677 | 4.7239 |
| Percentile moyen | 6.5% | 5.1% |
| Bits d'info moy. | -0.708 | -1.302 |
| Hit rate boules | 0.52/5 | 0.50/5 |
| Hit rate etoiles | 0.32/2 | 0.40/2 |
| Tirages > 50e pct | 4/50 (8%) | 3/50 (6%) |
| Rang estime | ~131M / 140M | ~133M / 140M |

## Resultats sur 600 tirages (~5 ans)

| Metrique | Calibre | Uniforme |
|---|---|---|
| Score moyen | 1.0038 | 1.0356 |
| Score min | 0.3286 | 0.1041 |
| Score max | 2.7748 | 5.6637 |
| Percentile moyen | 3.8% | 4.4% |
| Bits d'info moy. | -0.912 | -1.491 |
| Hit rate boules | 0.52/5 | 0.53/5 |
| Hit rate etoiles | 0.34/2 | 0.34/2 |
| Tirages > 50e pct | 24/600 (4.0%) | 23/600 (3.8%) |
| Rang estime | ~135M / 140M | ~134M / 140M |

## Analyse

### Avantage calibration
- **Bits d'information** : gain stable de ~0.58 bit par tirage a travers toutes les tailles d'echantillon
- **Variance reduite** : scores min plus eleves (plancher releve), scores max plus bas
- **Percentile** : legerement meilleur sur 10 et 50 tirages, equivalent sur 600

### Avantage uniforme
- **Scores extremes** : max plus eleves (coups de chance isoles amplifie par certains modeles)
- **Score moyen** : legerement superieur sur 600 tirages (1.036 vs 1.004), mais non significatif

### Conclusion
- La calibration apporte un gain reel mais modeste en information extraite (bits), avec un comportement plus stable
- Les deux approches restent tres proches du hasard pur (score ~1.0, rang ~135M/140M, <5% des tirages > 50e percentile)
- Le taux de succes attendu pour un vrai pouvoir predictif serait >50% au 50e percentile ; on observe ~4%
- L'EuroMillions est confirme comme un jeu veritablement aleatoire
