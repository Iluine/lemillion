use std::collections::HashMap;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// Modèle spectral : analyse fréquentielle des séries binaires de présence/absence.
/// Utilise la FFT pour identifier les périodicités et extrapoler via autocorrélation.
pub struct SpectralModel {
    n_harmonics: usize,
    smoothing: f64,
}

impl SpectralModel {
    pub fn new(n_harmonics: usize, smoothing: f64) -> Self {
        Self {
            n_harmonics,
            smoothing,
        }
    }
}

impl Default for SpectralModel {
    fn default() -> Self {
        Self {
            n_harmonics: 5,
            smoothing: 0.7,
        }
    }
}

/// Calcule le spectre de puissance via FFT.
/// Retourne (fréquences, puissances) pour les fréquences positives (excluant DC).
fn power_spectrum(series: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = series.len();
    if n < 4 {
        return (vec![], vec![]);
    }

    let mean = series.iter().sum::<f64>() / n as f64;

    // Fenêtre de Hann + centrage
    let mut buffer: Vec<Complex<f64>> = series
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let hann = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos());
            Complex::new((x - mean) * hann, 0.0)
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let n_freq = n / 2;
    let freqs: Vec<f64> = (1..=n_freq).map(|k| k as f64 / n as f64).collect();
    let powers: Vec<f64> = (1..=n_freq)
        .map(|k| buffer[k].norm_sqr() / (n as f64 * n as f64))
        .collect();

    (freqs, powers)
}

/// Prédit la probabilité de présence pour un numéro donné,
/// en utilisant les composantes spectrales dominantes et l'autocorrélation.
fn predict_number(binary_series: &[f64], n_harmonics: usize) -> f64 {
    let n = binary_series.len();
    if n < 20 {
        return binary_series.iter().sum::<f64>() / n as f64;
    }

    let mean = binary_series.iter().sum::<f64>() / n as f64;
    let (freqs, powers) = power_spectrum(binary_series);

    if freqs.is_empty() {
        return mean;
    }

    // Trouver les top n_harmonics fréquences par puissance
    let mut indices: Vec<usize> = (0..powers.len()).collect();
    indices.sort_by(|&a, &b| {
        powers[b]
            .partial_cmp(&powers[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(n_harmonics);

    // Pour chaque fréquence dominante, utiliser la période correspondante
    // pour prédire via l'autocorrélation
    let variance: f64 = binary_series
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / n as f64;

    if variance < 1e-12 {
        return mean;
    }

    let total_power: f64 = powers.iter().sum();
    if total_power < 1e-12 {
        return mean;
    }

    let mut prediction = mean;

    for &idx in &indices {
        let freq = freqs[idx];
        let period = (1.0 / freq).round() as usize;

        if period == 0 || period >= n {
            continue;
        }

        // Poids proportionnel à la puissance relative
        let weight = powers[idx] / total_power;

        // Autocorrélation au lag = period
        let acf = autocorrelation_at_lag(binary_series, mean, variance, period);

        // Contribution : si le numéro était présent il y a `period` tirages
        // et que l'autocorrélation est positive, augmenter la probabilité
        let last_at_period = binary_series[n - 1 - period.min(n - 1)];
        prediction += weight * acf * (last_at_period - mean);
    }

    prediction.max(1e-6)
}

/// Calcule l'autocorrélation normalisée au lag donné.
fn autocorrelation_at_lag(series: &[f64], mean: f64, variance: f64, lag: usize) -> f64 {
    let n = series.len();
    if lag >= n || variance < 1e-12 {
        return 0.0;
    }

    let covariance: f64 = series[..n - lag]
        .iter()
        .zip(series[lag..].iter())
        .map(|(&x, &y)| (x - mean) * (y - mean))
        .sum::<f64>()
        / (n - lag) as f64;

    covariance / variance
}

impl ForecastModel for SpectralModel {
    fn name(&self) -> &str {
        "Spectral"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < 30 {
            return uniform;
        }

        let mut raw_probs = Vec::with_capacity(size);

        for num in 1..=size as u8 {
            // Série binaire en ordre chronologique
            let binary: Vec<f64> = draws
                .iter()
                .rev()
                .map(|d| {
                    if pool.numbers_from(d).contains(&num) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();

            let pred = predict_number(&binary, self.n_harmonics);
            raw_probs.push(pred);
        }

        // Lisser avec la distribution uniforme
        let uniform_val = 1.0 / size as f64;
        let mut dist: Vec<f64> = raw_probs
            .iter()
            .map(|&p| self.smoothing * p + (1.0 - self.smoothing) * uniform_val)
            .collect();

        // Normaliser
        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for p in &mut dist {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_harmonics".into(), self.n_harmonics as f64),
            ("smoothing".into(), self.smoothing),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_power_spectrum_basic() {
        // Série sinusoïdale simple
        let n = 64;
        let series: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let (freqs, powers) = power_spectrum(&series);
        assert!(!freqs.is_empty());
        assert!(!powers.is_empty());

        // Le pic devrait être autour de freq = 4/64 = 0.0625
        let max_idx = powers
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak_freq = freqs[max_idx];
        assert!(
            (peak_freq - 4.0 / 64.0).abs() < 2.0 / 64.0,
            "pic spectral attendu ~0.0625, obtenu {peak_freq}"
        );
    }

    #[test]
    fn test_autocorrelation_at_lag() {
        // Série périodique de période 4
        let series: Vec<f64> = (0..100).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let var: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;
        let acf4 = autocorrelation_at_lag(&series, mean, var, 4);
        let acf2 = autocorrelation_at_lag(&series, mean, var, 2);
        assert!(
            acf4 > acf2,
            "autocorrélation au lag 4 devrait être plus forte que lag 2"
        );
    }

    #[test]
    fn test_spectral_predict_valid_distribution() {
        let model = SpectralModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_spectral_predict_stars() {
        let model = SpectralModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars));
    }

    #[test]
    fn test_spectral_few_draws_uniform() {
        let model = SpectralModel::default();
        let draws = make_test_draws(5);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_predict_number_basic() {
        let series: Vec<f64> = (0..200).map(|i| if i % 10 == 0 { 1.0 } else { 0.0 }).collect();
        let pred = predict_number(&series, 5);
        assert!(pred > 0.0, "prédiction devrait être positive");
        assert!(pred < 1.0, "prédiction devrait être < 1.0");
    }
}
