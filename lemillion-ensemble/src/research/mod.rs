pub mod physical;
pub mod mathematical;
pub mod informational;
pub mod dfa;
pub mod rqa;

use lemillion_db::models::Draw;

#[derive(Debug, Clone, PartialEq)]
pub enum ResearchVerdict {
    Significant,
    Marginal,
    NotSignificant,
}

impl std::fmt::Display for ResearchVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResearchVerdict::Significant => write!(f, "SIGNIFICATIF"),
            ResearchVerdict::Marginal => write!(f, "MARGINAL"),
            ResearchVerdict::NotSignificant => write!(f, "NON-SIGNIFICATIF"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub category: String,
    pub statistic: f64,
    pub p_value: Option<f64>,
    pub effect_size: f64,
    pub verdict: ResearchVerdict,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ResearchReport {
    pub physical: Vec<TestResult>,
    pub mathematical: Vec<TestResult>,
    pub informational: Vec<TestResult>,
    pub dfa: Vec<TestResult>,
    pub rqa: Vec<TestResult>,
}

impl ResearchReport {
    pub fn all_results(&self) -> Vec<&TestResult> {
        self.physical.iter()
            .chain(self.mathematical.iter())
            .chain(self.informational.iter())
            .chain(self.dfa.iter())
            .chain(self.rqa.iter())
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResearchCategory {
    Physical,
    Mathematical,
    Informational,
    Dfa,
    Rqa,
    All,
}

pub fn run_all_research(draws: &[Draw], category: ResearchCategory, window: Option<usize>) -> ResearchReport {
    let effective_draws = match window {
        Some(w) if w < draws.len() => &draws[..w],
        _ => draws,
    };

    let physical = if matches!(category, ResearchCategory::Physical | ResearchCategory::All) {
        physical::run_physical_tests(effective_draws)
    } else {
        vec![]
    };

    let mathematical = if matches!(category, ResearchCategory::Mathematical | ResearchCategory::All) {
        mathematical::run_mathematical_tests(effective_draws)
    } else {
        vec![]
    };

    let informational = if matches!(category, ResearchCategory::Informational | ResearchCategory::All) {
        informational::run_informational_tests(effective_draws)
    } else {
        vec![]
    };

    let dfa_results = if matches!(category, ResearchCategory::Dfa | ResearchCategory::All) {
        dfa::run_dfa_tests(effective_draws)
    } else {
        vec![]
    };

    let rqa_results = if matches!(category, ResearchCategory::Rqa | ResearchCategory::All) {
        rqa::run_rqa_tests(effective_draws)
    } else {
        vec![]
    };

    let mut report = ResearchReport {
        physical,
        mathematical,
        informational,
        dfa: dfa_results,
        rqa: rqa_results,
    };

    // Appliquer la correction FDR Benjamini-Hochberg à 5%
    apply_fdr_correction(&mut report, 0.05);

    report
}

/// Chi-squared statistic: sum of (observed - expected)^2 / expected
pub(crate) fn chi_squared(observed: &[f64], expected: &[f64]) -> f64 {
    observed.iter().zip(expected.iter())
        .filter(|(_, e)| **e > 0.0)
        .map(|(o, e)| (o - e).powi(2) / e)
        .sum()
}

/// Approximate chi-squared p-value using Wilson-Hilferty normal approximation
pub(crate) fn chi_squared_p_value(chi2: f64, df: usize) -> f64 {
    if df == 0 {
        return 1.0;
    }
    let k = df as f64;
    // Wilson-Hilferty approximation: transform chi2/k to approximate normal
    let z = ((chi2 / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    // Standard normal survival function approximation
    normal_survival(z)
}

/// Approximate standard normal survival function P(Z > z)
pub(crate) fn normal_survival(z: f64) -> f64 {
    // Abramowitz & Stegun approximation 26.2.17
    if z < -8.0 {
        return 1.0;
    }
    if z > 8.0 {
        return 0.0;
    }
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-z * z / 2.0).exp()
        * (t * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))));
    if z >= 0.0 { p } else { 1.0 - p }
}

/// Two-sided p-value from z-score
pub(crate) fn two_sided_p(z: f64) -> f64 {
    2.0 * normal_survival(z.abs())
}

/// Verdict from p-value
pub(crate) fn verdict_from_p(p: f64) -> ResearchVerdict {
    if p < 0.01 {
        ResearchVerdict::Significant
    } else if p < 0.05 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    }
}

/// Applique la correction Benjamini-Hochberg FDR sur un ResearchReport.
/// Re-étiquète les verdicts : seuls les tests survivant au FDR à `alpha` sont Significant.
pub fn apply_fdr_correction(report: &mut ResearchReport, alpha: f64) {
    // Collecter tous les résultats avec p-value
    let mut all_results: Vec<&mut TestResult> = report.physical.iter_mut()
        .chain(report.mathematical.iter_mut())
        .chain(report.informational.iter_mut())
        .chain(report.dfa.iter_mut())
        .chain(report.rqa.iter_mut())
        .filter(|r| r.p_value.is_some())
        .collect();

    if all_results.is_empty() {
        return;
    }

    // Trier par p-value croissante
    all_results.sort_by(|a, b| {
        a.p_value.unwrap().partial_cmp(&b.p_value.unwrap()).unwrap_or(std::cmp::Ordering::Equal)
    });

    let m = all_results.len();

    // BH: pour le rang k (1-indexed), seuil = alpha * k / m
    // On trouve le plus grand k tel que p(k) <= alpha * k / m
    let mut max_significant_rank = 0;
    for (i, r) in all_results.iter().enumerate() {
        let rank = i + 1;
        let threshold = alpha * rank as f64 / m as f64;
        if r.p_value.unwrap() <= threshold {
            max_significant_rank = rank;
        }
    }

    // Re-étiqueter les verdicts
    for (i, r) in all_results.iter_mut().enumerate() {
        let rank = i + 1;
        if rank <= max_significant_rank {
            r.verdict = ResearchVerdict::Significant;
        } else if r.p_value.unwrap() < 0.05 {
            r.verdict = ResearchVerdict::Marginal;
        } else {
            r.verdict = ResearchVerdict::NotSignificant;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_chi_squared_uniform() {
        let observed = vec![25.0, 25.0, 25.0, 25.0];
        let expected = vec![25.0, 25.0, 25.0, 25.0];
        let chi2 = chi_squared(&observed, &expected);
        assert!(chi2.abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_p_value_large() {
        // Large chi2 with 3 df should give small p-value
        let p = chi_squared_p_value(20.0, 3);
        assert!(p < 0.001);
    }

    #[test]
    fn test_chi_squared_p_value_small() {
        // Small chi2 with 3 df should give large p-value
        let p = chi_squared_p_value(1.0, 3);
        assert!(p > 0.5);
    }

    #[test]
    fn test_verdict_from_p() {
        assert_eq!(verdict_from_p(0.001), ResearchVerdict::Significant);
        assert_eq!(verdict_from_p(0.03), ResearchVerdict::Marginal);
        assert_eq!(verdict_from_p(0.1), ResearchVerdict::NotSignificant);
    }

    #[test]
    fn test_run_all_research_no_panic() {
        let draws = make_test_draws(100);
        let report = run_all_research(&draws, ResearchCategory::All, None);
        assert!(!report.physical.is_empty());
        assert!(!report.mathematical.is_empty());
        assert!(!report.informational.is_empty());
        assert!(!report.dfa.is_empty());
        assert!(!report.rqa.is_empty());
    }

    #[test]
    fn test_run_research_filtered() {
        let draws = make_test_draws(100);
        let report = run_all_research(&draws, ResearchCategory::Physical, None);
        assert!(!report.physical.is_empty());
        assert!(report.mathematical.is_empty());
        assert!(report.informational.is_empty());
        assert!(report.dfa.is_empty());
        assert!(report.rqa.is_empty());
    }

    #[test]
    fn test_run_research_dfa_filtered() {
        let draws = make_test_draws(200);
        let report = run_all_research(&draws, ResearchCategory::Dfa, None);
        assert!(report.physical.is_empty());
        assert!(report.mathematical.is_empty());
        assert!(report.informational.is_empty());
        assert!(!report.dfa.is_empty());
        assert!(report.rqa.is_empty());
    }

    #[test]
    fn test_run_research_rqa_filtered() {
        let draws = make_test_draws(100);
        let report = run_all_research(&draws, ResearchCategory::Rqa, None);
        assert!(report.physical.is_empty());
        assert!(report.mathematical.is_empty());
        assert!(report.informational.is_empty());
        assert!(report.dfa.is_empty());
        assert!(!report.rqa.is_empty());
    }

    #[test]
    fn test_run_research_with_window() {
        let draws = make_test_draws(200);
        let report = run_all_research(&draws, ResearchCategory::All, Some(50));
        assert!(!report.physical.is_empty());
    }
}
