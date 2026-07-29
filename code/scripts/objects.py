import numpy as np

from math_utils_v2 import (
    compute_trust_weight, 
    generate_copula_probability_tensor,
    generate_monte_carlo_copula_probability_tensor,
    initialize_fpd_prior
)


class BrandFPDNode:
    """Encapsulates the FPD state for a specific brand."""
    def __init__(self, brand_name: str, modeller_scores: np.ndarray, 
                 modeller_certainties: np.ndarray, score_range: int, 
                 correlation_matrix: np.ndarray, N_max: float, 
                 global_certainty: float, confidence_ratio: float, 
                 sigma_multiplier: float, approximate_copula: bool = False):
        self.brand_name = brand_name
        self.score_range = score_range
        self.num_features = len(modeller_scores)
        self.approximate_copula = approximate_copula
        
        # Initialize prior tensor ONCE
        self.prior_tensor = initialize_fpd_prior(
            modeller_scores, modeller_certainties, score_range, 
            correlation_matrix, N_max, global_certainty, 
            confidence_ratio, sigma_multiplier, approximate_copula
        )
        # Memory state begins identically to the prior
        self.memory_tensor = np.copy(self.prior_tensor)

    def get_normalized_pdf(self) -> np.ndarray:
        return self.memory_tensor / np.sum(self.memory_tensor)

    def apply_expert_update(self, expert_scores: np.ndarray, 
                            expert_certainties: np.ndarray, 
                            sigma_multiplier: float, 
                            correlation_matrix: np.ndarray,
                            disable_trust: bool = False):
        """
        Executes the Bayesian update and modifies the brand's memory.
        disable_trust allows us to run ablation tests (Model C).
        """
        if disable_trust:
            trust_weight = 1.0 # Naive pooling
        else:
            trust_weight = compute_trust_weight(expert_scores, self.score_range, correlation_matrix)

        if self.approximate_copula is False: 
            P_expert_tensor = generate_copula_probability_tensor(
                scores=expert_scores, certainties=expert_certainties,
                score_range=self.score_range, sigma_multiplier=sigma_multiplier,
                correlation_matrix=correlation_matrix)

        else:
            P_expert_tensor = generate_monte_carlo_copula_probability_tensor(scores=expert_scores, certainties=expert_certainties,
                        score_range=self.score_range, sigma_multiplier=sigma_multiplier,
                        correlation_matrix=correlation_matrix, num_samples=1000000
            )
        
        # V = n + W_i * P_{E_i}
        self.memory_tensor = self.memory_tensor + (trust_weight * P_expert_tensor)


class PosteriorEvaluator:
    """Calculates posterior ranking based on N-Dimensional Acceptance Slicing."""
    def __init__(self, brand_opinions: list[BrandFPDNode], global_brand_scores: np.ndarray):
        self.brand_opinions = brand_opinions
        self.global_brand_prior = global_brand_scores / np.sum(global_brand_scores)

    def calculate_posteriors(self, feature_score_preference: np.ndarray, tolerance_distance: float) -> dict:
        brand_posteriors = np.zeros(len(self.brand_opinions))
        thresholds = np.floor(feature_score_preference - tolerance_distance).astype(int)
        lower_bound_indices = np.maximum(thresholds - 1, 0)
        feature_slices = tuple(slice(idx, None) for idx in lower_bound_indices)
        
        for i, brand in enumerate(self.brand_opinions):
            current_pdf = brand.get_normalized_pdf()
            accepted_mass = np.sum(current_pdf[feature_slices])
            brand_posteriors[i] = accepted_mass * self.global_brand_prior[i]
            
        total_mass = np.sum(brand_posteriors)

        assert all(brand_posteriors) > 0, "Posterior probabilities cannot be negative."
        
        brand_posteriors = brand_posteriors / total_mass
            
        return {brand.brand_name: prob for brand, prob in zip(self.brand_opinions, brand_posteriors)}