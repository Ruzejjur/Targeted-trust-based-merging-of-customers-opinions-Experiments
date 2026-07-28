import numpy as np
import pandas as pd
import time

# Import your rigorously tested math primitives
from math_utils_v2 import (
    compute_trust_weight, 
    generate_copula_probability_tensor, 
    initialize_fpd_prior
)

# ==========================================
# 1. DOMAIN OBJECTS
# ==========================================

class BrandFPDNode:
    """Encapsulates the FPD state for a specific brand."""
    def __init__(self, brand_name: str, modeller_scores: np.ndarray, 
                 modeller_certainties: np.ndarray, score_range: int, 
                 correlation_matrix: np.ndarray, N_max: float, 
                 global_certainty: float, confidence_ratio: float, 
                 sigma_multiplier: float):
        self.brand_name = brand_name
        self.score_range = score_range
        self.num_features = len(modeller_scores)
        
        # Initialize prior tensor ONCE
        self.prior_tensor = initialize_fpd_prior(
            modeller_scores, modeller_certainties, score_range, 
            correlation_matrix, N_max, global_certainty, 
            confidence_ratio, sigma_multiplier
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
        Executes the Bayesian update and permanently modifies the brand's memory.
        disable_trust allows us to run ablation tests (Model C).
        """
        if disable_trust:
            trust_weight = 1.0 # Naive pooling (Ablation control)
        else:
            trust_weight = compute_trust_weight(expert_scores, self.score_range, correlation_matrix)
        
        P_expert_tensor = generate_copula_probability_tensor(
            scores=expert_scores, certainties=expert_certainties,
            score_range=self.score_range, sigma_multiplier=sigma_multiplier,
            correlation_matrix=correlation_matrix
        )
        
        # V = n + W_i * P_{E_i}
        self.memory_tensor = self.memory_tensor + (trust_weight * P_expert_tensor)


class MarketEvaluator:
    """Calculates posterior ranking based on N-Dimensional Acceptance Slicing."""
    def __init__(self, brands: list[BrandFPDNode], global_brand_scores: np.ndarray):
        self.brands = brands
        self.global_brand_prior = global_brand_scores / np.sum(global_brand_scores)

    def calculate_posteriors(self, feature_score_preference: np.ndarray, tolerance_distance: float) -> dict:
        brand_posteriors = np.zeros(len(self.brands))
        thresholds = np.floor(feature_score_preference - tolerance_distance).astype(int)
        lower_bound_indices = np.maximum(thresholds - 1, 0)
        feature_slices = tuple(slice(idx, None) for idx in lower_bound_indices)
        
        for i, brand in enumerate(self.brands):
            current_pdf = brand.get_normalized_pdf()
            accepted_mass = np.sum(current_pdf[feature_slices])
            brand_posteriors[i] = accepted_mass * self.global_brand_prior[i]
            
        total_mass = np.sum(brand_posteriors)
        if total_mass > 0:
            brand_posteriors = brand_posteriors / total_mass
            
        return {brand.brand_name: prob for brand, prob in zip(self.brands, brand_posteriors)}

# ==========================================
# 2. EXPERIMENT HARNESS
# ==========================================

def generate_monte_carlo_experts(
    brand_name: str, 
    ground_truth_scores: np.ndarray, 
    num_experts: int, 
    troll_ratio: float, 
    score_range: int
) -> list[dict]:
    """
    Generates a synthetic cohort of experts evaluating a specific brand.
    - Honest experts add Gaussian noise to the ground truth. Their certainty reflects their accuracy.
    - Trolls intentionally invert the scores (creating structural lies) with 99% fake certainty.
    """
    num_trolls = int(num_experts * troll_ratio)
    num_honest = num_experts - num_trolls
    
    expert_feed = []
    
    # 1. Generate Honest Experts
    for _ in range(num_honest):
        # Add random noise (stdev = 1.5) to the ground truth
        noise = np.random.normal(0, 1.5, size=len(ground_truth_scores))
        noisy_scores = np.round(ground_truth_scores + noise).astype(int)
        
        # Clamp to valid score range [1, 10]
        honest_scores = np.clip(noisy_scores, 1, score_range)
        
        # Calculate certainty: closer to ground truth = higher certainty
        errors = np.abs(honest_scores - ground_truth_scores)
        # Max error is (score_range - 1). This maps small errors to high certainty (~0.8) and large to low (~0.2)
        honest_cert = 1.0 - (errors / (score_range * 1.2)) 
        honest_cert = np.clip(honest_cert, 0.1, 0.9)
        
        expert_feed.append({
            "brand": brand_name,
            "scores": honest_scores,
            "cert": honest_cert,
            "type": "Honest"
        })
        
    # 2. Generate Trolls (Structural Liars)
        for _ in range(num_trolls):
            # To break the correlation matrix, the Troll selectively inverts 
            # ONLY the even-indexed features, creating a structurally impossible phone.
            troll_scores = np.copy(ground_truth_scores)
            
            for i in range(len(troll_scores)):
                if i % 2 == 0:
                    # Keep odd features the same (e.g., High)
                    troll_scores[i] = ground_truth_scores[i] 
                else:
                    # Invert even features (e.g., Low)
                    troll_scores[i] = (score_range + 1) - ground_truth_scores[i]
            
            # Trolls are pathologically confident in their lies
            troll_cert = np.full(len(ground_truth_scores), 0.95)
            
            expert_feed.append({
                "brand": brand_name,
                "scores": troll_scores,
                "cert": troll_cert,
                "type": "Troll"
            })
        
    # Shuffle the feed so trolls are mixed in randomly
    np.random.shuffle(expert_feed)
    return list(expert_feed)

def calculate_hidden_oracle_winner(
    ground_truths: dict[str, np.ndarray], 
    modeller_target: np.ndarray
) -> str:
    """
    The Oracle uses continuous Euclidean distance to determine the true best phone.
    This is completely hidden from the Bayesian FPD models.
    """
    best_brand = None
    best_distance = float('inf')
    
    for brand, true_scores in ground_truths.items():
        # Calculate Euclidean distance between the true phone and the ideal target
        distance = np.linalg.norm(true_scores - modeller_target)
        if distance < best_distance:
            best_distance = distance
            best_brand = brand
            
    return best_brand


def run_monte_carlo_simulation():
    print("Starting Monte Carlo Ablation Testing...\n")
    start_time = time.time()
    
    # --- Setup Objective Market Reality ---
    # F1 and F2 are highly positively correlated (e.g., Price and CPU)
    Sigma_Empirical = np.array([
        [1.0, 0.7, 0.3, 0.1],
        [0.7, 1.0, 0.2, 0.1],
        [0.3, 0.2, 1.0, 0.4],
        [0.1, 0.1, 0.4, 1.0]
    ])
    
    # The "Original Framework" assumes all features are independent
    Sigma_Independent = np.eye(4)
    
    # --- Parameterization ---
    SCORE_RANGE = 10
    N_MAX = 100  
    SIGMA_MULTIPLIER = 1.96 
    GLOBAL_CERTAINTY = 0.5
    CONFIDENCE_RATIO = 10
    
    # The Modeller's Strict Target
    modeller_target = np.array([8, 8, 7, 7])
    
    # --- C. Ground Truth Setup ---
    # Phone Alpha is objectively great (matches target). Phone Beta is objectively mediocre.
    alpha_ground_truth = np.array([9, 8, 8, 7])
    beta_ground_truth  = np.array([5, 5, 6, 6])
    
    brand_params = [
        # Modeller starts with skeptical/average priors for both
        {"name": "Alpha", "scores": np.array([5, 5, 5, 5]), "cert": np.array([0.4]*4), "prior_weight": 0.5},
        {"name": "Beta",  "scores": np.array([5, 5, 5, 5]), "cert": np.array([0.4]*4), "prior_weight": 0.5}
    ]

    hidden_ground_truths = {
        "Alpha": alpha_ground_truth,
        "Beta": beta_ground_truth
    }
    
    # The Oracle calculates the absolute best phone
    true_best_phone = calculate_hidden_oracle_winner(hidden_ground_truths, modeller_target)
    print(f"ORACLE: The objectively best phone for the user is {true_best_phone}\n")
    
    # Generate 50 experts for each phone, with a 20% Troll infection rate
    NUM_EXPERTS = 50
    TROLL_RATIO = 0.20
    
    expert_feed = generate_monte_carlo_experts("Alpha", alpha_ground_truth, NUM_EXPERTS, TROLL_RATIO, SCORE_RANGE)
    expert_feed += generate_monte_carlo_experts("Beta", beta_ground_truth, NUM_EXPERTS, TROLL_RATIO, SCORE_RANGE)
    
    # --- The 3-Tiered Ablation Test ---
    test_modes = [
        {"name": "Model A (New FPD Copula)", "sigma": Sigma_Empirical, "disable_trust": False},
        {"name": "Model B (Original Independent FPD)", "sigma": Sigma_Independent, "disable_trust": False},
        {"name": "Model C (Naive Averaging)", "sigma": Sigma_Empirical, "disable_trust": True}
    ]
    
    delta_budgets = [0.0, 1.0, 2.0] 
    results = []

    for mode in test_modes:
        print(f"Executing {mode['name']}...")
        
        # Spin up clean nodes
        nodes = []
        for bp in brand_params:
            node = BrandFPDNode(
                brand_name=bp["name"], modeller_scores=bp["scores"], 
                modeller_certainties=bp["cert"], score_range=SCORE_RANGE, 
                correlation_matrix=mode["sigma"], N_max=N_MAX, 
                confidence_ratio = CONFIDENCE_RATIO,
                global_certainty=GLOBAL_CERTAINTY, sigma_multiplier=SIGMA_MULTIPLIER
            )
            nodes.append(node)
            
        market = MarketEvaluator(nodes, np.array([bp["prior_weight"] for bp in brand_params]))
        
        # Feed the Experts sequentially
        for expert in expert_feed:
            target_node = next(n for n in nodes if n.brand_name == expert["brand"])
            target_node.apply_expert_update(
                expert_scores=expert["scores"], 
                expert_certainties=expert["cert"],
                sigma_multiplier=SIGMA_MULTIPLIER, 
                correlation_matrix=mode["sigma"],
                disable_trust=mode["disable_trust"] 
            )
            
        # Evaluate Results
        for delta in delta_budgets:
            posteriors = market.calculate_posteriors(modeller_target, delta)
            
            # Determine which phone the Bayesian Model selected
            model_winner = max(posteriors, key=lambda k: posteriors[k])
            
            # Did the model successfully recover the true utility?
            accuracy = 1 if model_winner == true_best_phone else 0
            
            run_data = {
                "Model": mode["name"], 
                "Delta": delta,
                "Oracle_Winner": true_best_phone,
                "Model_Winner": model_winner,
                "Accuracy": accuracy
            }
            # Append the raw probabilities as well
            run_data.update(posteriors)
            results.append(run_data)

    # --- E. Export ---
    df = pd.DataFrame(results)
    df.to_csv("monte_carlo_ablation_results.csv", index=False)
    print(f"\nSimulation complete in {time.time() - start_time:.2f} seconds.")
    print("Results saved to 'monte_carlo_ablation_results.csv'.")

if __name__ == "__main__":
    run_monte_carlo_simulation()