import numpy as np
import pandas as pd
import time
from pathlib import Path 
import os
import yaml
from datetime import datetime
import logging

# Import math functions
from math_utils_v2 import (
    compute_trust_weight, 
    generate_copula_probability_tensor,
    generate_monte_carlo_copula_probability_tensor,
    initialize_fpd_prior
)
from objects import BrandFPDNode, PosteriorEvaluator

from custom_tqdm import tqdm, logging_redirect_tqdm

from helper_functions import setup_logging

def generate_hidden_user_profile(num_features: int, score_range: int):
    """
    Generates the absolute truth of what the simulated user wants.
    The FPD agent MUST NOT see these values.
    """
    # 1. Hidden Target (r_true)
    r_true = np.random.randint(1, score_range + 1, size=num_features)
    
    # 2. Hidden Strictness (a_hidden) - Uniform between 0.5 (flexible) and 5.0 (dealbreaker)
    a_hidden = np.random.uniform(0.5, 5.0, size=num_features)
    
    # 3. Hidden Weights (w_hidden) - Must sum to 1.0 using a flat Dirichlet distribution
    w_hidden = np.random.dirichlet(np.ones(num_features))
    
    return r_true, a_hidden, w_hidden

def corrupt_target_profile(r_true: np.ndarray, score_range: int):
    """
    Simulates bounded rationality by adding cognitive noise to the hidden target.
    This produces the f_target that the FPD agent actually receives.
    """
    noise = np.random.normal(loc=0.0, scale=1.0, size=len(r_true))
    f_target = np.round(r_true + noise).astype(int)
    
    # Clamp to ensure the noise doesn't push the score out of the bounds
    f_target = np.clip(f_target, 1, score_range)
    return f_target

def calculate_hidden_oracle_winner(
    ground_truths: dict[str, np.ndarray], 
    r_true: np.ndarray,
    a_hidden: np.ndarray,
    w_hidden: np.ndarray
) -> str:
    """
    The Oracle uses the continuous Sigmoid Utility function to evaluate the 
    true physical scores against the user's hidden continuous reality.
    """
    best_brand = None
    best_utility = -1.0
    
    for brand, true_scores in ground_truths.items():
        # U_j(f) = 1 / (1 + e^(-a_j * (f - r_true_j)))
        # Note: (true_scores - r_true) means exceeding the target yields utility approaching 1.0
        utilities = 1.0 / (1.0 + np.exp(-a_hidden * (true_scores - r_true)))
        
        # Weighted sum of feature utilities
        total_utility = np.sum(w_hidden * utilities)
        
        if total_utility > best_utility:
            best_utility = total_utility
            best_brand = brand

    assert best_brand is not None, "Oracle failed to find a winner."
    return best_brand


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
    - Trolls intentionally invert the scores (creating structural lies) with high fake certainty.
    """
    num_trolls = int(num_experts * troll_ratio)
    num_honest = num_experts - num_trolls
    
    expert_feed = []
    
    # Generate Honest Experts
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
        
    # Generate Trolls (Structural Liars)
    for _ in range(num_trolls):
        # To break the correlation matrix, the Troll selectively inverts 
        # ONLY the even-indexed features, creating a structurally impossible phone.
        troll_scores = np.copy(ground_truth_scores)
        
        for i in range(len(troll_scores)):
            if i % 2 == 0:
                # Keep even features the same (e.g., High)
                troll_scores[i] = ground_truth_scores[i] 
            else:
                # Invert odd features (e.g., Low)
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


def run_monte_carlo_simulation(results_path: Path) -> None:
    logging.info("Starting Monte Carlo ablation testing...\n")
    start_time = time.time()
    
    # --- Setup Objective Market Reality ---
    Sigma_Empirical = np.array([
        [1.0, 0.7, 0.3, 0.1],
        [0.7, 1.0, 0.2, 0.1],
        [0.3, 0.2, 1.0, 0.4],
        [0.1, 0.1, 0.4, 1.0]
    ])
    Sigma_Independent = np.eye(Sigma_Empirical.shape[0])
    
    # --- Parameterization ---
    SCORE_RANGE = 10
    N_MAX = 100  
    SIGMA_MULTIPLIER = 1.96 
    GLOBAL_CERTAINTY = 0.5
    CONFIDENCE_RATIO = 10
    APPROXIMATE_COPULA = True
    NUM_EXPERTS = 10
    TROLL_RATIO = 0.7
    
    # NEW: Define the statistical power of your simulation
    NUM_MC_ITERATIONS = 2
    NUM_FEATURES = Sigma_Empirical.shape[0]
    
    alpha_ground_truth = np.array([9, 8, 8, 7])
    beta_ground_truth  = np.array([5, 5, 6, 6])
    
    hidden_ground_truths = {
        "Brand_0": alpha_ground_truth,
        "Brand_1": beta_ground_truth
    }
    
    test_modes = [
        {"name": "Model A (FPD Copula)", "sigma": Sigma_Empirical, "disable_trust": False},
        {"name": "Model B (Original Independent FPD)", "sigma": Sigma_Independent, "disable_trust": False},
        {"name": "Model C (Naive Averaging)", "sigma": Sigma_Empirical, "disable_trust": True}
    ]
    
    delta_budgets = [0.0, 1.0, 2.0] 
    results = []

    context_manager = logging_redirect_tqdm()

    with context_manager:
        # THE MONTE CARLO LOOP
        for mc in tqdm(range(NUM_MC_ITERATIONS), desc="Monte Carlo Iterations"):
            
            # 1. Generate the Objective Truth of the User
            r_true, a_hidden, w_hidden = generate_hidden_user_profile(NUM_FEATURES, SCORE_RANGE)
            
            # 2. Corrupt the truth to simulate bounded articulation
            f_target_corrupted = corrupt_target_profile(r_true, SCORE_RANGE)
            
            # 3. The Oracle grades physical reality against hidden continuous desires
            true_best_phone = calculate_hidden_oracle_winner(
                hidden_ground_truths, r_true, a_hidden, w_hidden
            )
            
            # 4. Generate the stochastic experts for this specific run
            expert_feed = generate_monte_carlo_experts("Brand_0", alpha_ground_truth, NUM_EXPERTS, TROLL_RATIO, SCORE_RANGE)
            expert_feed += generate_monte_carlo_experts("Brand_1", beta_ground_truth, NUM_EXPERTS, TROLL_RATIO, SCORE_RANGE)
            
            # 5. Execute the Models
            for i in range(len(test_modes)):
                nodes = []
                for brand_name in ["Brand_0", "Brand_1"]:
                    node = BrandFPDNode(
                        brand_name=brand_name, 
                        modeller_scores=np.array([5]*NUM_FEATURES), # Weak uninformative start 
                        modeller_certainties=np.array([0.4]*NUM_FEATURES), 
                        score_range=SCORE_RANGE, 
                        correlation_matrix=test_modes[i]["sigma"], 
                        N_max=N_MAX, 
                        confidence_ratio=CONFIDENCE_RATIO,
                        global_certainty=GLOBAL_CERTAINTY, 
                        sigma_multiplier=SIGMA_MULTIPLIER,
                        approximate_copula=APPROXIMATE_COPULA
                    )
                    nodes.append(node)
                    
                market = PosteriorEvaluator(nodes, np.array([5, 5])) # Neutral brand preference
                
                # Feed the Experts
                for e in expert_feed:
                    target_node = next(n for n in nodes if n.brand_name == e["brand"])
                    target_node.apply_expert_update(
                        expert_scores=e["scores"], 
                        expert_certainties=e["cert"],
                        sigma_multiplier=SIGMA_MULTIPLIER, 
                        correlation_matrix=test_modes[i]["sigma"],
                        disable_trust=test_modes[i]["disable_trust"] 
                    )
                    
                # Evaluate Results using the CORRUPTED target, not the hidden truth
                for delta in delta_budgets:
                    posteriors = market.calculate_posteriors(f_target_corrupted, delta)
                    
                    model_winner = max(posteriors, key=lambda k: posteriors[k])
                    accuracy = 1 if model_winner == true_best_phone else 0
                    
                    run_data = {
                        "MC_Iteration": mc,
                        "Model": test_modes[i]["name"], 
                        "Delta": delta,
                        "Oracle_Winner": true_best_phone,
                        "Model_Winner": model_winner,
                        "Accuracy": accuracy
                    }
                    run_data.update(posteriors)
                    results.append(run_data)

    # Export Logic remains the same...
    df = pd.DataFrame(results)
    os.makedirs(results_path, exist_ok=True)
    
    experiment_params = {
        "NUM_MC_ITERATIONS": NUM_MC_ITERATIONS,
        "SCORE_RANGE": SCORE_RANGE,
        "N_MAX": float(N_MAX),
        "SIGMA_MULTIPLIER": float(SIGMA_MULTIPLIER),
        "GLOBAL_CERTAINTY": float(GLOBAL_CERTAINTY),
        "CONFIDENCE_RATIO": float(CONFIDENCE_RATIO),
        "APPROXIMATE_COPULA": APPROXIMATE_COPULA,
        "NUM_EXPERTS": NUM_EXPERTS,
        "TROLL_RATIO": float(TROLL_RATIO),
        "delta_budgets": delta_budgets
    }
    
    yaml_path = results_path / "parameters.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(experiment_params, yaml_file, default_flow_style=False, sort_keys=False)
    
    csv_path = results_path / "monte_carlo_ablation_results.csv"
    df.to_csv(csv_path, index=False)
    
    logging.info(f"\nSimulation complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(os.getcwd()) / "results/" / f"{timestamp}"

    setup_logging(results_path)

    run_monte_carlo_simulation(results_path)