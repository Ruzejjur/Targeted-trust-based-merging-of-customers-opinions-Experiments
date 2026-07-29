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

def calculate_hidden_oracle_winner(
    ground_truths: dict[str, np.ndarray], 
    modeller_preference: np.ndarray
) -> str:
    """
    The Oracle uses continuous Euclidean distance to determine the true best phone.
    This is completely hidden from the Bayesian FPD models.
    """
    best_brand = None
    best_distance = float('inf')
    
    for brand, true_scores in ground_truths.items():
        # Calculate Euclidean distance between the true phone and the ideal target
        distance = np.linalg.norm(true_scores - modeller_preference)
        if distance < best_distance:
            best_distance = distance
            best_brand = brand

    assert best_brand is not None, "best_brand cannot be of type None."
    
    return best_brand


def run_monte_carlo_simulation(results_path: Path) -> None:
    logging.info("Starting Monte Carlo ablation testing...\n")
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
    Sigma_Independent = np.eye(Sigma_Empirical.shape[0])
    
    # --- Parameterization ---
    SCORE_RANGE = 10
    N_MAX = 100  
    SIGMA_MULTIPLIER = 1.96 
    GLOBAL_CERTAINTY = 0.5
    CONFIDENCE_RATIO = 10
    APPROXIMATE_COPULA = True

    # Generate NUM_EXPERTS experts for each phone, with a 20% Troll infection rate
    NUM_EXPERTS = 10
    TROLL_RATIO = 0.7
    
    # The Modeller's minimal preference for scores
    modeller_target = np.array([8, 8, 7, 7])
    
    # Ground truth setup
    # Phone Alpha is objectively great (matches target). Phone Beta is objectively mediocre.
    alpha_ground_truth = np.array([9, 8, 8, 7])
    beta_ground_truth  = np.array([5, 5, 6, 6])
    
    initial_brand_opinions = [
        # Modeller starts with skeptical/average priors for both
        {"name": "Brand_0", "scores": np.array([5, 5, 5, 5]), "cert": np.array([0.4]*4), "overal_brand_preference_score": 5},
        {"name": "Brand_1",  "scores": np.array([5, 5, 5, 5]), "cert": np.array([0.4]*4), "overal_brand_preference_score": 5}
    ]

    hidden_ground_truths = {
        "Brand_0": alpha_ground_truth,
        "Brand_1": beta_ground_truth
    }
    
    # The Oracle calculates the absolute best phone
    true_best_phone = calculate_hidden_oracle_winner(hidden_ground_truths, modeller_target)
    logging.info(f"ORACLE: The objectively best phone for the user is {true_best_phone}\n")
    

    
    expert_feed = generate_monte_carlo_experts("Brand_0", alpha_ground_truth, NUM_EXPERTS, TROLL_RATIO, SCORE_RANGE)
    expert_feed += generate_monte_carlo_experts("Brand_1", beta_ground_truth, NUM_EXPERTS, TROLL_RATIO, SCORE_RANGE)
    
    # --- The 3-Tiered Ablation Test ---
    test_modes = [
        {"name": "Model A (FPD Copula)", "sigma": Sigma_Empirical, "disable_trust": False},
        {"name": "Model B (Original Independent FPD)", "sigma": Sigma_Independent, "disable_trust": False},
        {"name": "Model C (Naive Averaging)", "sigma": Sigma_Empirical, "disable_trust": True}
    ]
    
    delta_budgets = [0.0, 1.0, 2.0] 
    results = []

    context_manager = logging_redirect_tqdm()

    with context_manager:
        for i in tqdm(range(len(test_modes)), desc="Test mode"):
            logging.info(f"\n Executing {test_modes[i]['name']}...")
            
            # Spin up clean nodes
            nodes = []
            for bp in initial_brand_opinions:
                node = BrandFPDNode(
                    brand_name=bp["name"], modeller_scores=bp["scores"], 
                    modeller_certainties=bp["cert"], score_range=SCORE_RANGE, 
                    correlation_matrix=test_modes[i]["sigma"], N_max=N_MAX, 
                    confidence_ratio = CONFIDENCE_RATIO,
                    global_certainty=GLOBAL_CERTAINTY, sigma_multiplier=SIGMA_MULTIPLIER,
                    approximate_copula = APPROXIMATE_COPULA
                )
                nodes.append(node)
                
            market = PosteriorEvaluator(nodes, np.array([bp["overal_brand_preference_score"] for bp in initial_brand_opinions]))
            
            # Feed the Experts sequentially

            for e in tqdm(range(len(expert_feed))):
                target_node = next(n for n in nodes if n.brand_name == expert_feed[e]["brand"])
                target_node.apply_expert_update(
                    expert_scores=expert_feed[e]["scores"], 
                    expert_certainties=expert_feed[e]["cert"],
                    sigma_multiplier=SIGMA_MULTIPLIER, 
                    correlation_matrix=test_modes[i]["sigma"],
                    disable_trust=test_modes[i]["disable_trust"] 
                )
                
            # Evaluate Results
            for delta in delta_budgets:
                posteriors = market.calculate_posteriors(modeller_target, delta)
                
                # Determine which phone the Bayesian Model selected
                model_winner = max(posteriors, key=lambda k: posteriors[k])
                
                # Did the model successfully recover the true utility?
                accuracy = 1 if model_winner == true_best_phone else 0
                
                run_data = {
                    "Model": test_modes[i]["name"], 
                    "Delta": delta,
                    "Oracle_Winner": true_best_phone,
                    "Model_Winner": model_winner,
                    "Accuracy": accuracy
                }
                # Append the raw probabilities as well
                run_data.update(posteriors)
                results.append(run_data)

# Export
    df = pd.DataFrame(results)
    
    # Generate Timestamp and Directory
    os.makedirs(results_path, exist_ok=True)
    
    # Package the Hyperparameters
    experiment_params = {
        "SCORE_RANGE": SCORE_RANGE,
        "N_MAX": float(N_MAX),
        "SIGMA_MULTIPLIER": float(SIGMA_MULTIPLIER),
        "GLOBAL_CERTAINTY": float(GLOBAL_CERTAINTY),
        "CONFIDENCE_RATIO": float(CONFIDENCE_RATIO),
        "APPROXIMATE_COPULA": APPROXIMATE_COPULA,
        "NUM_EXPERTS": NUM_EXPERTS,
        "TROLL_RATIO": float(TROLL_RATIO),
        "modeller_target": modeller_target.tolist(), # Convert numpy array to list for clean YAML
        "delta_budgets": delta_budgets
    }
    
    # Save Parameters to YAML
    yaml_path = results_path / "parameters.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(experiment_params, yaml_file, default_flow_style=False, sort_keys=False)
    
    # Save Results to CSV
    csv_path = results_path / "monte_carlo_ablation_results.csv"
    df.to_csv(csv_path, index=False)
    
    logging.info(f"\nSimulation complete in {time.time() - start_time:.2f} seconds.")
    logging.info(f"Results and parameters saved to directory: '{results_path}/'")

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(os.getcwd()) / "results/" / f"{timestamp}"

    setup_logging(results_path)

    run_monte_carlo_simulation(results_path)