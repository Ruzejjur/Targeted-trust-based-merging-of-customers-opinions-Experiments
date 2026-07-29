import numpy as np
import pandas as pd
from statsmodels.stats.correlation_tools import corr_nearest
from scipy.stats import norm, beta, multivariate_normal
from scipy.spatial.distance import mahalanobis
from itertools import product

from typing import Iterator


# Correlation matrix object
class CorrelationMatrix: 
    """
    #TODO: Add description
    """

    def __init__(self, data):
        # Calculate Kendall's Tau-b for all columns of the data
        self.correlation_matrix = data.corr(method='kendall', numeric_only=True)

        self.pearson_correlation_matrix = self._greiner_relation(self.correlation_matrix)
        # Ensure the matrix is positive semi-definite
        self.pearson_correlation_matrix = corr_nearest(self.pearson_correlation_matrix)

    def _greiner_relation(self, correlation_matrix: pd.DataFrame) -> np.ndarray:
        """
        Applies Greiner's relation to transform Kendall's Tau-b to Persons correlation coefficient.
        #TODO: Finish the description
        Parameters: 

        Returns: 
        
        """
        pearson_correlation_matrix = correlation_matrix.map(lambda x: np.sin((np.pi/2)*x))
        pearson_correlation_matrix = pearson_correlation_matrix.to_numpy()

        return pearson_correlation_matrix


# Trust calculation
def normalise_score(score:int, score_range:int)->float: 
    """
    #TODO: Add description
    """
    normlised_score = float((score-0.5)/score_range)

    return normlised_score


def calculate_z_scores(arr:np.ndarray)->np.ndarray:
    """
    #TODO: Add description
    """

    z_scores = norm.ppf(arr)

    return z_scores

def compute_mahalanobis_distance(z_scores:np.ndarray, pearson_correlation_matrix:np.ndarray)-> float:
    """
    #TODO: Add description
    """

    inverse_corr_matrix = np.linalg.inv(pearson_correlation_matrix)

    mahalanobis_distance = mahalanobis(z_scores, np.zeros(z_scores.shape), inverse_corr_matrix)

    return mahalanobis_distance


def compute_trust_weight(
    scores: np.ndarray, 
    score_range: int, 
    pearson_correlation_matrix: np.ndarray
) -> float: 
    """
    #TODO: Add description
    """
    # Normalize correctly using score_range, not len(scores)
    normalised_scores = np.array([normalise_score(score, score_range) for score in scores])
    
    # Pass the normalized probabilities to the Z-score function
    z_scores = calculate_z_scores(normalised_scores)
    
    # Calculate distance
    distance = compute_mahalanobis_distance(z_scores, pearson_correlation_matrix)

    # Exponential decay for trust
    trust_weight = np.exp(-0.5 * (distance ** 2)) # Note: Mahalanobis returns distance, we need distance^2 for the Gaussian kernel

    return float(trust_weight)

# Projecting marginal scores to distributions

def parametrise_beta_dist(normalised_score: float, certainty:float) -> tuple[float, float]:
    """
    #TODO: Add description
    """

    alpha = 1+2*normalised_score*(certainty/(1-certainty))
    beta = 1+2*(1-normalised_score)*(certainty/(1-certainty))

    return alpha, beta

def v_max(score_range: int, sigma_multiplier: float) -> float:
    """
    #TODO: Add description
    """
    # Calculating mid value of the score and flooring. 
    # Note: For the given beta distribution parametrisation the variance is the worst 
    #       for the midlle score, so we create an upper estimate for the variance
    mid_score_val = 0.5
    if score_range % 2 == 0:
        mid_score_val = 0.45

    #TODO: Find intuitive explanation for v_max
    v_max = (mid_score_val*(1-mid_score_val))*(2*score_range*sigma_multiplier)**2 - 1

    return v_max

def maximum_certainty(v_max: float) -> float:
    """
    #TODO: Add description
    """

    max_c = 1 - 2/v_max

    return max_c

# Copula modelling 

def generate_copula_probability_tensor(
    scores: np.ndarray, 
    certainties: np.ndarray, 
    score_range: int, 
    sigma_multiplier: float, 
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Generates the full N-dimensional joint probability tensor for a given opinion 
    using the Gaussian Copula over discrete score bins.
    """
    num_features = len(scores)
    feature_z_bounds = []
    
    # Uniform bin edges (e.g., 0.0, 0.1 ... 1.0)
    normalized_edges = np.linspace(0, 1, score_range + 1)

    variance_cap = v_max(score_range, sigma_multiplier)
    allowed_max_certainty = maximum_certainty(variance_cap)
    
    # Map Marginal Opinions to Latent Z-Bounds
    for score, certainty in zip(scores, certainties):
        
        # Parameterize the Beta distribution
        norm_score = normalise_score(score, score_range) 

        # Clamp the user's certainty so it does not exceed the mathematical limits
        safe_certainty = min(certainty, allowed_max_certainty)

        alpha_param, beta_param = parametrise_beta_dist(norm_score, safe_certainty)
        
        # Calculate cumulative probability mass at each bin edge
        cumulative_probs = beta.cdf(normalized_edges, alpha_param, beta_param)

        # Clamp intermediate probabilities to prevent float64 absolute 0.0 or 1.0.
        # This prevents norm.ppf from generating 'inf' on intermediate edges, 
        # which causes SciPy to attempt 'inf - inf' -> NaN.
        cumulative_probs = np.clip(cumulative_probs, 1e-15, 1.0 - 1e-15)
        
        # Translate to Standard Normal Z-space
        z_edges = norm.ppf(cumulative_probs)
        
        # Force absolute boundaries to catch tail probability mass
        z_edges[0] = -np.inf
        z_edges[-1] = np.inf
        
        feature_z_bounds.append(z_edges)
        
    # Extract 1D bounds for the meshgrid
    lower_bounds_1d = [z[:-1] for z in feature_z_bounds]
    upper_bounds_1d = [z[1:] for z in feature_z_bounds]
    
    # Generate the N-Dimensional Grid
    grid_lower = np.meshgrid(*lower_bounds_1d, indexing='ij')
    grid_upper = np.meshgrid(*upper_bounds_1d, indexing='ij')
    
    # Flatten the grid into a list of bounding boxes
    flat_lower = np.stack(grid_lower, axis=-1).reshape(-1, num_features)
    flat_upper = np.stack(grid_upper, axis=-1).reshape(-1, num_features)
    
    # Evaluate the Copula
    mvn_dist = multivariate_normal(mean=np.zeros(num_features), cov=correlation_matrix) # type: ignore (for false positive Pylance detection)
    
    prob_mass_flat = np.array([
        mvn_dist.cdf(x=upper, lower_limit=lower)
        for lower, upper in zip(flat_lower, flat_upper)
    ])
    
    # Reshape back into the discrete N-Dimensional tensor
    tensor_shape = tuple([score_range] * num_features)
    P_copula_tensor = prob_mass_flat.reshape(tensor_shape)
    
    return P_copula_tensor

def generate_monte_carlo_copula_probability_tensor(
    scores: np.ndarray, 
    certainties: np.ndarray, 
    score_range: int, 
    sigma_multiplier: float, 
    correlation_matrix: np.ndarray,
    num_samples: int = 500000  # Number of Monte Carlo draws for the approximation
) -> np.ndarray:
    """
    Generates the N-dimensional joint probability tensor using a highly optimized 
    Monte Carlo sampling approach, bypassing the slow SciPy CDF integration loop.
    """
    num_features = len(scores)
    
    # Generate massive block of correlated samples from the Latent Space
    # Shape: (num_samples, num_features)
    latent_samples = np.random.multivariate_normal(
        mean=np.zeros(num_features), 
        cov=correlation_matrix, 
        size=num_samples
    )
    
    # Convert Latent Gaussian samples to Uniform [0, 1] marginals (The Copula Step)
    uniform_samples = norm.cdf(latent_samples)
    
    # Create an empty array to hold the mapped discrete scores
    discrete_scores = np.zeros_like(uniform_samples, dtype=int)
    
    # Map the Uniform samples into the Expert's specific Beta Distributions
    variance_cap = v_max(score_range, sigma_multiplier)
    allowed_max_certainty = maximum_certainty(variance_cap)
    
    for j in range(num_features):
        norm_score = normalise_score(scores[j], score_range)
        safe_certainty = min(certainties[j], allowed_max_certainty)
        alpha_param, beta_param = parametrise_beta_dist(norm_score, safe_certainty)
        
        # We use the Inverse Beta (ppf) to map the uniform Copula structure 
        # directly onto the expert's subjective feature distribution.
        # This converts the [0, 1] uniform values into [0, 1] Beta values.
        beta_samples = beta.ppf(uniform_samples[:, j], alpha_param, beta_param)
        
        # Quantize the continuous Beta values into discrete bins (1 to score_range)
        # e.g., a beta value of 0.85 on a 10-pt scale becomes score 9.
        discrete_bins = np.floor(beta_samples * score_range).astype(int)
        discrete_scores[:, j] = np.clip(discrete_bins, 0, score_range - 1)
        
    # Count the frequencies of every N-Dimensional state combination
    # We flatten the multi-dimensional coordinates into a single 1D index array 
    # using ravel_multi_index so NumPy can count them instantly in C.
    tensor_shape = tuple([score_range] * num_features)
    flat_indices = np.ravel_multi_index(discrete_scores.T, tensor_shape)
    
    # Count occurrences
    counts = np.bincount(flat_indices, minlength=np.prod(tensor_shape))
    
    # Convert counts to probability mass and reshape back to N-Dimensions
    prob_mass_flat = counts / num_samples
    P_copula_tensor = prob_mass_flat.reshape(tensor_shape)
    
    return P_copula_tensor

# Prior initialisation

def calculate_global_brand_preference(global_brand_score:np.ndarray) -> np.ndarray:
    """
    #TODO: Add description
    """

    prob = global_brand_score/np.sum(global_brand_score)

    return prob


def initialize_fpd_prior(
    modeller_scores: np.ndarray, 
    modeller_certainties: np.ndarray, 
    score_range: int, 
    correlation_matrix: np.ndarray, 
    N_max: float, 
    global_certainty: float,
    confidence_ratio: float,
    sigma_multiplier: float,
    approximate_copula: bool = False
) -> np.ndarray:
    """
    Constructs the N-dimensional discrete prior tensor based on the primary modeller's 
    subjective marginal opinions and the objective Gaussian Copula correlation.
    
    Parameters:
    - modeller_scores (np.ndarray): The expected scores for each feature of each bramd (e.g., ["brand",8, 6, 7, 5]).
    - modeller_certainties (np.ndarray): The confidence [0, 1) for each score (e.g., ["brand", 0.3]).
    - score_range (int): The max score (e.g., 10).
    - correlation_matrix (np.ndarray): The objective Sigma matrix (shape: N x N).
    - global_certainty (float):
    """


    maximum_achievable_certainty = (confidence_ratio)/(confidence_ratio + 1)

    K = ((1-maximum_achievable_certainty)/maximum_achievable_certainty)*N_max

    global_certainty = K*(global_certainty/(1-global_certainty))

    # Generate the Copula distribution for the Modeller

    if approximate_copula is False:
        P_copula_tensor = generate_copula_probability_tensor(
            scores=modeller_scores,
            certainties=modeller_certainties,
            score_range=score_range,
            sigma_multiplier=sigma_multiplier,
            correlation_matrix=correlation_matrix
        )
    else: 
        P_copula_tensor = generate_monte_carlo_copula_probability_tensor(
            scores=modeller_scores,
            certainties=modeller_certainties,
            score_range=score_range,
            sigma_multiplier=sigma_multiplier,
            correlation_matrix=correlation_matrix
        )

    fpd_prior_tensor = 1e-6 + (global_certainty * P_copula_tensor)
    
    return fpd_prior_tensor

