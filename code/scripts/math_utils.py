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

        alpha_param, beta_param = parametrise_beta_dist(norm_score, certainty)
        
        # Calculate cumulative probability mass at each bin edge
        cumulative_probs = beta.cdf(normalized_edges, alpha_param, beta_param)
        
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
    mvn_dist = multivariate_normal(mean=np.zeros(num_features), cov=correlation_matrix)
    
    prob_mass_flat = np.array([
        mvn_dist.cdf(x=upper, lower_limit=lower)
        for lower, upper in zip(flat_lower, flat_upper)
    ])
    
    # Reshape back into the discrete N-Dimensional tensor
    tensor_shape = tuple([score_range] * num_features)
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
    sigma_multiplier: float
) -> np.ndarray:
    """
    Constructs the N-dimensional discrete prior tensor based on the primary modeller's 
    subjective marginal opinions and the objective Gaussian Copula correlation.
    
    Parameters:
    - modeller_scores (np.ndarray): The expected scores for each feature of each bramd (e.g., ["brand",8, 6, 7, 5]).
    - modeller_certainties (np.ndarray): The confidence [0, 1) for each score (e.g., ["brand", 0.3]).
    - score_range (int): The max score (e.g., 10).
    - correlation_matrix (np.ndarray): The objective Sigma matrix (shape: N x N).
    - global_certainty (float): .
    """

    maximum_achievable_certainty = (confidence_ratio)/(1-confidence_ratio)

    K = ((1-maximum_achievable_certainty)/maximum_achievable_certainty)*N_max

    global_certainty = K*(global_certainty/(1-global_certainty))

    # Generate the Copula distribution for the Modeller
    P_copula_tensor = generate_copula_probability_tensor(
        scores=modeller_scores,
        certainties=modeller_certainties,
        score_range=score_range,
        sigma_multiplier=sigma_multiplier,
        correlation_matrix=correlation_matrix
    )
        
    fpd_prior_tensor = 1.0 + (global_certainty * P_copula_tensor)
    
    return fpd_prior_tensor


# Opinion update

def apply_expert_update(
    memory_tensor: np.ndarray,
    expert_scores: np.ndarray, 
    expert_certainties: np.ndarray, 
    score_range: int, 
    sigma_multiplier: float, 
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Executes the Bayesian update for an incoming expert opinion.
    #TODO: Add description
    """
    # Calculate how much we trust this expert (using corrected Mahalanobis logic)
    trust_weight = compute_trust_weight(expert_scores, score_range, correlation_matrix)
    
    # Generate the Copula distribution for the Expert
    P_expert_tensor = generate_copula_probability_tensor(
        scores=expert_scores,
        certainties=expert_certainties,
        score_range=score_range,
        sigma_multiplier=sigma_multiplier,
        correlation_matrix=correlation_matrix
    )
    
    # Apply the exact FPD fractional update rule: V = n + W_i * P_{E_i}
    updated_memory = memory_tensor + (trust_weight * P_expert_tensor)
    
    return updated_memory

# Posterior on phone brands
    
def posterior_on_brands(
    feature_score_preference: np.ndarray,
    global_brand_score: np.ndarray, 
    posterior_feature_tensor: np.ndarray, # Shape: (brands, f1, f2, ..., fn)
    tolerance_distance: float
) -> np.ndarray:
    """
    Calculates the posterior probability of each brand by summing the discrete 
    probability mass that strictly falls within the accepted hyper-rectangle.
    """
    num_of_brands = posterior_feature_tensor.shape[0]
    
    # Calculate the normalized prior on brands P(B)
    brand_prior = calculate_global_brand_preference(global_brand_score)
    brand_posterior = np.zeros(num_of_brands)
    
    # Define the strict lower bounds for acceptance (ensure it doesn't drop below index 0)
    # Note: We subtract 1 because a score of '7' lives at index 6.
    thresholds = np.floor(feature_score_preference - tolerance_distance).astype(int)
    lower_bound_indices = np.maximum(thresholds - 1, 0)
    
    # Create the N-Dimensional slice object dynamically
    # For a 4D feature space with thresholds [7, 5, 8, 4], this creates:
    # (slice(6, None), slice(4, None), slice(7, None), slice(3, None))
    feature_slices = tuple(slice(idx, None) for index, idx in enumerate(lower_bound_indices))
    
    # Integrate (sum) the accepted probability mass for each brand
    for b in range(num_of_brands):
        # Extract the specific brand's feature tensor
        brand_tensor = posterior_feature_tensor[b]
        
        # Slice out the accepted block and sum the probability mass: P(S \in Accept | B=b)
        accepted_mass = np.sum(brand_tensor[feature_slices])
        
        # Apply Bayes Rule: P(B=b | S \in Accept) \propto P(S \in Accept | B=b) * P(B=b)
        brand_posterior[b] = accepted_mass * brand_prior[b]
        
    # Normalize the final posterior across all brands
    if np.sum(brand_posterior) > 0:
        brand_posterior = brand_posterior / np.sum(brand_posterior)
        
    return brand_posterior




    



        






    



    







def generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_array, score_range, initial_feature_weight):
    """
    Generates weights for the primary modeler's scores, adjusted by opinion certainty.

    Parameters:
    primary_modeler_scores (np.ndarray): Scores given by the primary modeler for different brands and features.
    opinion_certainty_array (np.ndarray): Array representing the certainty of the primary modeler's opinions.
    score_range (int): Highest possible score for each feature.
    initial_feature_weight (float): Initial feature weight for the Dirichlet distribution.

    Returns:
    np.ndarray: A weight table adjusted by the primary modeler's scores and opinion certainty.
    """

    # Initialize a weight table filled with the initial feature weight
    # Shape: (number of features, score_range, number of brands)
    weight_table = np.full((primary_modeler_scores.shape[1], score_range, primary_modeler_scores.shape[0]), initial_feature_weight, dtype=np.float32)
    
    # Create boolean masks to identify non-zero certainty and non-zero scores
    non_zero_certainty = opinion_certainty_array != 0
    non_zero_scores = primary_modeler_scores != 0

    # Combine masks to identify positions where both conditions are met
    mask = non_zero_certainty[:, np.newaxis] & non_zero_scores

    # Find indices where the combined mask is true
    i_indices, j_indices = np.where(mask)

    # Adjust scores to be 0-based (subtract 1 from each score)
    scores = primary_modeler_scores[mask] - 1

    # Update the weight table with the adjusted scores and opinion certainty
    weight_table[j_indices, scores, i_indices] += opinion_certainty_array[i_indices]

    return weight_table


def generate_expert_weights(expert_scores, score_range, initial_feature_weight):
    """
    Generates weights for expert scores, adjusting by a given initial feature weight.

    Parameters:
    expert_scores (np.ndarray): Scores given by experts for different brands and features.
    score_range (int): Highest possible score for each feature.
    initial_feature_weight (float): Initial feature weight for the Dirichlet distribution.

    Returns:
    np.ndarray: A weight table adjusted by the expert scores.
    """

    # Initialize a weight table filled with the initial feature weight
    # Shape: (number of features, score_range, number of experts)
    weight_table = np.full((expert_scores.shape[1], score_range, expert_scores.shape[0]), initial_feature_weight, dtype=np.int8)
    
    # Create a boolean mask to identify non-zero scores
    non_zero_scores = expert_scores != 0

    # Find indices where the scores are non-zero
    i_indices, j_indices = np.nonzero(non_zero_scores)
    
    # Adjust scores to be 0-based (subtract 1 from each score)
    scores = expert_scores[non_zero_scores] - 1
    
    # Update the weight table with the adjusted scores
    # Add 1 to the weight table at the positions indicated by the indices
    weight_table[j_indices, scores, i_indices] += 1

    return weight_table


def compute_primary_modeler_posterior_brands(primary_modeler_opinion_features, primary_modeler_brand_pref, primary_modeler_score_preference):
    """
    Computes the primary modeler's posterior probabilities for brands based on primary modeler's scores in form of weights, brand preferences, and score preferences.

    Parameters:
    primary_modeler_opinion_features (np.ndarray): Normalized vector of primary modelers weights. Representing opinion about features.
    primary_modeler_brand_pref (np.ndarray): Distribution representing primary modeler's brand preference.
    primary_modeler_score_preference (np.ndarray): Primary modeler's score preference for each feature.

    Returns:
    tuple: A tuple containing:
        - primary_modeler_posterior_brands (np.ndarray): Posterior probabilities for each brand.
    """

    # Initialize posterior probabilities for brands as np.ndarray of ones
    primary_modeler_posterior_brands = np.ones((primary_modeler_opinion_features.shape[2],), dtype=np.float16)

    ## * Filter scores based on preferences
    # * Testing for speed showed that vector implementation is slower than for loop, but might be faster with large datasets
    
    # Create a different reference to primary_modeler_opinion_features for better readibility
    primary_modeler_preferred_scores = primary_modeler_opinion_features
    
    # Generate an array of row indices corresponding to the scores
    # * primary_modeler_preferred_scores.shape[1] gives the number of possible scores (score range)
    # * np.arange creates an array of integers from 0 to (score range - 1)
    # * reshape(1, -1) changes the shape to (1, score range) for broadcasting purposes
    rows = np.arange(primary_modeler_preferred_scores.shape[1], dtype=np.int8).reshape(1, -1)

    # Create a mask based on the primary modeler's score preferences
    # * primary_modeler_score_preference is an array of preferred scores for each feature
    # * Subtract 1 to adjust for 0-based indexing (e.g., preference of 1 means the score should be >= 0 in 0-based indexing)
    # * Reshape the preferences to (number of features, 1) to align for broadcasting
    # * rows[:, None] expands the dimensions to (1, score range, 1)
    # * The comparison checks if each score (rows) is less than the preferred score minus 1
    mask = rows[:, None] < (primary_modeler_score_preference - 1)[:, None].reshape(-1, 1)

    # Transpose the mask to align dimensions correctly
    # * The mask has shape (number of features, score range, 1) after the previous step
    # * Transpose to (score range, number of features, 1)
    mask = mask.transpose(1, 2, 0)

    # Broadcast the mask to match the shape of primary_modeler_preferred_scores
    # * The original mask shape is (score range, number of features, 1)
    # * Expand to (number of brands, score range, number of features)
    # * primary_modeler_preferred_scores has shape (number of brands, score range, number of features)
    mask = np.broadcast_to(mask, (primary_modeler_preferred_scores.shape[0], primary_modeler_preferred_scores.shape[1], primary_modeler_preferred_scores.shape[2]))
    
    # Apply the mask to set the scores below preference to zero
    primary_modeler_preferred_scores[mask] = 0

    # Calculate the maximum probabilities for preferred scores
    max_probabilities_in_preference_matrix = np.max(primary_modeler_preferred_scores, axis=1)
    
    # Calculate the product of maximum probabilities across all features
    max_probabilities_in_preference_matrix_product = np.prod(max_probabilities_in_preference_matrix, axis=0)
    
    # Update the posterior probabilities for brands by primary modellers brand preference
    primary_modeler_posterior_brands = max_probabilities_in_preference_matrix_product * primary_modeler_brand_pref
    
    # Normalize the posterior probabilities to sum to 1
    primary_modeler_posterior_brands = primary_modeler_posterior_brands / np.sum(primary_modeler_posterior_brands)
    
    return primary_modeler_posterior_brands



def simulated_example(primary_modeler_scores, opinion_certainty_array, apply_certainty, number_of_responders,
                      trust_matrix, primary_modeler_score_preference, primary_modeler_brand_pref, score_range, initial_feature_weight, 
                      Brand_1_expert_opinions, Brand_2_expert_opinions, Brand_3_expert_opinions):
    """
    Simulates the primary modeler's posterior preferences by combining initial scores with expert opinions.

    Parameters:
    primary_modeler_scores (np.ndarray): Scores given by the primary modeler for different brands and features.
    opinion_certainty_array (np.ndarray): Array representing the certainty of the primary modeler's opinions.
    apply_certainty (bool): Flag indicating whether to apply the certainty weights.
    number_of_responders (np.ndarray): Number of responders for each brand.
    trust_matrix (np.ndarray): Matrix representing the trust levels for each expert's opinion. Possible values: interval [0,1].
    score_preference (np.ndarray): Primary modeler's score preference for each feature.
    primary_modeler_brand_pref (np.ndarray): Distribution representing primary modeler's brand preference.
    score_range (int): Highest possible score for each feature.
    initial_feature_weight (float): Initial feature weight for the Dirichlet distribution.
    Brand_1_expert_opinions (np.ndarray): Expert opinions for Brand_1.
    Brand_2_expert_opinions (np.ndarray): Expert opinions for Brand_2.
    Brand_3_expert_opinions (np.ndarray): Expert opinions for Brand_3.

    Returns:
    tuple: A tuple containing:
        - primary_modeler_posterior_initial (np.ndarray): Initial posterior probabilities for the primary modeler.
        - primary_modeler_posterior_updated (np.ndarray): Updated posterior probabilities for the primary modeler after considering expert opinions.
    """

    if apply_certainty:
        # Apply certainty by scaling the opinion certainty array with the number of responders
        opinion_certainty_array = opinion_certainty_array * number_of_responders
    else:
        # Ensure that the opinion certainty array is ones (equivalent to no certainty being applied)
        opinion_certainty_array = np.ones_like(opinion_certainty_array, dtype=np.float16)
        
    
    # Generate expert opinion weights for Brand_1 experts
    # ! Note 1: Ensure data type is float to correctly apply trust
    Brand_1_expert_opinion_weights = generate_expert_weights(Brand_1_expert_opinions, score_range, 0)

    # Generate expert opinion weights for Brand_2 experts
    # ! Note 1: Ensure data type is float to correctly apply trust
    Brand_2_expert_opinion_weights = generate_expert_weights(Brand_2_expert_opinions, score_range, 0)
                
    # Generate expert opinion weights for Brand_3 experts
    # ! Note 1: Ensure data type is float to correctly apply trust
    Brand_3_expert_opinion_weights = generate_expert_weights(Brand_3_expert_opinions, score_range, 0)
    
    ## * Apply trust matrix to expert opinion weights for Brand_1

    # Create an array of expert indices for Brand_1
    # This is essentially a range of numbers from 0 to the number of experts - 1
    i_indices = np.arange(Brand_1_expert_opinion_weights.shape[2], dtype=np.int32)

    # Multiply the expert opinion weights by the corresponding trust values for Brand_1
    # * Note 1: Brand_1_expert_opinion_weights is a 3D array where the third dimension represents different experts
    # * trust_matrix[0, i_indices] selects the trust values for Brand_1 experts
    # * This applies the trust factor to each expert's weights
    Brand_1_expert_opinion_weights_trust = Brand_1_expert_opinion_weights[:, :, i_indices] * trust_matrix[0, i_indices]

    ## * Apply trust matrix to expert opinion weights for Brand_2

    # Create an array of expert indices for Brand_2
    i_indices = np.arange(Brand_2_expert_opinion_weights.shape[2], dtype=np.int32)

    # Multiply the expert opinion weights by the corresponding trust values for Brand_2
    # * Note 1: Brand_2_expert_opinion_weights is a 3D array where the third dimension represents different experts
    # * trust_matrix[1, i_indices] selects the trust values for Brand_2 experts
    # * This applies the trust factor to each expert's weights
    Brand_2_expert_opinion_weights_trust = Brand_2_expert_opinion_weights[:, :, i_indices] * trust_matrix[1, i_indices]

    ## * Apply trust matrix to expert opinion weights for Brand_3

    # Create an array of expert indices for Brand_3
    i_indices = np.arange(Brand_3_expert_opinion_weights.shape[2], dtype=np.int32)

    # Multiply the expert opinion weights by the corresponding trust values for Brand_3
    # * Note 1: Brand_3_expert_opinion_weights is a 3D array where the third dimension represents different experts
    # * trust_matrix[2, i_indices] selects the trust values for Brand_3 experts
    # * This applies the trust factor to each expert's weights
    Brand_3_expert_opinion_weights_trust = Brand_3_expert_opinion_weights[:, :, i_indices] * trust_matrix[2, i_indices]
    
    ## Initialize cumulative expert weights
    # * Creating a 3D array to hold the cumulative weights for each brand and feature
    # * Shape: (`number of features`, score_range, `number of brands`)
    # * - The first dimension represents the features.
    # * - The second dimension (score_range) represents the possible scores for each feature.
    # * - The third dimension represents the brands: Brand_1, Brand_2, and Brand_3.
    cumulative_expert_weights = np.zeros((3, score_range, 3), dtype=np.float32)

    # Summing the trusted expert opinion weights for Brand_1 across all experts
    # and storing the result in the first slice of the cumulative weights array.
    # * Note 1: Axis 2 represents summing over all experts for each feature and score.
    cumulative_expert_weights[:, :, 0] = np.sum(Brand_1_expert_opinion_weights_trust, axis=2)

    # Summing the trusted expert opinion weights for Brand_2 across all experts
    # and storing the result in the second slice of the cumulative weights array.
    # * Note 1: Axis 2 represents summing over all experts for each feature and score.
    cumulative_expert_weights[:, :, 1] = np.sum(Brand_2_expert_opinion_weights_trust, axis=2)

    # Summing the trusted expert opinion weights for Brand_3 across all experts
    # and storing the result in the third slice of the cumulative weights array.
    # * Note 1: Axis 2 represents summing over all experts for each feature and score.
    cumulative_expert_weights[:, :, 2] = np.sum(Brand_3_expert_opinion_weights_trust, axis=2)


    ## Calculating trust and certainty weighting linear opinion pooling
    
    # Generate primary modeler's weights witj initial feature weight 0, because this weight matrix represents pure opinion
    primary_modeler_weights_linear_pool = generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_array, score_range, initial_feature_weight=0)

    # Add the certainty weight of primary modeler to the trust weights for each brand
    trust_and_opinion_certainty_weight_matrix = np.concatenate([opinion_certainty_array[:,np.newaxis], trust_matrix], axis=1)
    
    # Calculated weighted sum of expert opinions and primary modeller opinion
    #* Note 1: Weights for experts = trust_matrix
    #* Note 2: Weights for primary modeller = opinion_certainty
    
    trust_and_opinion_certainty_weight_matrix_sum = np.sum(trust_and_opinion_certainty_weight_matrix, axis=1)

    # Mix primary modeler's weights with cumulative expert weights
    linear_opinion_pool_weights = cumulative_expert_weights + primary_modeler_weights_linear_pool
    
    # Calculate linear opinion pool of expert weights and primary modellers weights
    
    linear_opinion_pool = linear_opinion_pool_weights/trust_and_opinion_certainty_weight_matrix_sum
    
    # Compute updated posterior probabilities for the primary modeler
    primary_modeler_posterior_updated_lin_pool = compute_primary_modeler_posterior_brands(linear_opinion_pool, primary_modeler_brand_pref, primary_modeler_score_preference)
    
    
    ## Calculating FPD merging of opinions 
    
    # Generate primary modeler's weights
    primary_modeler_weights_FPD = generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_array, score_range, initial_feature_weight)
    
    # Calculate primary modellers initial opinion on features
    
    # # Calculate the sum along the second dimension (axis=1) and keep the dimensions for broadcasting
    # sums = np.sum(primary_modeler_weights_FPD, axis=1, keepdims=True)

    # # Perform the division using broadcasting to normalize the weights
    # primary_modeler_opinion_features_initial = primary_modeler_weights_FPD / sums
    
    # Calculate primary modellers updated opinion on features
    
    # Update primary modeler's weights with cumulative expert weights
    primary_modeler_updated_weights = cumulative_expert_weights + primary_modeler_weights_FPD
    
    # Calculate the sum along the second dimension (axis=1) and keep the dimensions for broadcasting
    sums = np.sum(primary_modeler_updated_weights, axis=1, keepdims=True)

    # Perform the division using broadcasting to normalize the weights
    primary_modeler_opinion_features_updated = primary_modeler_updated_weights / sums
    
    # Compute initial posterior probabilities for the primary modeler
    #primary_modeler_posterior_initial = compute_primary_modeler_posterior_brands(primary_modeler_opinion_features_initial, primary_modeler_brand_pref, primary_modeler_score_preference)
    
    # Compute updated posterior probabilities for the primary modeler
    primary_modeler_posterior_updated_FPD = compute_primary_modeler_posterior_brands(primary_modeler_opinion_features_updated, primary_modeler_brand_pref, primary_modeler_score_preference)

    return primary_modeler_posterior_updated_lin_pool, primary_modeler_posterior_updated_FPD

