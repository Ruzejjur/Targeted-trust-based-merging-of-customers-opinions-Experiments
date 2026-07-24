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
def normalise_score(score:np.integer, score_range:np.integer)->np.float64: 
    """
    #TODO: Add description
    """
    normlised_score = np.float64((score-0.5)/score_range)

    return normlised_score


def calculate_z_scores(arr:np.ndarray)->np.ndarray:
    """
    #TODO: Add description
    """

    z_scores = norm.ppf(arr)

    return z_scores

def mahalanobis_distance(z_scores:np.ndarray, pearson_correlation_matrix:pd.DataFrame)-> np.float64:
    """
    #TODO: Add description
    """

    inverse_corr_matrix = np.linalg.inv(pearson_correlation_matrix)

    mahalanobis_distance = mahalanobis(z_scores, np.zeros(z_scores.shape), inverse_corr_matrix)

    return mahalanobis_distance


def trust_weight(distance:np.float64)->np.float64: 
    """
    #TODO: Add description
    """

    trust_weight = np.exp(-(1/2)*distance)

    return trust_weight

# Projecting marginal scores to distributions

def parametrise_beta_dist(normalised_score: np.float64, certainty:np.float64) -> tuple[np.float64, np.float64]:
    """
    #TODO: Add description
    """

    alpha = 1+2*normalised_score*(certainty/(1-certainty))
    beta = 1+2*(1-normalised_score)*(certainty/(1-certainty))

    return alpha, beta

def v_max(score_range: np.integer, sigma_multiplier: np.float64) -> np.float64:
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

def maximum_certainty(v_max: np.float64) -> np.float64:
    """
    #TODO: Add description
    """

    max_c = 1 - 2/v_max

    return max_c

class MarginalOpinion: 
    """
    #TODO: Add description
    """

    def __init__(self, score:np.integer, certainty:np.float64, score_range:np.integer, sigma_multiplier:np.float64): 
        self.score = score
        self.certainty = certainty
        self.score_range = score_range
        self.sigma_multiplier = sigma_multiplier

        self.normlised_score = normalise_score(self.score, self.score_range)
        self.alpha, self.beta = parametrise_beta_dist(self.normlised_score, self.certainty)
        self.v_max = v_max(self.score_range, self.sigma_multiplier)
        self.max_c = maximum_certainty(self.v_max)
        self.cdf_quantiles = self._discrete_CDF_in_edge_bins(self.score_range)
        self.z_scores = calculate_z_scores(self.cdf_quantiles)

    def _discrete_CDF_in_edge_bins(self, score_range: np.integer) -> np.ndarray:
        """
        #TODO: Add description
        """

        bin_edge_quantiles = np.array([beta.ppf(k/score_range, self.alpha, self.beta) for k in range(score_range+1)])

        return bin_edge_quantiles

# Copula modelling 

def gauss_copula_probab(scores: np.ndarray, certainties: np.ndarray, score_range: np.integer, sigma_multiplier: np.float64, correlation_matrix: np.ndarray) -> np.ndarray: 
    """
    #TODO: Add description
    """

    marginal_opinions = np.array([MarginalOpinion(score, certainty, score_range, sigma_multiplier) for score, certainty in zip(scores, certainties)])
    
    marginal_z_scores = np.array([marginal_opinion.z_scores for marginal_opinion in marginal_opinions])

    z_upper = np.array([])
    z_lower = np.array([])

    for score, z_scores in zip(scores, marginal_z_scores): 

        z_upper = np.append(z_upper, z_scores[score - 1])
        z_lower = np.append(z_lower, z_scores[score + 1])

    means = np.zeros(4) # The Copula latent space is centered at 0

    # Instantiate a multivariate normal object
    mvn_dist = multivariate_normal(mean=means, cov=correlation_matrix)

    # Calculate the probability mass of the exact hyper-rectangle
    # 'x' serves as the upper limit, 'lower_limit' creates the bounding box
    prob = mvn_dist.cdf(x=z_upper, lower_limit=z_lower)

    return prob

# Posterior on phones

def generate_acceptance_region(
    feature_score_preference: np.ndarray, 
    score_range: np.integer, 
    number_of_features: np.integer,
    tolerance_distance: np.integer
) -> Iterator[np.ndarray]: 
    """
    Generates discrete state permutations that fall within the modeller's acceptance region.
    Uses a Python generator to prevent memory overflow during multi-dimensional state space traversal.
    
    Parameters:
    - feature_score_preference (np.ndarray): The target score vector (e.g., [8, 7, 9, 5]).
    - score_range (int): The maximum score value (e.g., 10).
    - number_of_features (int): The dimensionality of the state space (e.g., 4).
    - tolerance_distance (float): How far a score can deviate from the preference and still be "accepted".
    
    Yields:
    - np.ndarray: A single valid state vector that meets the acceptance criteria.
    """
    
    # Possible discrete scores (e.g., 1 through 10)
    # Note: np.arange(1, score_range) stops at score_range - 1. You must add +1.
    possible_scores = np.arange(1, score_range + 1)
    
    # Iterate through the Cartesian product without casting to a list in memory
    for state_tuple in product(possible_scores, repeat=number_of_features):
        state_vector = np.array(state_tuple)
        
        # The state must be strictly greater than or equal to the 
        # preference vector minus the allowed tolerance.
        if np.all(state_vector >= (feature_score_preference - tolerance_distance)):
            
            # Yield hands the valid state back to your FPD loop one at a time, 
            # keeping RAM usage essentially at zero.
            yield state_vector

    
def calculate_global_brand_preference(global_brand_score:np.ndarray) -> np.ndarray:
    """
    #TODO: Add description
    """

    prob = global_brand_score/np.sum(global_brand_score)

    return prob

    

def posterior_on_brands(feature_score_preference: np.ndarray,
    global_brand_score:np.ndarray, 
    num_of_brands: np.int64,    
    score_range: np.integer, 
    number_of_features: np.integer,
    tolerance_distance: np.integer):
    """
    #TODO: Add description
    """

    global_brand_preference = calculate_global_brand_preference(global_brand_score)
    feature_score_acceptance_region = generate_acceptance_region(feature_score_preference, number_of_features, score_range, tolerance_distance)

    brand_posterior = np.zeros(num_of_brands)

    # Assuming that the feature space is of shape (b,f1,f2, ..., fn)
    #TODO: Make use of boolean mask in the original code. Here we only need to sum up specific rows in a smart way. 


    


    pass

    



        






    



    

    

# Marginal score distributions on features

#TODO: Find better name for v_max
def map_score_to_distribution(scores: np.ndarray, certainty_vec: np.ndarray, score_range: np.integer, v_max: np.integer):
    """
    #TODO: Add description
    """
    normalised_scores = np.frompyfunc(normalise_score, 2, 1)(scores, score_range)

    parametrise_beta_dist(normalised_scores, certainty_vec, v_max)









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

