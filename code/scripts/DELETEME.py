import numpy as np
import pandas as pd
from statsmodels.stats.correlation_tools import corr_nearest
from scipy.stats import norm, beta, multivariate_normal
from scipy.spatial.distance import mahalanobis
import time

# score_range = 10
# alpha_dist = 6.5
# beta_dist = 5


# bin_edge_quantiles = np.array([beta.ppf(k/score_range, alpha_dist, beta_dist) for k in range(0,score_range+1)])

# print(bin_edge_quantiles)

a = [1, 2, 3]
b = [4, 5, 6]

# 1. Define your objective correlation matrix (from Higham's algorithm)
# Example: 4D matrix for Price, Battery, RAM, Camera
# Sigma = np.array([
#     [1.0, 0.2, 0.3, 0.4],
#     [0.2, 1.0, 0.1, 0.2],
#     [0.3, 0.1, 1.0, 0.5],
#     [0.4, 0.2, 0.5, 1.0]
# ]) 
# means = np.zeros(4) # The Copula latent space is STRICTLY centered at 0

# # 2. You have your continuous score bounds (e.g., mapped from discrete 1-10 scores)
# uniform_lower = np.array([0.7, 0.5, 0.3, 0.8])
# uniform_upper = np.array([0.8, 0.6, 0.4, 0.9])

# # 3. Translate uniform bounds to the latent Standard Normal bounds (Inverse CDF)
# z_lower = norm.ppf(uniform_lower)
# z_upper = norm.ppf(uniform_upper)

# # 4. The Modern Integration (SciPy > 1.9.0)
# # Instantiate a frozen multivariate normal object
# mvn_dist = multivariate_normal(mean=means, cov=Sigma)

# # Calculate the probability mass of the exact hyper-rectangle
# # 'x' serves as the upper limit, 'lower_limit' creates the bounding box
# prob = mvn_dist.cdf(x=z_upper, lower_limit=z_lower)

# print(f"Copula Probability Mass: {prob}")

# from itertools import product

# score_range = 10

# possible_scores = np.arange(1,score_range+1,1)

# possible_permutations = np.array(list(product(possible_scores, repeat=3)))

# print(possible_permutations)

# print(possible_permutations.shape)

# # Filtering scores which have score for 3 feature >= 9

# bool_mask = np.zeros_like(possible_permutations, dtype=bool)
# possible_permutations

A = [[1, 2], [3, 4], [5,6]]
B = [[6, 2], [4, 5], [1,2]]

C = zip(A,B)

print(A)
print(B)

print(list(C))










