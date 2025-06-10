
# Project Overview

## Description

This code accompanies the paper Targeted Merging of Customers’ Opinions Based on Trust and executes the merging of the primary modeller’s opinions with those of the experts.

The objective is to design experiments in a way that allows the brand selection process to be inferred from the experts’ setup and additionally to test edge cases that might indicate potential underperformance of the proposed methodology.

# Simulated examples 

To simplify the analysis, specific parameters are defined for the experts, which include the following:

 - 10 experts for each brand. 
 - 3 mobile brands.
 - 3 features.
 - 6 score values.

These parameters have been deliberately chosen to configure the system and demonstrate the functionality of the proposed solution.

The parameters associated with the modeller are as follows:

 - The preference score $r_{I,n}$ = 4 for all features.
 - The modeller's brand $P_{I}(B)$ preference follows a uniform distribution.
 - The modeller's opinion on brands $b\in{B}$ is established in the experimental setup.
 - The modeller's certainty $c_{I,b,n} \in \langle 0, 1 \rangle$ is established in the experimental setup.
 - The modeller's trust $t_{I,E_{i}} \in \langle 0, 1 \rangle$ is established in the experimental setup.

The preference score is set for simplicity, eliminating one hyperparameter to tune. 
The modeller's brand preference $P_{I}(B)$ is configured to ensure that the choice of the brand $b\in{B}$ is not influenced by the modeller's bias.

The expert opinions are manually setup in the main.ipynb file.

## Files
- **main.ipynb**: Jupyter Notebook containing code, explanations, and analysis.
- **auxiliary.py**: Auxiliary functions used in the execution of the opinion merging.
- **testing.py**: Testing with chosen saved results.

## How to Use
1. Open `main.ipynb` in Jupyter Notebook.
2. Run each cell sequentially to execute the analysis.
3. The parameters of each experiment can be modified for testing of different scenarios.

## Requirements
Ensure the following dependencies are installed:

1. Pip users: ``` pip install -r requirements.txt ```
2. Conda users: ```conda env create -f environment.yml```


## Author
Jurij Ružejnikov
