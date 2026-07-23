import numpy as np

def test_resulting_posterior(output_array, test_array, delta=1e-10):
    """
    Compares two numpy arrays for equality within a certain tolerance.

    # Parameters: 
    output_array (np.ndarray): The first array to compare.
    test_array (np.ndarray): The second array to compare.
    delta (float, optional): The maximum allowed difference for the comparison. Default is 1e-10.

    Returns:
    np.ndarray: A boolean array where each element indicates whether the corresponding elements
                of the input arrays are equal within the given delta.

    Raises:
    TypeError: If either of the input arrays is not of type np.ndarray.
    """

    # Check if both input arrays are of type np.ndarray
    if not isinstance(output_array, np.ndarray) or not isinstance(test_array, np.ndarray):
        raise TypeError("Both input arrays must be of type np.ndarray.")
    
    # Perform element-wise comparison within the delta tolerance
    return np.abs(output_array - test_array) <= delta


## Section 1: No opinion certainty and no trust in expert opinion

# Experiment 1.1: Initial and updated posterior distributions without trust and certainty
primary_modeler_posterior_old_experiment_1_1 = np.array([0.4975369458128079, 0.4975369458128079, 0.004926108374384236])
primary_modeler_posterior_updated_experiment_1_1 = np.array([0.4164588528678305, 0.3333333333333333, 0.2502078137988363])

# Experiment 1.2: Initial and updated posterior distributions without trust and certainty
primary_modeler_posterior_old_experiment_1_2 = np.array([0.4975369458128079, 0.4975369458128079, 0.004926108374384236])
primary_modeler_posterior_updated_experiment_1_2 = np.array([0.33560988469495, 0.3287802306100998, 0.3356098846949502])

# Experiment 1.3: Initial and updated posterior distributions without trust and certainty
primary_modeler_posterior_old_experiment_1_3 = np.array([0.4975369458128079, 0.4975369458128079, 0.004926108374384236])
primary_modeler_posterior_updated_experiment_1_3 = np.array([0.3287802306100998, 0.3356098846949502, 0.33560988469495007])


## Section 2: Inclusion of trust

# Experiment 2.1: Initial and updated posterior distributions with inclusion of trust
primary_modeler_posterior_old_experiment_2_1 = np.array([0.4975369458128079, 0.4975369458128079, 0.004926108374384236])
primary_modeler_posterior_updated_experiment_2_1 = np.array([0.2881936482488426, 0.4601860306195987, 0.2516203211315587])

# Experiment 2.2: Initial and updated posterior distributions with inclusion of trust
primary_modeler_posterior_old_experiment_2_2 = np.array([0.4975369458128079, 0.4975369458128079, 0.004926108374384236])
primary_modeler_posterior_updated_experiment_2_2 = np.array([0.35024741398019626, 0.23634733513892422, 0.4134052508808795])


## Section 3: Inclusion of certainty

# Experiment 3.1: Initial and updated posterior distributions with inclusion of certainty
primary_modeler_posterior_old_experiment_3_1 = np.array([0.00000101144236453323, 0.991187969173845, 0.008811019383790657])
primary_modeler_posterior_updated_experiment_3_1 = np.array([0.19971084151150445, 0.47452130713417084, 0.32576785135432473])

# Experiment 3.2: Initial and updated posterior distributions with inclusion of certainty
primary_modeler_posterior_old_experiment_3_2 = np.array([0.00000011611543346453, 0.00000011611543346453, 0.999999767769133])
primary_modeler_posterior_updated_experiment_3_2 = np.array([0.17952821819922996, 0.17952821819922996, 0.64094356360154])

# Experiment 3.3: Initial and updated posterior distributions with inclusion of certainty
primary_modeler_posterior_old_experiment_3_3 = np.array([0.00000010607908554925, 0.49999994696045713, 0.49999994696045713])
primary_modeler_posterior_updated_experiment_3_3 = np.array([0.5197444551999661, 0.3309885325173468, 0.14926701228268713])
