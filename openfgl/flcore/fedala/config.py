config = {
    # Learning rate for the Adaptive Local Aggregation (ALA) weight training phase
    "ala_eta": 0.1,
    
    # Number of epochs to train the aggregation weights
    "ala_epochs": 10,
    
    # Percentage of local data to randomly sample for the ALA phase (s% in the paper)
    # Default is 80%
    "ala_rand_percent": 80,
    
    "cluster_alpha": 0.9,  # Weight for cluster model interpolation (rest is global)
    # Number of layers (from the end of the model) to apply ALA.
    # Lower layers (generic features) are overwritten by global model.
    # Higher layers (specific features) are adaptively aggregated.
    "ala_layer_idx": 2
}