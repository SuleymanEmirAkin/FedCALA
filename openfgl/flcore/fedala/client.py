import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.config import config


class FedALAClient(BaseClient):
    """
    FedALAClient implements the Adaptive Local Aggregation logic.
    
    Paper Ref: "FedALA: Adaptive Local Aggregation for Personalized Federated Learning" (AAAI-23)
    
    Key Steps:
    1. Download Global Model.
    2. ALA Phase: Learn weights (W) to merge Global and Local models.
       Formula: Local_New = Local_Old + (Global - Local_Old) * W
    3. Local Training Phase: Train the merged model normally.
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        # Hyperparameters from args
        self.eta = config["ala_eta"]
        self.rand_percent = config["ala_rand_percent"]
        self.layer_idx = config["ala_layer_idx"]
        self.ala_epochs = config["ala_epochs"]
        
        # Store the learned aggregation weights (W) for persistence across rounds
        self.weights = None 
        
        # Used to identify how many layers to apply ALA to
        self.params_length = len(list(self.task.model.parameters()))

    def init_ala_weights(self, parameters):
        """
        Initialize the ALA weights (W) to 1.0. 
        Only initializes for the top 'p' layers (defined by layer_idx).
        """
        if self.weights is None:
            self.weights = []
            for i, p in enumerate(parameters):
                # We only learn weights for the last 'layer_idx' layers
                if i >= self.params_length - self.layer_idx:
                    self.weights.append(torch.ones_like(p).to(self.device))
                else:
                    # For lower layers, weight is effectively 1 (Overwriting)
                    # We store None to save memory and skip computation
                    self.weights.append(None)

    def execute(self):
        """
        Main execution flow for FedALA.
        """
        # Ensure we have the global model from the server
        if "weight" not in self.message_pool["server"]:
            return 

        global_params = self.message_pool["server"]["weight"]
        local_params = list(self.task.model.parameters())

        # Initialize ALA weights if first run
        self.init_ala_weights(local_params)

        # --- Phase 1: Adaptive Local Aggregation (ALA) ---
        # We only run ALA if it's not the very first round (need history)
        # However, OpenFGL initializes clients; assuming t > 0 logic here.
        
        # Clone local params to keep 'Theta_i^{t-1}' safe while we train weights
        # We need detached clones to avoid graph issues during manual update
        local_params_detached = [p.clone().detach() for p in local_params]
        
        # Temporarily switch model to training mode for ALA weight learning
        self.task.model.train()
        
        # ALA Optimization Loop
        for _ in range(self.ala_epochs):
            for batch_data in self.task.train_dataloader:
                # Approximate 'sample s% of local data' by probabilistically skipping batches
                if self.rand_percent < 100 and np.random.rand() > (self.rand_percent / 100.0):
                    continue

                # Ensure data is on the correct device
                if hasattr(batch_data, 'to'):
                    batch_data = batch_data.to(self.device)

                # 1. Construct the "Merged Model" using current W
                # Theta_temp = Local + (Global - Local) * W
                # We apply this directly to the model parameters to perform forward pass
                with torch.no_grad():
                    for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                        diff = p_glob - p_loc
                        if self.weights[i] is not None:
                            # Higher layers: Use learned weights
                            p_model.data.copy_(p_loc + diff * self.weights[i])
                        else:
                            # Lower layers: Overwrite (Standard FedAvg behavior)
                            p_model.data.copy_(p_glob)
                
                # 2. Compute Gradients w.r.t Model Parameters
                output = self.task.model(batch_data)
                
                # Handle cases where model returns tuple (logits, embedding)
                if isinstance(output, (list, tuple)):
                    output = output[0]

                # FIXED: Directly use PyTorch criterion or F.cross_entropy
                if hasattr(self.task, 'criterion') and self.task.criterion is not None:
                    loss = self.task.criterion(output, batch_data.y)
                else:
                    # Direct fallback for Graph Classification
                    # Ensure targets are 1D for CrossEntropy
                    target = batch_data.y
                    if target.dim() > 1:
                        target = target.squeeze()
                    loss = F.cross_entropy(output, target)

                # FIXED: Use model.zero_grad() instead of task.optimizer.zero_grad()
                # We just need to clear model gradients to calculate gradients for W
                self.task.model.zero_grad()
                loss.backward()
                
                # 3. Update Weights (W) manually
                # Chain rule: dL/dW = dL/dTheta_model * dTheta_model/dW
                # Theta_model = Local + (Global - Local) * W
                # dTheta_model/dW = (Global - Local)
                # Therefore: dL/dW = p_model.grad * (Global - Local)
                with torch.no_grad():
                    for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                        if self.weights[i] is not None and p_model.grad is not None:
                            diff = p_glob - p_loc
                            grad_w = p_model.grad * diff
                            
                            # Update W: W = W - eta * grad
                            self.weights[i] -= self.eta * grad_w
                            
                            # Clip weights to [0, 1] as per paper
                            self.weights[i].clamp_(0, 1)

        # --- Phase 2: Final Model Initialization ---
        # Apply the final trained weights to set the starting point for local training
        with torch.no_grad():
            for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                diff = p_glob - p_loc
                if self.weights[i] is not None:
                    p_model.data.copy_(p_loc + diff * self.weights[i])
                else:
                    p_model.data.copy_(p_glob)

        # --- Phase 3: Standard Local Training ---
        # Now train the model parameters Theta using standard optimizer provided by the task
        self.task.train()

    def send_message(self):
        """
        Sends updated model and sample count to server.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }