import torch
import torch.nn.functional as F
import numpy as np
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.config import config

class FedALAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.eta = config["ala_eta"]
        self.rand_percent = config["ala_rand_percent"]
        self.layer_idx = config["ala_layer_idx"]
        self.ala_epochs = config["ala_epochs"]
        self.cluster_alpha = config.get("cluster_alpha", 0.9)

        # Store the learned aggregation weights (W) for persistence across rounds
        self.weights = None 
        self.params_length = len(list(self.task.model.parameters()))

    def init_ala_weights(self, parameters):
        if self.weights is None:
            self.weights = []
            for i, p in enumerate(parameters):
                if i >= self.params_length - self.layer_idx:
                    self.weights.append(torch.ones_like(p).to(self.device))
                else:
                    self.weights.append(None)

    def execute(self):
        """
        Main execution flow for FedALA.
        """
        # Ensure we have the global model from the server
        if "server" not in self.message_pool or "weight" not in self.message_pool["server"]:
            return 

        # --- FIX 1: Define server_msg ---
        server_msg = self.message_pool["server"]

        # 1. Get the Standard Global Model (The Anchor)
        global_anchor = server_msg["weight"]
        
        # 2. Get the Cluster Model (The Specialist)
        target_model = global_anchor # Default to global
        
        # --- FIX 2: Check 'server_msg' for flags, NOT 'target_model' ---
        if server_msg.get("is_clustered", False):
            client_map = server_msg.get("client_map", {})
            cid = int(self.client_id)
            
            if cid in client_map:
                cluster_id = client_map[cid]
                # Ensure the cluster ID exists in the model list
                if cluster_id in server_msg["cluster_models"]:
                    cluster_model = server_msg["cluster_models"][cluster_id]
                    
                    # --- NEW: SOFT CLUSTERING (INTERPOLATION) ---
                    alpha = 0.8
                    
                    interpolated_model = []
                    for p_clust, p_glob in zip(cluster_model, global_anchor):
                        # Mix them
                        mixed = alpha * p_clust + (1 - alpha) * p_glob
                        interpolated_model.append(mixed)
                    
                    target_model = interpolated_model
                    print(f"[Client {cid}] Using Soft Cluster Model (Alpha {alpha})")

                    # --- FIX 3: RESET ALA WEIGHTS ON CLUSTERING EVENT ---
                    # If this is the first time we switch to clustering, reset ALA to relearn
                    # if not hasattr(self, 'has_reset_ala'):
                    #     print(f"[Client {self.client_id}] Clustering Event Detected! Resetting ALA Weights...")
                    #     self.weights = None 
                    #     self.has_reset_ala = True
        
        # --- SELECTION LOGIC END --
        global_params = target_model
        local_params = list(self.task.model.parameters())

        # Initialize ALA weights if first run (or if reset)
        self.init_ala_weights(local_params)

        # --- Phase 1: Adaptive Local Aggregation (ALA) ---
        local_params_detached = [p.clone().detach() for p in local_params]
        self.task.model.train()
        
        # Phase 1: ALA weight learning
        for _ in range(self.ala_epochs):
            for batch_data in self.task.train_dataloader:
                if self.rand_percent < 100 and np.random.rand() > (self.rand_percent / 100.0):
                    continue
                if hasattr(batch_data, 'to'):
                    batch_data = batch_data.to(self.device)

                with torch.no_grad():
                    for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                        diff = p_glob - p_loc
                        if self.weights[i] is not None:
                            p_model.data.copy_(p_loc + diff * self.weights[i])
                        else:
                            p_model.data.copy_(p_glob)
                
                output = self.task.model(batch_data)
                if isinstance(output, (list, tuple)): output = output[0]

                if hasattr(self.task, 'criterion') and self.task.criterion is not None:
                    loss = self.task.criterion(output, batch_data.y)
                else:
                    target = batch_data.y
                    if target.dim() > 1: target = target.squeeze()
                    loss = F.cross_entropy(output, target)

                self.task.model.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                        if self.weights[i] is not None and p_model.grad is not None:
                            diff = p_glob - p_loc
                            grad_w = p_model.grad * diff
                            self.weights[i] -= self.eta * grad_w
                            self.weights[i].clamp_(0, 1)

        # --- Phase 2: Final Model Initialization ---
        with torch.no_grad():
            for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                if self.weights[i] is not None:
                    p_model.data.copy_(p_loc + (p_glob - p_loc) * self.weights[i])
                else:
                    p_model.data.copy_(p_glob)

        # --- Phase 3: Standard Local Training ---
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }