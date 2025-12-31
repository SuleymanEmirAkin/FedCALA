import torch
import numpy as np
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedcala.config import config

class FedCALAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedCALAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.layer_idx = config["ala_layer_idx"]
        self.epoch_count = 0
        self.warm_up = 0
        self.client_models = {}
    
    def fast_cosine_similarity(self, X):
        # 1. Compute the L2 norm of each row (N, 1)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        
        # 2. Normalize the matrix (avoid division by zero with a small epsilon)
        X_normalized = X / (norms + 1e-8)
        
        # 3. Compute the full similarity matrix using matrix multiplication
        # Shape: (N, D) @ (D, N) -> (N, N)
        return np.dot(X_normalized, X_normalized.T)
    
    def execute(self):
        self.epoch_count += 1
        all_clients = self.message_pool["sampled_clients"]
        if not all_clients: return

        # 1. Collect features
        client_features = np.array([self.message_pool[f"client_{cid}"]["features"] for cid in all_clients])
        full_sim = self.fast_cosine_similarity(client_features)
        
        # 2. Initialize storage using actual client IDs
        self.client_models = {cid: [torch.zeros_like(p) for p in self.task.model.parameters()] for cid in all_clients}
        
        num_params = len(list(self.task.model.parameters()))
        total_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in all_clients])

        with torch.no_grad():
            for idx_i, cid_i in enumerate(all_clients):
                # --- Layer-wise Aggregation ---
                for i in range(num_params):
                    if i < num_params - self.layer_idx:
                        # GLOBAL AGGREGATION for lower layers
                        for cid_j in all_clients:
                            w_j = self.message_pool[f"client_{cid_j}"]["num_samples"] / total_samples
                            p_j = self.message_pool[f"client_{cid_j}"]["weight"][i]
                            self.client_models[cid_i][i] += w_j * p_j
                    else:
                        # ADAPTIVE/CLUSTERED AGGREGATION for top layers
                        if self.epoch_count <= self.warm_up:
                            # Standard FedAvg during warm-up
                            for cid_j in all_clients:
                                w_j = self.message_pool[f"client_{cid_j}"]["num_samples"] / total_samples
                                p_j = self.message_pool[f"client_{cid_j}"]["weight"][i]
                                self.client_models[cid_i][i] += w_j * p_j
                        else:
                            # CALA weighted aggregation
                            # Calculate denominator for normalization: sum(sim_ij * n_j)
                            denom = 0
                            for idx_j, cid_j in enumerate(all_clients):
                                denom += full_sim[idx_i, idx_j] * self.message_pool[f"client_{cid_j}"]["num_samples"]
                            
                            for idx_j, cid_j in enumerate(all_clients):
                                sim_ij = full_sim[idx_i, idx_j]
                                n_j = self.message_pool[f"client_{cid_j}"]["num_samples"]
                                p_j = self.message_pool[f"client_{cid_j}"]["weight"][i]
                                
                                # Weight = (similarity * sample_count) / normalization_factor
                                weight = (sim_ij * n_j) / (denom + 1e-8)
                                self.client_models[cid_i][i] += weight * p_j

    def send_message(self):
        """ Sends specific cluster-aggregated weights to each client. """
        self.message_pool["server"] = {}
        for cid, model_params in self.client_models.items():
            self.message_pool["server"][f"weight_client_{cid}"] = model_params