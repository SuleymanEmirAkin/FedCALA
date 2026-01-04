import torch
import numpy as np
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedcala.config import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap  # pip install umap-learn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import json
import os

class FedCALAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedCALAServer, self).__init__(
            args, global_data, data_dir, message_pool, device
        )
        self.dataset_name = args.dataset
        self.num_clients = args.num_clients
        self.all_rounds_similarity = []  # Persistent list for the whole training

        self.layer_idx = config["ala_layer_idx"]
        self.epoch_count = 0
        self.warm_up = 0
        self.client_models = {}

        # New storage for visualization
        self.history = []  # List of dicts: {"round": r, "cid": id, "features": feat}
        self.kl_history = []
        self.prev_dist = None

    def save_evolution_graph(self):
        base_name = f"{self.dataset_name}_clients{self.num_clients}"

        for f in ["features", "params"]:
            if len(self.history) < 10:
                print("Not enough data to visualize yet.")
                return

            # 1. Prepare Data
            rounds = np.array([h["round"] for h in self.history])
            cids = np.array([h["cid"] for h in self.history])
            feats = np.stack([h[f] for h in self.history])

            # Scale features for better projection
            feats_scaled = StandardScaler().fit_transform(feats)

            # 2. Dimensionality Reduction with UMAP for a "cleaner" look
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(feats_scaled)

            # 3. Create the Visualization
            plt.figure(figsize=(12, 10), dpi=300)
            sns.set_style("whitegrid", {"axes.grid": True, "grid.color": ".95"})

            # Plot trajectories (the lines connecting a client's movement)
            unique_clients = np.unique(cids)
            for cid in unique_clients:
                mask = cids == cid
                plt.plot(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    color="gray",
                    alpha=0.2,
                    lw=1,
                    zorder=1,
                )

            # Plot the points with a temporal color map (Cool to Warm)
            scatter = plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=rounds,
                cmap="coolwarm",
                s=40,
                alpha=0.8,
                edgecolors="none",
                zorder=2,
            )

            # 4. Aesthetics and Labels
            cb = plt.colorbar(scatter, shrink=0.8)
            cb.set_label("Training Round", fontsize=12)
            cb.outline.set_visible(False)

            plt.title(
                f"Temporal Evolution of Client Similarities (Round {self.epoch_count})",
                fontsize=15,
                pad=20,
            )
            plt.xlabel("Similarity Dimension 1", fontsize=10)
            plt.ylabel("Similarity Dimension 2", fontsize=10)

            # Despine for a cleaner look
            sns.despine(left=True, bottom=True)

        # 5. Save to File with the new naming convention
            filename = f"evolution_{f}_{base_name}.png"
            plt.tight_layout()
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
            print(f"Visualization saved to {filename}")

    def fast_cosine_similarity(self, X):
        # 1. Compute the L2 norm of each row (N, 1)
        norms = np.linalg.norm(X, axis=1, keepdims=True)

        # 2. Normalize the matrix (avoid division by zero with a small epsilon)
        X_normalized = X / (norms + 1e-8)

        # 3. Compute the full similarity matrix using matrix multiplication
        # Shape: (N, D) @ (D, N) -> (N, N)
        return np.dot(X_normalized, X_normalized.T)

    def extract_top_layers(self, weights):
        """
        weights: list of torch tensors (client model parameters)
        returns: 1D numpy vector of top-layer parameters
        """
        top_params = weights[-self.layer_idx :]
        return torch.cat([p.flatten() for p in top_params]).detach().cpu().numpy()

    def track_similarity_drift(self, sim_matrix, temperature=0.1):
        """
        Tracks how much the client relationship structure changes.
        temperature: Lower values (< 1.0) make the distribution "sharper"
                    and the KL Divergence more sensitive.
        """
        # 1. Extract upper triangle (unique pairs)
        mask = np.triu_indices(sim_matrix.shape[0], k=1)
        flat_sim = sim_matrix[mask]

        # 2. Convert to torch tensor for stable softmax
        sim_tensor = torch.tensor(flat_sim, dtype=torch.float32)

        # 3. Apply Softmax with Temperature
        # p_i = exp(sim_i / T) / sum(exp(sim_j / T))
        p = F.softmax(sim_tensor / temperature, dim=0).numpy()

        if self.prev_dist is not None:
            # Check if length matches (important if client sampling is dynamic)
            if len(p) == len(self.prev_dist):
                epsilon = 1e-10
                kl_div = np.sum(p * np.log((p + epsilon) / (self.prev_dist + epsilon)))
                self.kl_history.append(kl_div)

                # Diagnostic print
                print(f"Round {self.epoch_count} | KL Drift: {kl_div:.6f}")
                self.save_kl_plot()
            else:
                print(
                    "Warning: Client count changed. Skipping KL drift for this round."
                )

        self.prev_dist = p

    def save_kl_plot(self):
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, len(self.kl_history) + 1),
            self.kl_history,
            marker="o",
            color="#2ca02c",
            label="KL Divergence",
        )
        plt.title("Similarity Distribution Drift (Stability)")
        plt.xlabel("Communication Round")
        plt.ylabel("KL Div (Round t || Round t-1)")
        plt.grid(True, linestyle="--", alpha=0.6)
        filename = f"kl_drift_{self.dataset_name}_c{self.num_clients}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_full_session_json(self):
        """Saves all rounds recorded so far into one file."""
        filename = f"full_session_similarity_{self.dataset_name}_c{self.num_clients}.json"
        
        output = {
            "dataset": self.dataset_name,
            "num_clients": self.num_clients,
            "total_rounds_recorded": len(self.all_rounds_similarity),
            "history": self.all_rounds_similarity
        }

        with open(filename, 'w') as f:
            json.dump(output, f) # Removed indent=4 to keep file size smaller for long sessions
        
        # print(f"Session data updated in {filename}")

    def execute(self):
        self.epoch_count += 1
        all_clients = self.message_pool["sampled_clients"]
        if not all_clients:
            return

        # --- 1. Collect & Store Features for Viz ---
        for cid in all_clients:
            feat = self.message_pool[f"client_{cid}"]["features"]
            # We store a copy to avoid reference issues
            self.history.append(
                {
                    "round": self.epoch_count,
                    "cid": cid,
                    "features": feat.copy() if hasattr(feat, "copy") else feat,
                    "params": self.extract_top_layers(
                        self.message_pool[f"client_{cid}"]["weight"]
                    ),
                }
            )

        self.save_evolution_graph()

        # 1. Collect features
        client_features = np.array(
            [self.message_pool[f"client_{cid}"]["features"] for cid in all_clients]
        )
        full_sim = self.fast_cosine_similarity(client_features)

        # 2. Append to persistent list
        self.all_rounds_similarity.append({
            "round": self.epoch_count,
            "client_ids": [int(cid) for cid in all_clients],
            "matrix": full_sim.tolist()
        })

        # 3. Save/Overwrite the single training file
        self.save_full_session_json()

        self.track_similarity_drift(full_sim)

        # 2. Initialize storage using actual client IDs
        self.client_models = {
            cid: [torch.zeros_like(p) for p in self.task.model.parameters()]
            for cid in all_clients
        }

        num_params = len(list(self.task.model.parameters()))
        total_samples = sum(
            [self.message_pool[f"client_{cid}"]["num_samples"] for cid in all_clients]
        )

        with torch.no_grad():
            for idx_i, cid_i in enumerate(all_clients):
                # --- Layer-wise Aggregation ---
                for i in range(num_params):
                    if i < num_params - self.layer_idx:
                        # GLOBAL AGGREGATION for lower layers
                        for cid_j in all_clients:
                            w_j = (
                                self.message_pool[f"client_{cid_j}"]["num_samples"]
                                / total_samples
                            )
                            p_j = self.message_pool[f"client_{cid_j}"]["weight"][i]
                            self.client_models[cid_i][i] += w_j * p_j
                    else:
                        # ADAPTIVE/CLUSTERED AGGREGATION for top layers
                        if self.epoch_count <= self.warm_up:
                            # Standard FedAvg during warm-up
                            for cid_j in all_clients:
                                w_j = (
                                    self.message_pool[f"client_{cid_j}"]["num_samples"]
                                    / total_samples
                                )
                                p_j = self.message_pool[f"client_{cid_j}"]["weight"][i]
                                self.client_models[cid_i][i] += w_j * p_j
                        else:
                            # CALA weighted aggregation
                            # Calculate denominator for normalization: sum(sim_ij * n_j)
                            denom = 0
                            for idx_j, cid_j in enumerate(all_clients):
                                denom += (
                                    full_sim[idx_i, idx_j]
                                    * self.message_pool[f"client_{cid_j}"][
                                        "num_samples"
                                    ]
                                )

                            for idx_j, cid_j in enumerate(all_clients):
                                sim_ij = full_sim[idx_i, idx_j]
                                n_j = self.message_pool[f"client_{cid_j}"][
                                    "num_samples"
                                ]
                                p_j = self.message_pool[f"client_{cid_j}"]["weight"][i]

                                # Weight = (similarity * sample_count) / normalization_factor
                                weight = (sim_ij * n_j) / (denom + 1e-8)
                                self.client_models[cid_i][i] += weight * p_j

    def send_message(self):
        """Sends specific cluster-aggregated weights to each client."""
        self.message_pool["server"] = {}
        for cid, model_params in self.client_models.items():
            self.message_pool["server"][f"weight_client_{cid}"] = model_params
