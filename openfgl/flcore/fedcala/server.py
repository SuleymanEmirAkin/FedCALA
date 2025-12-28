import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedcala.config import config

class FedCALAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedCALAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.num_clusters = config.get("num_clusters", 3)
        self.layer_idx = config["ala_layer_idx"]
        self.client_clusters = {}

    def execute(self):
        sampled_clients = self.message_pool["sampled_clients"]
        if not sampled_clients: return

        # 1. Collect features and cluster clients
        client_features = []
        for cid in sampled_clients:
            client_features.append(self.message_pool[f"client_{cid}"]["features"])
        
        # Normalize for Cosine Similarity during K-Means
        norm_features = normalize(np.array(client_features))
        kmeans = KMeans(n_clusters=min(self.num_clusters, len(sampled_clients)), random_state=42).fit(norm_features)
        
        clusters = {i: [] for i in range(kmeans.n_clusters)}
        for i, cid in enumerate(sampled_clients):
            clusters[kmeans.labels_[i]].append(cid)
            self.client_clusters[cid] = kmeans.labels_[i]

        # 2. Perform Hybrid Aggregation
        # Lower layers = Global; Top layers = Cluster-specific
        num_params = len(list(self.task.model.parameters()))
        self.cluster_models = {k: [torch.zeros_like(p) for p in self.task.model.parameters()] for k in clusters.keys()}

        with torch.no_grad():
            # Aggregate Shared Lower Layers (Global)
            total_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled_clients])
            global_lower_params = [torch.zeros_like(p) for p in self.task.model.parameters()]
            
            for cid in sampled_clients:
                weight = self.message_pool[f"client_{cid}"]["num_samples"] / total_samples
                for i, p in enumerate(self.message_pool[f"client_{cid}"]["weight"]):
                    if i < num_params - self.layer_idx:
                        global_lower_params[i] += weight * p

            # Aggregate Cluster-Specific Top Layers
            for kid, cluster_cids in clusters.items():
                cluster_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in cluster_cids])
                for cid in cluster_cids:
                    c_weight = self.message_pool[f"client_{cid}"]["num_samples"] / cluster_samples
                    for i, p in enumerate(self.message_pool[f"client_{cid}"]["weight"]):
                        if i >= num_params - self.layer_idx:
                            self.cluster_models[kid][i] += c_weight * p
                        else:
                            # Assign the global lower layers to the cluster model
                            self.cluster_models[kid][i] = global_lower_params[i]

    def send_message(self):
        """ Sends specific cluster-aggregated weights to each client. """
        self.message_pool["server"] = {}
        for cid, kid in self.client_clusters.items():
            self.message_pool["server"][f"weight_client_{cid}"] = self.cluster_models[kid]