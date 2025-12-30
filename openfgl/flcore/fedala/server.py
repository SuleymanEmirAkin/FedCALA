import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from openfgl.flcore.base import BaseServer
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

class FedALAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        self.current_round = 0
        # --- Clustering Configuration ---
        self.warmup_rounds = getattr(args, "warmup_rounds", 10)       # Run standard FedALA for 10 rounds first
        self.perform_clustering = False 
        self.client_clusters = {}      # Map: client_id -> cluster_id
        self.cluster_models = {}       # Map: cluster_id -> model_parameters (list of tensors)
        self.num_clusters = 0

        self.distance_threshold = args.distance_threshold
        print(f"--- [Server] Clustering Distance Threshold set to {self.distance_threshold} ---")
    
    def visualize_gradients(self, weight_matrix, round_num):
        """Saves a 2D scatter plot with cluster coloring."""
        try:
            pca = PCA(n_components=2)
            components = pca.fit_transform(weight_matrix)
            
            plt.figure(figsize=(12, 8))
            
            client_ids = self.message_pool["sampled_clients"]
            
            # Assign colors
            colors = []
            labels_list = []
            for cid in client_ids:
                if cid in self.client_clusters:
                    cluster_id = self.client_clusters[cid]
                    colors.append(cluster_id)
                    labels_list.append(f"C{cluster_id}")
                else:
                    colors.append(-1)
                    labels_list.append("OUT")
            
            # Plot with colors
            scatter = plt.scatter(
                components[:, 0], 
                components[:, 1], 
                c=colors,
                cmap='tab10',
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidths=2
            )
            
            # Annotate
            for i, cid in enumerate(client_ids):
                plt.annotate(
                    f"{cid}\n{labels_list[i]}", 
                    (components[i, 0], components[i, 1]),
                    fontsize=10,
                    ha='center',
                    fontweight='bold'
                )
            
            plt.colorbar(scatter, label='Cluster ID', ticks=range(self.num_clusters))
            plt.title(f"Client Clustering (Round {round_num}) - {self.num_clusters} Clusters", fontsize=14)
            plt.xlabel("PCA Component 1", fontsize=12)
            plt.ylabel("PCA Component 2", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.4)
            
            filename = f"gradient_pca_round_{round_num}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"--- Visualization saved to {os.path.abspath(filename)} ---")
        except Exception as e:
            print(f"Visualization failed: {e}")

    def execute(self):
        """Modified execution: Handles both Warm-up (Global) and Clustered phases."""
        print(f"--- Server executing Round {self.current_round} ---")
        
        # 1. Check if it's time to cluster (One-time event)
        if self.current_round == self.warmup_rounds:
            print(f"--- [Round {self.current_round}] Performing Clustering... ---")
            self._perform_clustering_logic()
            self.perform_clustering = True

            # Count label distribution
            if hasattr(self.task, 'train_dataloader'):
                labels = []
                for batch in self.task.train_dataloader:
                    labels.extend(batch.y.cpu().numpy().tolist())
                from collections import Counter
                label_dist = Counter(labels)
                print(f"  Label distribution: {dict(label_dist)}")

        with torch.no_grad():
            if not self.perform_clustering:
                self._standard_aggregation()
            else:
                self._clustered_aggregation()

        self.current_round += 1

    def _standard_aggregation(self):
        """Original FedAvg/FedALA aggregation logic"""
        num_tot_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in self.message_pool["sampled_clients"]])
        
        for p in self.task.model.parameters():
            p.data.zero_()

        for cid in self.message_pool["sampled_clients"]:
            weight = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot_samples
            local_params = self.message_pool[f"client_{cid}"]["weight"]
            
            for (global_param, local_param) in zip(self.task.model.parameters(), local_params):
                global_param.data += weight * local_param

    def _clustered_aggregation(self):
        """
        Hybrid Aggregation:
        1. Updates Cluster Models (for cluster clients).
        2. Updates Global Model (for outliers, using EVERYONE's data).
        """
        # --- DEBUG: Verify we are running Hybrid ---
        print("--- [Server] Running Hybrid Aggregation ---")
        
        # 1. Prepare Zero Models
        for c_id in self.cluster_models:
             for p in self.cluster_models[c_id]: p.data.zero_()
        
        # Global accumulators for the outlier fallback model
        global_accumulators = [torch.zeros_like(p) for p in self.task.model.parameters()]
        
        # 2. Calculate Totals
        cluster_samples = {c_id: 0 for c_id in self.cluster_models}
        total_global_samples = 0
        
        for cid in self.message_pool["sampled_clients"]:
            n_samples = self.message_pool[f"client_{cid}"]["num_samples"]
            total_global_samples += n_samples
            
            if cid in self.client_clusters:
                c_id = self.client_clusters[cid]
                cluster_samples[c_id] += n_samples

        # 3. Aggregate
        for cid in self.message_pool["sampled_clients"]:
            local_params = self.message_pool[f"client_{cid}"]["weight"]
            n_samples = self.message_pool[f"client_{cid}"]["num_samples"]
            
            # A. Update Global Model (Everyone contributes!)
            global_weight = n_samples / total_global_samples
            for i, p_local in enumerate(local_params):
                global_accumulators[i].data += global_weight * p_local
            
            # B. Update Cluster Model (Only if member)
            if cid in self.client_clusters:
                c_id = self.client_clusters[cid]
                if cluster_samples[c_id] > 0:
                    cluster_weight = n_samples / cluster_samples[c_id]
                    for i, p_local in enumerate(local_params):
                        self.cluster_models[c_id][i].data += cluster_weight * p_local

        # 4. Save Global Model
        for i, p_global in enumerate(self.task.model.parameters()):
            p_global.data.copy_(global_accumulators[i])

    def _perform_clustering_logic(self):
        """CFL: Hierarchical Clustering on Client Gradients/Weights"""
        client_ids = self.message_pool["sampled_clients"]
        flat_weights = []
        
        global_params = list(self.task.model.parameters())

        for cid in client_ids:
            local_params = self.message_pool[f"client_{cid}"]["weight"]
            
            delta_list = []
            for p_local, p_global in zip(local_params[-2:], global_params[-2:]):
                delta = p_local.detach().cpu().numpy() - p_global.detach().cpu().numpy()
                delta_list.append(delta.flatten())
            
            flat_weights.append(np.concatenate(delta_list))
            
        weight_matrix = np.array(flat_weights)

        # Visual Debug
        self.visualize_gradients(weight_matrix, self.current_round)

        # Clustering Logic
        sim_matrix = cosine_similarity(weight_matrix)
        dist_matrix = np.maximum(1 - sim_matrix, 0)

        clustering = AgglomerativeClustering(
            n_clusters=None, 
            metric='precomputed', 
            linkage='average',
            distance_threshold=self.distance_threshold  # ← INCREASE THIS
        )
        labels = clustering.fit_predict(dist_matrix)

        # ===== FIXED FILTERING LOGIC =====
        counts = Counter(labels)
        print(f"--- DEBUG: Raw Cluster Sizes: {dict(counts)} ---")
        
        # Create mapping: old_label -> new_label (only for valid clusters)
        valid_clusters = {label: size for label, size in counts.items() if size >= 1}
        label_mapping = {old_label: new_id for new_id, old_label in enumerate(valid_clusters.keys())}
        
        print(f"--- DEBUG: Valid Clusters (size >= 2): {valid_clusters} ---")
        print(f"--- DEBUG: Label Mapping: {label_mapping} ---")
        
        # Assign clients to clusters
        self.client_clusters = {}
        for idx, label in enumerate(labels):
            real_client_id = client_ids[idx]
            
            if label in label_mapping:
                # This client is in a valid cluster
                new_cluster_id = label_mapping[label]
                self.client_clusters[real_client_id] = new_cluster_id
                print(f"✓ Client {real_client_id}: Assigned to Cluster {new_cluster_id} (original label {label}, size {counts[label]})")
            else:
                # This client is an outlier
                print(f"✗ Client {real_client_id}: OUTLIER (original label {label}, size {counts[label]})")

        # --- Statistics ---
        self.num_clusters = len(valid_clusters)
        cluster_ratio = len(self.client_clusters) / len(client_ids)
        
        print("\n" + "="*50)
        print(f"CLUSTERING SUMMARY (Round {self.current_round})")
        print("="*50)
        print(f"Total Clients: {len(client_ids)}")
        print(f"Clustered: {len(self.client_clusters)} ({cluster_ratio:.1%})")
        print(f"Outliers: {len(client_ids) - len(self.client_clusters)}")
        print(f"Number of Clusters: {self.num_clusters}")
        print("="*50 + "\n")

        # Initialize Cluster Models
        current_global = list(self.task.model.parameters())
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = [p.clone().detach() for p in current_global]

    def send_message(self):
        """Sends appropriate model to clients."""
        standard_weight = list(self.task.model.parameters())
        
        self.message_pool["server"] = {
            "weight": standard_weight,           # Fallback / Warmup
            "is_clustered": self.perform_clustering,
            "cluster_models": self.cluster_models, 
            "client_map": self.client_clusters   
        }