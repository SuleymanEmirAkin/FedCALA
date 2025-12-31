import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedala.config import config

class FedALAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.layer_idx = config["ala_layer_idx"]
        self.client_model = []
    
    def execute(self):
        all_clients = self.message_pool["sampled_clients"]
        if not all_clients: return

        self.client_model = [torch.zeros_like(p) for p in self.task.model.parameters()]

        num_params = len(list(self.task.model.parameters()))
        total_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in all_clients])

        with torch.no_grad():
            for i in range(num_params):
                for cid_j in all_clients:
                    w_j = self.message_pool[f"client_{cid_j}"]["num_samples"] / total_samples
                    p_j = self.message_pool[f"client_{cid_j}"]["weight"][i]
                    self.client_model[i] += w_j * p_j


    def send_message(self):
        """ Sends specific cluster-aggregated weights to each client. """
        self.message_pool["server"] = {}
        self.message_pool["server"]["weight"] = self.client_model