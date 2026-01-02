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
        # Retrieve the model specifically intended for this client's cluster
        if f"weight" not in self.message_pool["server"]:
            return 

        global_params = self.message_pool["server"]["weight"]
        local_params = list(self.task.model.parameters())
        self.init_ala_weights(local_params)

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
                
                target = batch_data.y
                if target.dim() > 1: target = target.squeeze()
                loss = F.cross_entropy(output, target)

                self.task.model.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                        if self.weights[i] is not None and p_model.grad is not None:
                            grad_w = p_model.grad * (p_glob - p_loc)
                            self.weights[i] -= self.eta * grad_w
                            self.weights[i].clamp_(0, 1)

        # Phase 2: Apply ALA for local training initialization
        with torch.no_grad():
            for i, (p_loc, p_glob, p_model) in enumerate(zip(local_params_detached, global_params, self.task.model.parameters())):
                if self.weights[i] is not None:
                    p_model.data.copy_(p_loc + (p_glob - p_loc) * self.weights[i])
                else:
                    p_model.data.copy_(p_glob)

        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }