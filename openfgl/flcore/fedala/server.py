import torch
from openfgl.flcore.base import BaseServer

class FedALAServer(BaseServer):
    """
    FedALAServer implements the server-side logic for FedALA.
    
    In the FedALA paper, the server-side aggregation remains a standard 
    Weighted Average based on the number of samples. The 'Adaptive' part 
    happens on the client side after downloading the global model.
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)

    def execute(self):
        """
        Executes the server-side aggregation.
        Formula: Theta_global = Sum( (n_k / n_total) * Theta_k )
        """
        with torch.no_grad():
            # 1. Calculate total samples from selected clients
            num_tot_samples = sum([
                self.message_pool[f"client_{client_id}"]["num_samples"] 
                for client_id in self.message_pool["sampled_clients"]
            ])

            # 2. Iterate through sampled clients and aggregate parameters
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                # Calculate weight coefficient for this client
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                # Fetch local model parameters uploaded by the client
                local_params = self.message_pool[f"client_{client_id}"]["weight"]
                
                # Aggregate
                for (local_param, global_param) in zip(local_params, self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
        
    def send_message(self):
        """
        Broadcasts the aggregated global model to all clients.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }