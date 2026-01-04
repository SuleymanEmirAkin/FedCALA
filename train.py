import openfgl.config as config
import sys


dataset_name = sys.argv[1]
fl_algorithm = sys.argv[2]
num_clients = int(sys.argv[3])
num_rounds = int(sys.argv[4])
usce_cuda = bool(int(sys.argv[5]))

from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "dataset"



args.dataset = [dataset_name]
args.num_clients = num_clients
args.num_rounds = num_rounds
args.use_cuda = usce_cuda


args.fl_algorithm = fl_algorithm
args.model = ["gin"]

args.metrics = ["accuracy"]


args.skew_alpha = 1


trainer = FGLTrainer(args)

trainer.train()