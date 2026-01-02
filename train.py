import openfgl.config as config
import sys


dataset_name = sys.argv[1]
fl_algorithm = sys.argv[2]

from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "dataset"



args.dataset = [dataset_name]
args.num_clients = 10
args.num_rounds = 5


args.fl_algorithm = fl_algorithm
args.model = ["gin"]

args.metrics = ["accuracy"]


args.skew_alpha = 1


trainer = FGLTrainer(args)

trainer.train()