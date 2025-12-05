import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "dataset"


args.dataset = ["IMDB-MULTI"]
args.num_clients = 10


args.fl_algorithm = "fedala"
args.model = ["gin"]

args.metrics = ["accuracy"]


args.skew_alpha = 1



trainer = FGLTrainer(args)

trainer.train()