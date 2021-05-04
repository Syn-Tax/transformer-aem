import pandas as pd
from simpletransformers.classification import ClassificationArgs, ClassificationModel
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

wandb_config = {"epochs": 100, "train_batch_size": 4, "eval_batch_size": 4, "lr": 1e-4, "samples": 20, "max_seq_len": 256, "model": "roberta", "save": "roberta-base"}

df = pd.read_csv("data.csv")

train_df = df.iloc[:wandb_config["samples"], :]
eval_df = df.iloc[wandb_config["samples"]:, :]

model_args = ClassificationArgs()
model_args.num_train_epochs = wandb_config["epochs"]
model_args.eval_batch_size = wandb_config["eval_batch_size"]
model_args.train_batch_size = wandb_config["train_batch_size"]
model_args.wandb_project = "transformer-aes"
model_args.wandb_kwards = {"config": wandb_config}
model_args.learning_rate = wandb_config["lr"]
# model_args.max_seq_length = wandb_config["max_seq_length"]
model_args.regression = True
model_args.no_save = True

model = ClassificationModel(wandb_config["model"], wandb_config["save"], num_labels=1, args=model_args)

model.train(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)