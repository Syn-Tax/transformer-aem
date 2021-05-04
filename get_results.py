import pandas as pd
from simpletransformers.classification import ClassificationArgs, ClassificationModel
import logging
import sklearn

logging.basicConfig(level=logging.ERROR)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

wandb_config = {"epochs": 20, "train_batch_size": 4, "eval_batch_size": 4, "lr": 5e-5, "samples": 20, "max_seq_len": 256, "model": "roberta", "save": "roberta-base"}

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
model_args.overwrite_output_dir = True
model_args.logging_steps = 1
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = 5
model_args.use_eval_cached_features = True

model = ClassificationModel(wandb_config["model"], wandb_config["save"], num_labels=1, args=model_args)

model.train_model(train_df, eval_df=eval_df, mse=sklearn.metrics.mean_squared_error, mae=sklearn.metrics.mean_absolute_error)

result, model_outputs, wrong_predictions = model.eval_model(eval_df, mse=sklearn.metrics.mean_squared_error, mae = sklearn.metrics.mean_absolute_error)