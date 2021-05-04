import pandas as pd
# import simpletransformers

wandb_config = {"epochs": 100, "train_batch_size": 4, "eval_batch_size": 4, "lr": 1e-4, "samples": 20}

df = pd.read_csv("data.csv")

train_df = df.iloc[:wandb_config["samples"], :]
eval_df = df.iloc[wandb_config["samples"]:, :]
print(len(eval_df))