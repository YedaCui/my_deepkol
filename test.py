from deep_kolmogorov import modeling, pdes, trainer
import torch
config={
  "bs": 120000,
  "checkpoint": True,
  "factor": 6,
  "gpus": 1,
  "levels": 4,
  "lr": 0.01,
  "lr_decay": 0.25,
  "lr_decay_patience": 2,
  "min_lr": 1e-08,
  "mode": "avg_bs_r",
  "n_iterations": 30,
  "n_test_batches": 1,
  "n_train_batches": 2000,
  "net": "DeepONet",
  "norm_layer": "layernorm",
  "num_depth": 7,
  "num_width": 75,
  "opt": "adamw",
  "pde": "BSr",
  "seed": 0,
  "size_t_x_u": [
    1,
    1,
    3
  ],
  "unfreeze": "all",
  "unfreeze_patience": 1,
  "weight_decay": 0.01
}

mytrainer = trainer.Trainer(config)
path = "/home/ycui/Documents/my_deepkol/exp/avg_bs_r_2023_11_08_21_47_16/Trainer_2023-11-08_21-47-16/Trainer_5467d_00005_5_num_depth=7,num_width=75,seed=0_2023-11-08_21-47-16/checkpoint_000023/model.pth"
mytrainer.load_checkpoint(path)

scores_test = mytrainer._test_loop()
print(scores_test["current"])

mytrainer.pde.hypercubes["kappa"] = pdes.Hypercube(interval=[1.2,1.6])

mytrainer.test_loader = mytrainer.pde.dataloader(120000, 1, 'test')

scores_test = mytrainer._test_loop()
print(scores_test["current"])