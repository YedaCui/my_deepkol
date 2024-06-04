from deep_kolmogorov import modeling, pdes, trainer
import torch

config = {
  "bs": 120000,
  "checkpoint": True,
  "factor": 6,
  "gpus": 1,
  "levels": 4,
  "lr": 0.01,
  "lr_decay": 0.25,
  "lr_decay_patience": 2,
  "min_lr": 1e-08,
  "mode": "avg_bs_basket",
  "n_iterations": 30,
  "n_test_batches": 1,
  "n_train_batches": 2000,
  "net": "DeepONet",
  "norm_layer": "layernorm",
  "num_depth": 7,
  "num_width": 75,
  "opt": "adamw",
  "pde": "BSbasket",
  "seed": 0,
  "size_t_x_u": [
    1,
    10,
    13
  ],
  "unfreeze": "all",
  "unfreeze_patience": 1,
  "weight_decay": 0.01
}

mytrainer = trainer.Trainer(config)
path = "/home/ycui/Documents/my_deepkol/exp/avg_bs_basket_2023_10_31_17_27_14/Trainer_2023-10-31_17-27-14/Trainer_ad10d_00005_5_num_depth=7,num_width=75,seed=0_2023-10-31_17-27-14/checkpoint_000016/model.pth"
# mytrainer.net.PI_layers(torch.randn(1,10,1, device=torch.device("cuda"))) # add it when loading checkpoint with the input shape same as the experiment
mytrainer.load_checkpoint(path)

scores_test = mytrainer._test_greeks_loop(greeks=[ "vega"], method="finidiff", d= 0.05)