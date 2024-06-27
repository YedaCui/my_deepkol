import os
import random
import torch
import ray
from ray import tune # for parallelly tunning the hyperparameters.
from datetime import datetime
from argparse import ArgumentParser
from .modeling import Metrics, KolmogorovNet, NETS, NORMLAYERS
from .pdes import HYPERCUBES, PDES

OPTIMIZERS = {
    "adamw": lambda params, lr, weight_decay: torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay
    ),
    "sgd": lambda params, lr, weight_decay: torch.optim.SGD(
        params, lr=lr, momentum=0.9, weight_decay=weight_decay
    ),
}


def compatibility(config):
    """
    Backward compatibility to previous versions.
    """
    if "weight_decay" not in config:
        config["weight_decay"] = 0.01
    if "decay" in config:
        config["lr_decay"] = config.pop("decay")
    if "decay_patience" in config:
        config["lr_decay_patience"] = config.pop("decay_patience")
    # for k in ["net", "pde"]:
    #     config[k] = "".join(
    #         [
    #             s
    #             for word in config[k].split("_")
    #             for s in [word[0].capitalize(), word[1:]]
    #         ]
    #     )
    return config


class Trainer(tune.Trainable):
    """
    Tune trainer.
    """

    def setup(self, config):
        # backward compatibility
        config = compatibility(config)
        # determinism
        if "seed" in config:
            self._set_seed(config["seed"])
        # model
        pde_kwargs = (
            {"hypercubes": HYPERCUBES[config["hypercubes"]]}
            if "hypercubes" in config
            else {}
        )
        self.pde = PDES[config["pde"]](**pde_kwargs)
        self.net = NETS[config["net"]](self.pde.dim_flat, config)
        self.model = KolmogorovNet(self.net, self.pde)
        self.num_net_params = self.net.get_num_params()
        # cuda
        if torch.cuda.is_available() and config["gpus"] > 0:
            self.model = torch.nn.DataParallel(self.model)
            self.model.to("cuda")
        # optimizer
        self.opt = OPTIMIZERS[config["opt"]](
            self.net.params_groups, lr=config["lr"], weight_decay=config["weight_decay"]
        )
        # metrics
        self.train_metr = Metrics()
        self.test_metr = Metrics()
        self.val_metr = Metrics()
        # data
        self.train_loader = self.pde.dataloader(config["bs"], config["n_train_batches"], 'train')
        self.test_loader = self.pde.dataloader(config["bs"], config["n_test_batches"], 'test')
        self.val_loader = self.pde.dataloader(config["bs"], config["n_test_batches"], 'val')
        # stats
        # first_scores_test = self._test_loop()
        try:
            first_scores_test = self._test_loop()
        except RuntimeError as e:
            if "expected scalar type Float but found Double" in str(e):
                print("Caught type mismatch error: converting model to float64")
                self.model.to(torch.float64)
                # Optionally, you may want to re-run the test loop after conversion
                first_scores_test = self._test_loop()
            else:
                # Raise the error again if it's not the specific type mismatch error
                raise e

        first_scores_val = self._val_loop()
        self.initial_stats = {
            "params": self.num_net_params,
            "test_initial": first_scores_test["current"],
            "val_initial": first_scores_val["current"],
        }

    @staticmethod
    def _set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def step(self):
        # training loop
        self.net.update_active_groups(self.iteration)
        self.net.unfreeze_only_active()
        lr_groups = [group["lr"] for group in self.net.params_groups]
        train_scores = self._train_loop()
        self.net.decay_lr(self.iteration)
        test_scores = self._test_loop()
        val_scores = self._val_loop()
        return {
            "val": val_scores,
            "test": test_scores,
            "train": train_scores,
            "initial_stats": self.initial_stats,
            "lr_groups": lr_groups,
            "iter": self.iteration,
        }

    def _train_loop(self):
        # training
        self.model.train()
        # zero running metrics
        self.train_metr.zero()
        for batch in self.train_loader:
            # forward and back propagation
            self.opt.zero_grad()
            output = self.model.forward(batch)
            loss = self.train_metr.store(output, return_loss="mse")
            loss.backward()
            self.opt.step()
        return self.train_metr.finalize()

    def _test_loop(self):
        # test
        self.model.eval()
        # zero running metrics
        self.test_metr.zero()
        with torch.no_grad():
            for batch in self.test_loader:
                # forward and metrics
                output = self.model.forward(batch, train=False)
                self.test_metr.store(output)
        return self.test_metr.finalize()
    
    def _test_greeks_loop(self, greeks=["delta"], method="autodiff", d=0.001):
        # test greeks
        self.test_greeks_metr = {_g:Metrics() for _g in greeks}
        self.model.eval()
        for batch in self.test_loader:
            if isinstance(self.model, torch.nn.DataParallel):
                output = self.model.module.test_greeks(batch, greeks=greeks, method=method, d=d)
            else:
                output = self.model.test_greeks(batch, greeks=greeks, method=method, d=d)
            for _g in greeks:
                self.test_greeks_metr[_g].store({"pde": output["pde"][_g], "net": output["net"][_g]})
        return {
            _g: self.test_greeks_metr[_g].finalize() for _g in greeks
        }
    
    def _val_loop(self):
        # test
        self.model.eval()
        # zero running metrics
        self.val_metr.zero()
        with torch.no_grad():
            for batch in self.val_loader:
                # forward and metrics
                output = self.model.forward(batch, train=False)
                self.val_metr.store(output)
        return self.val_metr.finalize()

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.net.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        if torch.cuda.is_available() and self.config["gpus"] > 0:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=device))



# never used
def get_args():
    parser = ArgumentParser(description="DL Kolmogorov")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus per trial")
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=HYPERCONFIGS.keys(),
        help="choose between hyperparamter search and single run with different seeds",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the experiment")
    parser.add_argument(
        "--checkpoint",
        default=False,
        action="store_true",
        help="save checkpoint at the end",
    )
    parser.add_argument(
        "--pde",
        type=str,
        default="BlackScholes",
        choices=PDES.keys(),
        help="choose the underlying PDE",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="MultilevelNet",
        choices=NETS.keys(),
        help="choose the normalization layer",
    )
    parser.add_argument(
        "--norm_layer",
        type=str,
        default="layernorm",
        choices=NORMLAYERS.keys(),
        help="choose the neural network architecture",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="adamw",
        choices=OPTIMIZERS.keys(),
        help="choose the optimizer",
    )
    parser.add_argument("--bs", default=65536, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument(
        "--min_lr", default=1e-8, type=float, help="threshold for learning rate"
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument(
        "--lr_decay",
        default=0.4,
        type=float,
        help="decay for the learning rate each iteration",
    )
    parser.add_argument(
        "--lr_decay_patience",
        default=0.4,
        type=float,
        help="number of iterations to next decay",
    )
    parser.add_argument(
        "--unfreeze",
        default="all",
        type=str,
        choices=["sequential", "single", "all"],
        help="how to unfreeze the model",
    )
    parser.add_argument(
        "--unfreeze_patience",
        default=5,
        type=int,
        help="number of iterations to next unfreeze",
    )
    parser.add_argument(
        "--levels", default=4, type=int, help="number of levels for the model"
    )
    parser.add_argument(
        "--factor",
        default=6,
        type=int,
        help="scaling factor for the input dimension of the model",
    )
    parser.add_argument(
        "--n_iterations", default=20, type=int, help="number of total iterations"
    )
    parser.add_argument(
        "--n_train_batches", default=1000, type=int, help="gradient steps per iteration"
    )
    parser.add_argument(
        "--n_test_batches",
        default=150,
        type=int,
        help="number of batches for the evaluation",
    )
    parser.add_argument(
        "--resume_exp", default=None, type=str, help="experiment name to resume"
    )
    return parser


def stopper_factory(metrics, thresholds, modes):
    # when to stop the traning process.
    def stopper(trial, result):
        for metric, threshold, mode in zip(metrics, thresholds, modes):
            value = result
            for metric_key in metric.split("/"):
                value = value[metric_key]
            if (mode == "max" and value >= threshold) or (
                mode == "min" and value <= threshold
            ):
                return True
        return False

    return stopper


HYPERCONFIGS = {
    "avg_bs": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BlackScholes",
        # "net": "MultilevelNet",
        "net": "DeepONet",
        "norm_layer": "batchnorm",
        "opt": "adamw",
        # "bs": 65536,
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "levels": 4,
        "factor": 5,
        "n_iterations": 15,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,1,2],
        # "num_depth" : tune.grid_search([4,5,6]),
        # "num_width" : tune.grid_search([45,55])
        "num_width" : tune.grid_search([35,55,75]),
        "num_depth" : tune.grid_search([5,7]),
    },
    "avg_bs_r": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSr",
        "net": "DeepONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,1,3],
        "num_width" : tune.grid_search([35,55,75]),
        # "num_depth" : 7,
        "num_depth" : tune.grid_search([5,7]),
    },
    "avg_bs_lookback": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSlookback",
        "net": "DeepONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 150,
        "size_t_x_u": [1,2,2],
        "num_width" : tune.grid_search([35,55,75]),
        "num_depth" : 5,
        # "num_width" : 55
    },
        "avg_bs_asian": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSasian",
        "net": "DeepONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,2,3],
        "num_width" : tune.grid_search([35,55,75]),
        "num_depth" : tune.grid_search([5,7]),
        # "num_width" : 55
    },
    "avg_bs_basket": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSbasket",
        "net": "DeepONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,10,13],
        "num_depth" : tune.grid_search([5,7]),
        "num_width" : tune.grid_search([35, 55, 75])
    },
        "avg_bs_basket_PI": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSbasket",
        "net": "DeepONetwithPI",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,50,13],
        "num_depth" : tune.grid_search([5,7]),
        "num_width" : tune.grid_search([35, 55, 75]),
        # "num_width" : tune.grid_search([75]),
        "num_assets" : 10,
        "pi_layer" :  tune.grid_search([[70,70],[100,100]])
    },
    "avg_bs_TI": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSTI",
        "net": "DeepKernelONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,1,3],
        "num_width" : tune.grid_search([35,55,75]),
        "num_depth" : tune.grid_search([5,7]),
        "in_channels": 2, # number of time inhomogeneoust parameters
        "num_timepoints": 20, # number of time points of the TI parameters
        "num_outputs": 6, # the output dimension of the embedding net
        "out_channels": 4,
        "kernel_size": 15,
    },
    "avg_bs_basket_TI": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSbasketTI",
        "net": "DeepKernelONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,10,13],
        "num_width" : tune.grid_search([35,55,75]),
        "num_depth" : tune.grid_search([5,7]),
        "in_channels": 11, # number of time inhomogeneoust parameters
        "num_timepoints": 20, # number of time points of the TI parameters
        "num_outputs": 6, # the output dimension of the embedding net
        "out_channels": 4,
        "kernel_size": 15,
    },
    "avg_bs_basket_TI_PI": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSbasketTI",
        "net": "DeepKernelONetwithPI",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,10,13],
        "num_width" : tune.grid_search([35,55,75]),
        "num_depth" : tune.grid_search([5,7]),
        "in_channels": 11, # number of time inhomogeneoust parameters
        "num_timepoints": 20, # number of time points of the TI parameters
        "num_outputs": 6, # the output dimension of the embedding net
        "out_channels": 6,
        "kernel_size": 15,
        "num_assets" : 10,
        "pi_layer" : [100,100]
    },
        "avg_MJD": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "MJD",
        "net": "DeepONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,1,6],
        "num_width" : tune.grid_search([35,55,75, 95, 115, 135, 155, 175, 195, 215]),
        # "num_depth" : 7,
        "num_depth" : tune.grid_search([5,7,9]),
    },
    "avg_MJDbasket": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "MJDbasket",
        "net": "DeepONet",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [1,10,35],
        "num_width" : tune.grid_search([35,55,75, 95, 115, 135, 155, 175, 195, 215]),
        "num_depth" : tune.grid_search([5,7,9]),
    },
    "avg_bs_r_expmlp": {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSr",
        "net": "ExpMLP",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "input_dim": 10,
        "hidden_dim" : tune.grid_search([50,100,300]),
    },
    "demo_0": {
        "seed": tune.grid_search([0]),
        "checkpoint": True, # whether to save net parameters.
        "pde": "BSr", # underlying model
        "net": "DeepONet", # which neural network
        "opt": "adamw", # optimizer type
        "bs": 120000, # batch size 
        "lr": 0.01, # initial learning rate
        "min_lr": 1e-8, # minimal learning rate 
        "lr_decay": 0.25, # decay parameter of learning rate
        "lr_decay_patience": 2, # decay parameter of learning rate
        "weight_decay": 0.01, # decay parameter of learning rate
        "unfreeze": "all", # decay parameter of learning rate
        "unfreeze_patience": 1, # decay parameter of learning rate
        "n_iterations": 30, # Num of iterations
        "n_train_batches": 2000,  # Num of traing batches
        "n_test_batches": 1, # Num of testing batch
        "size_t_x_u": [1,1,3], # input dim for network
        "num_width" : tune.grid_search([35,55,75]), # width of NN and use ray.tune.grid_search to tune the hyperparameters.
        "num_depth" : tune.grid_search([5,7]), # depths of NN and use ray.tune.grid_search to tune the hyperparameters.
    },
}


def main(config):
    if not config["mode"] == "default":
        config.update(HYPERCONFIGS[config["mode"]])
    if config.get("sched"):
        sched = tune.schedulers.ASHAScheduler(
            metric="val/current/L1",
            mode="min",
            max_t=config["n_iterations"],
            grace_period=config["n_iterations"] // 3,
        )
    else:
        sched = None
    if "num_samples" in config:
        num_samples = config.pop("num_samples")
    else:
        num_samples = 1
    if "stopper" in config:
        stopper = config.pop("stopper")
    else:
        stopper = {"training_iteration": config["n_iterations"]}

    print("Configuration:", config)
    ray.init(dashboard_host="0.0.0.0", num_gpus=torch.cuda.device_count())
    base_dir = os.path.join(os.getcwd(), "exp")
    if isinstance(config["resume_exp"], str):
        local_dir = os.path.join(base_dir, config["resume_exp"])
    else:
        local_dir = os.path.join(
            base_dir, "{}_{:%Y_%m_%d_%H_%M_%S}".format(config["mode"], datetime.now())
        )
    analysis = tune.run(
        Trainer, # 要运行的训练函数/类 输入为一个字典config包含超参信息
        local_dir=local_dir, # 各个hyperpapa下结果保存的路径
        scheduler=sched, # none in my case
        stop=stopper, # 如何停止，在我的case下为iteration的次数
        resources_per_trial={"gpu": config["gpus"]},  # 每次试验分配的资源
        num_samples=num_samples, 
        checkpoint_at_end=config["checkpoint"], # 是否保存checkpoints
        checkpoint_freq=1, # 保存checkpoints频率
        config=config, # 超参搜索空间，参考HYPERCONFIGS字典中不同case下的设置。
        resume=bool(config.pop("resume_exp")), # 恢复功能，我没用到。
    )
    ray.shutdown()
    return analysis