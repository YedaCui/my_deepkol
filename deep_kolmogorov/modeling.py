import math
import time
import torch
from torch import nn
from typing import List, Tuple

EPSILON = 1e-08
NORMLAYERS = {
    "layernorm": torch.nn.LayerNorm,
    "batchnorm": nn.BatchNorm1d,
    "none": nn.Identity,
}


class BaseNet(torch.nn.Module):
    """
    Base class for different networks.
    """

    def __init__(self, dim_in, config):
        super().__init__()
        self.dim_in = dim_in
        self.config = config
        self.params_groups = [{"params": self.parameters()}]
        self.active_groups = []

    def unfreeze_only_active(self):
        for group in self.params_groups:
            for param in group["params"]:
                if group in self.active_groups:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def update_active_groups(self, iteration):
        idx = iteration // self.config["unfreeze_patience"]
        if idx < len(self.params_groups):
            if self.config["unfreeze"] == "single":
                self.active_groups = [self.params_groups[idx]]
            elif self.config["unfreeze"] == "sequential":
                self.active_groups = self.params_groups[: idx + 1]
            else:
                self.active_groups = self.params_groups
        else:
            self.active_groups = self.params_groups

    def decay_lr(self, iteration):
        if not (iteration + 1) % self.config["lr_decay_patience"]:
            for params_group in self.active_groups:
                if params_group["lr"] > self.config["min_lr"]:
                    params_group["lr"] *= self.config["lr_decay"]

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class DenseNet(nn.Module):
    """
    The feed forward neural network
    """

    def __init__(self, num_layers: List[int]):
        super(DenseNet, self).__init__()
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(num_layers[i],
                eps=1e-6,
                momentum=0.99)
            for i in range(len(num_layers)-1)])
            
        self.dense_layers = nn.ModuleList([nn.Linear(num_layers[i-1], num_layers[i])
                             for i in range(1, len(num_layers))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        for i in range(len(self.dense_layers)):
            x = self.bn_layers[i](x)
            x = self.dense_layers[i](x)
            x = torch.relu(x)
        return x


class DeepONet(BaseNet):
    """
    The deepOnet, The arguments are hidden layers of brunch and trunk net
    brunch_layer: The list of hidden sizes of trunk nets;
    trunk_layer: The list of hidden sizes of trunk nets
    """

    def __init__(self, dim_in, config):
        super().__init__(dim_in, config)
        self.size_t, self.size_x, self.size_u = self.config["size_t_x_u"]
        self.branch = DenseNet([self.size_u] + [self.config["num_width"]] * self.config["num_depth"])
        self.trunk = DenseNet([self.size_t + self.size_x] + [self.config["num_width"]] * self.config["num_depth"])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        The input of state can be either 3-dim or 4-dim but once fixed a problem the
        dimension of the input tensor is fixed.
        """
        time_tensor, state_tensor, u_tensor = tensor[:, 0:self.size_t], tensor[:, self.size_t:self.size_x+self.size_t], tensor[:, self.size_x+self.size_t:]
        br = self.branch(u_tensor)
        tr = self.trunk(torch.cat([time_tensor, state_tensor], -1))
        value = torch.sum(br * tr, dim=-1, keepdim=True)
        return value


class PermutationInvariantLayer(nn.Module):
    def __init__(self, num_outputs):
        super(PermutationInvariantLayer, self).__init__()
        self.num_outputs = num_outputs
        ## valid when loading checkpoint the shape of the parameters should be same as the experiment
        self.kernel = None
        self.bias = None


    def forward(self, inputs):
        if self.kernel is None:
            _, _, in_features = inputs.shape
            self.kernel = nn.Parameter(torch.randn(in_features, self.num_outputs, device=inputs.device))
            self.bias = nn.Parameter(torch.randn(self.num_outputs, device=inputs.device))
        output = torch.tensordot(inputs, self.kernel, dims=([-1], [0])) + self.bias
        output = torch.relu(output)
        return output


class DeepONetwithPI(DeepONet):
    def __init__(self, dim_in, config):
        config["size_t_x_u"] = [config["size_t_x_u"][0], config["pi_layer"][0], config["size_t_x_u"][-1]] # update the size_t_x_u
        super().__init__(dim_in, config)
        self.num_assets = config["num_assets"]
        self.PI_layers = nn.Sequential(*[PermutationInvariantLayer(m) for m in config["pi_layer"]])
        # bin = self.PI_layers(torch.randn(1,10,1, device=torch.device("cuda"))) # add it when loading checkpoint with the input shape same as the experiment

    def reshape_state(self, state: torch.Tensor):
        batch_size, dim = state.shape
        num_markov = dim // self.num_assets
        return state.view(batch_size, self.num_assets, num_markov)

    def forward(self, tensor: Tuple[torch.Tensor]) -> torch.Tensor:
        time_tensor, state_tensor, u_tensor = tensor[:, 0:self.size_t], tensor[:, self.size_t:-self.size_u], tensor[:, -self.size_u:]
        state_tensor = self.reshape_state(state_tensor)
        state_before_pi = self.PI_layers(state_tensor)
        state_after_pi = torch.mean(state_before_pi, dim=-2)
        inputs_for_deeponet = torch.concat([time_tensor, state_after_pi, u_tensor], dim=1)
        return super().forward(inputs_for_deeponet)


class DenseOperator(nn.Module):
    def __init__(self, num_outputs):
        super(DenseOperator, self).__init__()
        self.num_outputs = num_outputs
        self.w = None
    
    def forward(self, x):
        flat_dim = x.shape[-2] * x.shape[-1]
        x = x.view(x.shape[0], flat_dim)
        if self.w == None:
            self.w = nn.Linear(flat_dim, self.num_outputs, device=x.device)
        x = nn.functional.relu(self.w(x))
        return x


class KernelOperator(DenseOperator):
    def __init__(self, in_channels, out_channels, kernel_size, num_outputs):
        super(KernelOperator, self).__init__(num_outputs)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        return super(KernelOperator, self).forward(x)
    

class DeepKernelONet(DeepONet):
    def __init__(self, dim_in, config):
        self.num_para = config["size_t_x_u"][-1] # number of all parameters
        self.in_channels = config["in_channels"] # number of time inhomogeneoust parameters
        self.num_timepoints = config["num_timepoints"] # number of time points of the TI parameters
        self.num_outputs = config["num_outputs"] # the output dimension of the embedding net
        self.total_u = self.num_para - self.in_channels + self.num_timepoints * self.in_channels # the total dims of parameters 

        config["size_t_x_u"] = [config["size_t_x_u"][0], config["size_t_x_u"][1], self.num_para - self.in_channels + self.num_outputs] # update the size_t_x_u
        super(DeepKernelONet, self).__init__(dim_in, config)
        self.kernel = KernelOperator(config["in_channels"], config["out_channels"], config["kernel_size"], config["num_outputs"])

    def forward(self, tensor: Tuple[torch.Tensor]) -> torch.Tensor:
        time_tensor, state_tensor, u_tensor = tensor[:, 0:self.size_t], tensor[:, self.size_t:-self.total_u], tensor[:, -self.total_u:]
        # embedding
        u_const, u_ti = u_tensor[:, 0:self.num_para-self.in_channels], u_tensor[:, self.num_para-self.in_channels:].reshape(u_tensor.shape[0], self.in_channels, -1)
        u_ti_after_embedding = self.kernel(u_ti)

        inputs_for_deeponet = torch.concat([time_tensor, state_tensor, u_const, u_ti_after_embedding], dim=1)
        return super(DeepKernelONet, self).forward(inputs_for_deeponet)


class DeepKernelONetwithPI(DeepONet):
    def __init__(self, dim_in, config):
        self.num_para = config["size_t_x_u"][-1] # number of all parameters
        self.in_channels = config["in_channels"] # number of time inhomogeneoust parameters
        self.num_timepoints = config["num_timepoints"] # number of time points of the TI parameters
        self.num_outputs = config["num_outputs"] # the output dimension of the embedding net
        self.total_u = self.num_para + self.num_timepoints * (self.num_para - self.in_channels) # the total dims of parameters 

        config["size_t_x_u"] = [config["size_t_x_u"][0], config["pi_layer"][0], self.num_outputs] # update the size_t_x_u
        super(DeepKernelONetwithPI, self).__init__(dim_in, config)
        self.num_assets = config["num_assets"]
        self.PI_layers = nn.Sequential(*[PermutationInvariantLayer(m) for m in config["pi_layer"]])
        # bin = self.PI_layers(torch.randn(1,10,1, device=torch.device("cuda"))) # add it when loading checkpoint with the input shape same as the experiment
        self.kernel = KernelOperator(config["in_channels"], config["out_channels"], config["kernel_size"], config["num_outputs"])

    def reshape_state(self, state: torch.Tensor):
        batch_size, dim = state.shape
        num_markov = dim // self.num_assets
        return state.view(batch_size, self.num_assets, num_markov)

    def forward(self, tensor: Tuple[torch.Tensor]) -> torch.Tensor:
        time_tensor, state_tensor, u_tensor = tensor[:, 0:self.size_t], tensor[:, self.size_t:-self.total_u], tensor[:, -self.total_u:]
        # PI net
        state_tensor = self.reshape_state(state_tensor)
        state_before_pi = self.PI_layers(state_tensor)
        state_after_pi = torch.mean(state_before_pi, dim=-2)
        # embedding
        u_const, u_ti = u_tensor[:, 0:self.num_para-self.in_channels], u_tensor[:, self.num_para-self.in_channels:].reshape(u_tensor.shape[0], self.in_channels, -1)
        u_ti_after_embedding = self.kernel(u_ti)

        inputs_for_deeponet = torch.concat([time_tensor, state_after_pi, u_const, u_ti_after_embedding], dim=1)
        return super(DeepKernelONetwithPI, self).forward(inputs_for_deeponet)


class LevelNet(nn.Module):
    """
    Network module for a single level.
    """

    def __init__(self, dim_in, dim, level, norm_layer):
        super().__init__()
        self.level = level
        self.dense_layers = nn.ModuleList([nn.Linear(dim_in, dim, bias=False)])
        self.dense_layers += [
            nn.Linear(dim, dim, bias=False) for _ in range(2 ** level - 1)
        ]
        self.dense_layers.append(nn.Linear(dim, 1))
        self.norm_layers = nn.ModuleList(
            [NORMLAYERS[norm_layer](dim, eps=EPSILON) for _ in range(2 ** level)]
        )
        self.act = nn.ReLU()

    def forward(self, tensor, res_tensors=None):
        out_tensors = []
        tensor = self.dense_layers[0](tensor)
        for i, dense in enumerate(self.dense_layers[1:]):
            tensor = self.norm_layers[i](tensor)
            tensor = self.act(tensor)
            tensor = dense(tensor)
            if res_tensors:
                tensor = tensor + res_tensors[i]
            if i % 2 or self.level == 0:
                out_tensors.append(tensor)
        return out_tensors


class MultilevelNet(BaseNet):
    """
    Multilevel net.
    """

    def __init__(self, dim_in, config):
        super().__init__(dim_in, config)
        dim = self.config["factor"] * self.dim_in
        self.nets = nn.ModuleList(
            [
                LevelNet(self.dim_in, dim, level, config["norm_layer"])
                for level in range(self.config["levels"])
            ]
        )
        self.params_groups = [{"params": net.parameters()} for net in self.nets]

    def forward(self, tensor):
        res_tensors = None
        for net in self.nets[::-1]:
            res_tensors = net(tensor, res_tensors)
        return res_tensors[-1]


class MultilevelNetNoRes(MultilevelNet):
    """
    Multilevel net without residual connections.
    """

    def __init__(self, dim_in, config):
        super().__init__(dim_in, config)

    def forward(self, tensor):
        output = self.nets[0](tensor)[-1]
        for net in self.nets[1:]:
            output += net(tensor)[-1]
        return output


class Feedforward(BaseNet):
    """
    Feedforward net.
    """

    def __init__(self, dim_in, config):
        super().__init__(dim_in, config)
        dim = self.config["factor"] * self.dim_in
        self.net = LevelNet(self.dim_in, dim, config["levels"], config["norm_layer"])

    def forward(self, tensor):
        return self.net(tensor)[-1]


NETS = {net.__name__: net for net in BaseNet.get_subclasses()}


class KolmogorovNet(torch.nn.Module):
    """
    DL Kolmogorov model.
    """

    def __init__(self, net, pde):
        super().__init__()
        self.net = net
        self.pde = pde

    def forward(self, batch, train=True):
        with torch.no_grad():
            if train:
                y = self.pde.sde(batch)
            else:
                if hasattr(self.pde, "get_rmt"):
                    y = torch.exp(- self.pde.get_rmt(0, batch["t"], batch["r0"], batch["r1"], batch["r2"])) * self.pde.solution(batch)
                else:
                    y = torch.exp(- batch["r"] * batch["t"]) * self.pde.solution(batch)
            tensor = self.pde.normalize_and_flatten(batch)
        if hasattr(self.pde, "get_rmt"):
            y_pred = torch.exp(- self.pde.get_rmt(0, batch["t"], batch["r0"], batch["r1"], batch["r2"])) * self.net.forward(tensor)
        else:
            y_pred = torch.exp(- batch["r"] * batch["t"]) * self.net.forward(tensor)
        return {"pde": y, "net": y_pred}


class Metrics:
    """
    Returns the metrics for our trainer.
    """

    names = ["mse", "L2^2", "mae", "L1", "L1_std"]

    def __init__(self):
        self.best = {name: 1.0e10 for name in self.names}
        self.last_improve = {name: 0 for name in self.names}
        self.t = 0.0
        self.steps = 0.0
        self._running = {metric: 0 for metric in self.names}
        self._count = 0
        self._current_t = time.time()

    def store(self, output, return_loss=None):
        abs_error = (output["pde"] - output["net"]).abs()
        magnitude = output["pde"].abs() + 1
        rel_error = abs_error / magnitude
        loss = {
            "mse": (abs_error ** 2).mean(),
            "L2^2": (rel_error ** 2).mean(),
            "mae": abs_error.mean(),
            "L1": rel_error.mean(),
            "L1_std": rel_error.std(),
        }
        for name in self.names:
            self._running[name] += loss[name].item()
        self._count += 1
        if return_loss:
            return loss[return_loss]

    def zero(self):
        self._running = {metric: 0 for metric in self._running}
        self._count = 0
        self._current_t = time.time()

    def finalize(self):
        current_t = time.time() - self._current_t
        self.t += current_t
        self.steps += self._count
        current = {
            "time": current_t,
            "steps": self._count,
            "overall time": self.t,
            "overall steps": self.steps,
        }
        current.update(
            {name: metr / self._count for name, metr in self._running.items()}
        )
        for name in self.names:
            if current[name] < self.best[name]:
                self.best[name] = current[name]
                self.last_improve[name] = 0
            else:
                self.last_improve[name] += 1
        current["L2"] = math.sqrt(current["L2^2"])
        return {
            "current": current,
            "best": self.best,
            "last improve": self.last_improve,
        }
