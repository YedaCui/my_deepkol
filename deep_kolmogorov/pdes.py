import math
from abc import ABC, abstractmethod
from operator import mul
from functools import reduce
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time


class Hypercube:
    """
    Hypercube for sampling of the input data.
    """

    def __init__(self, interval, dims=(1,)):
        self.interval = interval
        self.dims = dims

    @property
    def interval(self):
        return self.__interval

    @interval.setter
    def interval(self, value):
        if not len(value) == 2:
            raise ValueError(f"interval {value} must be of the form [a, b]")
        self.__interval = value

    @property
    def dims(self):
        return self.__dims

    @dims.setter
    def dims(self, value):
        if not isinstance(value, tuple):
            raise TypeError(f"dims {value} must be a tuple")
        self.__dims = value

    @property
    def mean(self):
        return sum(self.__interval) / 2

    @property
    def std(self):
        return (self.__interval[1] - self.__interval[0]) / math.sqrt(12)

    @property
    def dim_flat(self):
        return reduce(mul, self.__dims)

    def sample(self, batch_size):
        return torch.FloatTensor(batch_size, *self.__dims).uniform_(*self.__interval)

    def __repr__(self):
        return f'hypercube {self.__interval}^({"x".join(map(str,self.__dims))})'


class Data(Dataset):
    """
    Uniformly distributed input data as a PyTorch (infinite) dataset.
    """

    def __init__(self, hypercubes, batch_size, n_batches, get_X, get_K, get_r, get_sigma):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.hypercubes = hypercubes
        self.get_X = get_X
        self.get_K = get_K
        self.get_r = get_r
        self.get_sigma = get_sigma

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        batch = {
            key: cube.sample(self.batch_size) for key, cube in self.hypercubes.items()
        }
        if self.get_X is not None:
            batch["x"] = self.get_X(batch)
        if self.get_K is not None:
            batch["K"] = self.get_K(batch)
        if self.get_r is not None:
            batch["r"] = self.get_r(batch)
        if self.get_sigma is not None:
            batch["sigma"] = self.get_sigma(batch)
        if "x" not in batch:
            batch["x"] = batch["s"].clone()
        return batch

class Pde(ABC):
    """
    Base class for different parametrized PDEs.
    """

    def __init__(self, hypercubes):
        super().__init__()
        self.hypercubes = hypercubes

    @property
    def hypercubes(self):
        return self.__hypercubes

    @hypercubes.setter
    def hypercubes(self, value):
        if not (
            isinstance(value, dict)
            and all(isinstance(cube, Hypercube) for cube in value.values())
        ):
            raise TypeError(f"{value} must be a dictionary consisting of hypercubes")
        self.__hypercubes = value

    @property
    def dim_flat(self):
        return sum([cube.dim_flat for cube in self.__hypercubes.values()])

    def dataloader(self, batch_size, n_batches, data_type, Testset_narrow_x=False):
        if not Testset_narrow_x:
            return DataLoader(
                    Data(self.__hypercubes, batch_size, n_batches, self.get_X, self.get_K, self.get_r, self.get_sigma), batch_size=None
                )
        ### Use it when want the testset x be in [9,10] 
        if data_type == 'train': 
            return DataLoader(
                Data(self.__hypercubes, batch_size, n_batches, self.get_X, self.get_K, self.get_r, self.get_sigma), batch_size=None
            )
        else:
            return DataLoader(
                Data(self.__hypercubes, batch_size, n_batches, None, self.get_K, self.get_r, self.get_sigma), batch_size=None
            )
        ### only use for compare with berner's results ####
        # return DataLoader(
        #     Data(self.__hypercubes, batch_size, n_batches, None, None, self.get_r, None), batch_size=None
        # )

    def naf(self, batch, param):
        raise NotImplementedError

    def normalize_and_flatten(self, batch, No_normalization_and_flatten=False):
        # batch = [
        #     (batch[param] - self.__hypercubes[param].mean) / self.hypercubes[param].std
        #     for param in self.params
        # ]
        batch = [
            self.naf(batch, param, No_normalization_and_flatten) for param in self.params
        ]
        return torch.cat([tensor.flatten(start_dim=1) for tensor in batch], dim=1)

    @property
    @abstractmethod
    def params(self):
        pass

    @staticmethod
    @abstractmethod
    def _check_dims(hypercubes):
        pass

    @staticmethod
    @abstractmethod
    def sde(batch):
        pass

    @staticmethod
    @abstractmethod
    def solution(batch):
        pass

    def __repr__(self):
        return f"Parametrized {self.__class__.__name__} PDE with hypercubes {self.__hypercubes}"

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


HYPERCUBES = {
    f"basket_{d_basket}d": {
        "t": Hypercube(interval=[0.0, 1.0]),
        "x": Hypercube(interval=[9.0, 10.0], dims=(d_basket,)),
        "sigma": Hypercube(
            interval=[0.1, 0.6], dims=(d_basket, d_basket, d_basket + 1)
        ),
        "mu": Hypercube(interval=[0.1, 0.6], dims=(d_basket, d_basket + 1)),
        "K": Hypercube(interval=[10.0, 12.0]),
    }
    for d_basket in range(1, 6)
}


class Basket(Pde):
    params = ("t", "x", "sigma", "mu", "K")

    def __init__(self, hypercubes=HYPERCUBES["basket_3d"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        d = hypercubes["x"].dims[0]
        return all(
            [
                hypercubes["t"].dims == (1,),
                hypercubes["x"].dims == (d,),
                hypercubes["sigma"].dims == (d, d, d + 1),
                hypercubes["mu"].dims == (d, d + 1),
                hypercubes["K"].dims == (1,),
            ]
        )

    @staticmethod
    def sde(batch, steps=25):
        """
        Outputs batched realizations of the SDE.
        """
        batch_size, d = batch["x"].shape
        steplen = (batch["t"] / steps).flatten()
        std = torch.sqrt(steplen)
        outputs = batch["x"].clone()
        for _ in range(steps):
            dw = (
                torch.randn(
                    d, batch_size, dtype=batch["x"].dtype, device=batch["x"].device
                )
                * std
            )
            sigma_x = (
                torch.einsum("iklj, il -> ikj", batch["sigma"][:, :, :, :d], outputs)
                + batch["sigma"][:, :, :, d]
            )
            mu_x = (
                torch.einsum("ikj, ij -> ik", batch["mu"][:, :, :d], outputs)
                + batch["mu"][:, :, d]
            )
            outputs += torch.einsum("ij, i -> ij", mu_x, steplen) + torch.einsum(
                "ijk, ki -> ij", sigma_x, dw
            )
        return torch.nn.ReLU()(batch["K"] - outputs.mean(dim=1, keepdims=True))

    @staticmethod
    def solution(batch, steps=25, mc_rounds=1048576):
        """
        Outputs the MC approximated solution.
        """
        ys = []
        for t, x, sigma, mu, K in zip(
            batch["t"], batch["x"], batch["sigma"], batch["mu"], batch["K"]
        ):
            mu_t = mu[:, :-1].T
            steplen = t / steps
            std = torch.sqrt(steplen)
            outputs = x.expand(mc_rounds, -1).clone()
            for _ in range(steps):
                dw = (
                    torch.randn(mc_rounds, len(x), dtype=x.dtype, device=x.device) * std
                )
                sigma_x = (
                    torch.einsum("ijk, lj -> lik", sigma[:, :, :-1], outputs)
                    + sigma[:, :, -1]
                )
                mu_x = outputs @ mu_t + mu[:, -1]
                outputs += mu_x * steplen + torch.einsum("ijk, ik -> ij", sigma_x, dw)
            y = (torch.nn.ReLU()(K - outputs.mean(dim=1, keepdims=True))).mean(
                dim=0, keepdims=True
            )
            ys.append(y)
        return torch.cat(ys, dim=0)


def n_dist(x):
    """
    Cumulative distribution function of the standard normal distribution.
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def n_density(x):
    """
    Density function of the standard normal distribution.
    """
    return torch.exp(-(x ** 2) / 2.0) / math.sqrt(2.0 * math.pi)


HYPERCUBES["black_scholes"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "x": Hypercube(interval=[9.0, 10.0]),
    "sigma": Hypercube(interval=[0.1, 0.6]),
    "K": Hypercube(interval=[10.0, 12.0]),
}


class BlackScholes(Pde):
    params = ("t", "x", "sigma", "K")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        dw = torch.sqrt(t) * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        sde = batch["x"] * torch.exp(
            -0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        return torch.nn.ReLU()(batch["K"] - sde)

    @staticmethod
    def get_X(batch):
        """
        Outputs batched realizations of the SDE.
        The 't' is selected at grids and 'x' represents the stock price at time 0.
        """
        dw = torch.sqrt(batch["t"]) * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        sde = batch["x"] * torch.exp(
            -0.5 * batch["t"] * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        return sde
    
    @staticmethod
    def get_r(batch):
        return 0

    get_K, get_sigma = None, None

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        sigma_sqrtt = batch["sigma"] * torch.sqrt(t)
        _d = (
            -(
                torch.log(batch["x"] / batch["K"])
                + 0.5 * t * batch["sigma"] ** 2
            )
            / sigma_sqrtt
        )
        return batch["K"] * n_dist(_d + sigma_sqrtt) - batch["x"] * n_dist(_d)

    def naf(self, batch, param):
        return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std

HYPERCUBES["black_scholes_r"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0]),
    "r": Hypercube(interval=[0.005, 0.08]),
    # "r": Hypercube(interval=[0, 1e-8]),
    "sigma": Hypercube(interval=[0.1, 0.6]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}


class BSr(Pde):
    params = ("t", "x", "r", "sigma", "K")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_r"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        dw = torch.sqrt(t) * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        sde = batch["x"] * torch.exp(
             batch["r"] * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        return torch.exp(-batch["r"] * self.hypercubes['t'].interval[1]) * torch.nn.ReLU()(sde - batch["K"])

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0
        """
        dw = torch.sqrt(batch["t"]) * torch.randn(
            batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device
        )
        sde = batch["s"] * torch.exp(
            batch["r"] * batch["t"] - 0.5 * batch["t"] * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * batch["s"]
    
    get_r, get_sigma = None, None

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        sigma_sqrtt = batch["sigma"] * torch.sqrt(t)
        _d = (
            (
                torch.log(batch["x"] / batch["K"])
                + batch["r"] * t +  0.5 * t * batch["sigma"] ** 2
            )
            / sigma_sqrtt
        )
        return batch["x"] * n_dist(_d) - batch["K"] * torch.exp(-batch["r"]*t) * n_dist(_d - sigma_sqrtt)
    
    def naf(self, batch, param, No_normalization_and_flatten=False):
        if No_normalization_and_flatten:
            return batch[param]
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std
        
    

    def get_greeks(self, batch, greeks=["delta"]):
        """
        Outputs the delta of the given samples with a dict.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        S = batch["x"]
        K = batch["K"]
        r = batch["r"]
        sigma = batch["sigma"]
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * torch.sqrt(t))

        def _get_greek(greek):
            if greek == "delta":
                delta = n_dist(d1)
                return delta
            if greek == "vega":
                vega = S * n_density(d1) * torch.sqrt(t)
                return vega
        
        return {
            _v : _get_greek(_v) for _v in greeks
        }


HYPERCUBES["black_scholes_TI"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0]),
    "r0": Hypercube(interval=[0.005, 0.08]),
    "r1": Hypercube(interval=[0.001, 0.004]),
    "r2": Hypercube(interval=[0, 0.01]),
    "sigma_bar": Hypercube(interval=[0.1, 0.6]),
    "beta": Hypercube(interval=[0.01, 0.04]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}


class BSTI(Pde):
    # remember to keep the constant parmaeters in front of the TI parameters
    params = ("t", "x", "K", "r", "sigma")
    ### WARNNING ###
    # the function get_X assume the T is 1

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_TI"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())
    
    @staticmethod
    def get_rmt(t0, t1, r0, r1, r2):
        # n stands for the number of time slots
        return r0 * (t1 - t0) + 0.5 * r1 * (t1**2 - t0**2) + 1/3 * r2 * (t1**3 - t0**3)

    @staticmethod
    def get_s2mt(t0, t1, T, sigma_bar, beta):
        # n stands for the number of time slots
        beta = 2 * beta
        return sigma_bar**2 * torch.exp(-beta*T) / beta * (torch.exp(beta * t1) - torch.exp(beta * t0))

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        dw = torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        s2mt = BSTI.get_s2mt(batch["t"], self.hypercubes["t"].interval[1], self.hypercubes["t"].interval[1], batch["sigma_bar"], batch["beta"])
        rmt = BSTI.get_rmt(batch["t"], self.hypercubes["t"].interval[1], batch["r0"], batch["r1"], batch["r2"])
        sde = batch["x"] * torch.exp(
            rmt - 0.5 * s2mt + torch.sqrt(s2mt) * dw
        )
        return torch.exp(-BSTI.get_rmt(0, self.hypercubes["t"].interval[1], batch["r0"], batch["r1"], batch["r2"])) * torch.nn.ReLU()(sde - batch["K"])

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0
        """
        dw = torch.randn(
            batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device
        )
        s2mt = BSTI.get_s2mt(0, batch["t"], 1, batch["sigma_bar"], batch["beta"])
        rmt = BSTI.get_rmt(0, batch["t"], batch["r0"], batch["r1"], batch["r2"])
        sde = batch["s"] * torch.exp(
            rmt - 0.5 * s2mt + torch.sqrt(s2mt) * dw
        )
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * batch["s"]
    
    @staticmethod
    def get_r(batch):
        """
        Get the r(t_0), ..., r(t_K)
        """
        num_timepoints = 20
        ts = torch.linspace(1/num_timepoints, 1, num_timepoints, dtype=batch["s"].dtype, device=batch["s"].device).repeat(batch["s"].shape[0], 1)
        return batch["r0"] + batch["r1"] * ts + batch["r2"] * ts ** 2

    @staticmethod
    def get_sigma(batch):
        """
        Get the sigma(t_0), ..., sigma(t_K)
        """
        num_timepoints = 20
        ts = torch.linspace(1/num_timepoints, 1, num_timepoints, dtype=batch["s"].dtype, device=batch["s"].device).repeat(batch["s"].shape[0], 1)
        return batch["sigma_bar"] * torch.exp(-batch["beta"] * (1 - ts))

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        t, T = batch["t"], self.hypercubes["t"].interval[1]
        s2mt = BSTI.get_s2mt(t, T, T, batch["sigma_bar"], batch["beta"])
        rmt = BSTI.get_rmt(t, T, batch["r0"], batch["r1"], batch["r2"])
        rt, st2 = (rmt / (T-t)), s2mt / (T-t)
        mask_rt, mask_st2 = torch.isnan(rt), torch.isnan(st2)
        rt, st2 = torch.where(mask_rt, 0, rt), torch.where(mask_st2, 0, st2)
        sigma_sqrtt = torch.sqrt(st2) * torch.sqrt(T-t)
        _d = (
            (
                torch.log(batch["x"] / batch["K"])
                + rt * (T - t) +  0.5 * (T - t) * st2
            )
            / sigma_sqrtt
        )
        return batch["x"] * n_dist(_d) - batch["K"] * torch.exp(-rt*(T-t)) * n_dist(_d - sigma_sqrtt)
    
    def naf(self, batch, param):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        elif param == "r":
            return (batch[param] - self.hypercubes["r0"].mean) /  self.hypercubes["r0"].std
        elif param == "sigma":
            return (batch[param] - self.hypercubes["sigma_bar"].mean) /  self.hypercubes["sigma_bar"].std
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std


HYPERCUBES["black_scholes_lookback"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0]),
    "r": Hypercube(interval=[0.005, 0.08]),
    "sigma": Hypercube(interval=[0.1, 0.6]),
}

class BSlookback(Pde):
    params = ("t", "x", "r", "sigma")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_lookback"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        # t = self.hypercubes["t"].interval[1] - batch["t"]
        # n = 100
        # dt = t/n
        # sqrt_dt = torch.sqrt(dt)
        # dw =  torch.randn(
        #     batch["s"].shape + (n,), dtype=batch["s"].dtype, device=batch["s"].device
        # )
        # path = torch.ones(batch["s"].shape + (n+1,), dtype=batch["s"].dtype, device=batch["s"].device)
        # path[:,:,0] = batch["x"][:,0].reshape(-1,1)
        # for i in range(n):
        #     # path[:,:,i+1] = torch.abs(path[:,:,i] * (1 + batch["r"] * dt +  batch["sigma"] * sqrt_dt * dw[:,:,i]))
        #     path[:,:,i+1] = path[:,:,i] * torch.exp(batch["r"] * dt - 0.5 * dt * batch["sigma"] ** 2 +  batch["sigma"] * sqrt_dt * dw[:,:,i])
        # return torch.exp(-batch["r"] * self.hypercubes['t'].interval[1]) * torch.nn.ReLU()(path[:,:,-1] - torch.minimum(path.min(dim=2).values, batch["x"][:,1].reshape(-1,1)))
    
        t = self.hypercubes["t"].interval[1] - batch["t"]
        dw = torch.sqrt(t) * torch.randn(
            batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device
        )
        St, mt = batch["x"][:,0].reshape(-1,1), batch["x"][:,1].reshape(-1,1)
        x = St * torch.exp(
            batch["r"] * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        z = torch.rand(batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device)
        b = - torch.log(St) - torch.log(x)
        c = torch.log(St) * torch.log(x) + 0.5 * t * batch["sigma"]**2 * torch.log(z)
        mt = torch.minimum(mt, torch.exp(( -b - torch.sqrt(abs(b**2 - 4 * c))) / 2))
        return torch.exp(-batch["r"] * self.hypercubes['t'].interval[1]) * torch.nn.ReLU()(x - mt)

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0 
        """
        # n = 100
        # dt = batch["t"]/n
        # sqrt_dt = torch.sqrt(dt)
        # dw =  torch.randn(
        #     batch["s"].shape + (n,), dtype=batch["s"].dtype, device=batch["s"].device
        # )
        # path = torch.ones(batch["s"].shape + (n+1,), dtype=batch["s"].dtype, device=batch["s"].device)
        # path[:,:,0] = batch["s"]
        # for i in range(n):
        #     # path[:,:,i+1] = torch.abs(path[:,:,i] * (1 + batch["r"] * dt +  batch["sigma"] * sqrt_dt * dw[:,:,i]))
        #     path[:,:,i+1] = path[:,:,i] * torch.exp(batch["r"] * dt - 0.5 * dt * batch["sigma"] ** 2 +  batch["sigma"] * sqrt_dt * dw[:,:,i])
        # x = torch.concat([path[:,:,-1], path.min(dim=2).values], dim=1)
        # return x

        dw = torch.sqrt(batch["t"]) * torch.randn(
            batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device
        )
        x = batch["s"] * torch.exp(
            batch["r"] * batch["t"] - 0.5 * batch["t"] * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        z = torch.rand(batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device)
        b = - torch.log(batch["s"]) - torch.log(x)
        c = torch.log(batch["s"]) * torch.log(x) + 0.5 * batch["t"] * batch["sigma"]**2 * torch.log(z)
        # delta =  torch.sqrt(b**2 - 4 * c)
        # if torch.isnan(torch.exp(( -b - torch.sqrt(abs(b**2 - 4 * c))) / 2)).any().item():
        #     print("z,b,c is")
        #     print((b**2 - 4 * c)[torch.isnan(delta)])
        #     print((b**2 - 4 * c).dtype)
        #     print("finish")
        return torch.concat([x, torch.exp(( -b - torch.sqrt(abs(b**2 - 4 * c))) / 2)], dim=1)

    get_K, get_r, get_sigma = None, None, None

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        sigma_sqrtt = batch["sigma"] * torch.sqrt(t)
        St, mt = batch["x"][:,0].reshape(-1,1), batch["x"][:,1].reshape(-1,1)
        a1 = (
                torch.log(St /mt)
                + batch["r"] * t +  0.5 * t * batch["sigma"] ** 2
            ) / sigma_sqrtt
        a2 = a1 - sigma_sqrtt
        a3 = a1 - 2 * batch["r"] / batch["sigma"] * torch.sqrt(t)
        return St * n_dist(a1) - mt * torch.exp(-batch["r"]*t) * n_dist(a2) \
            - St * batch["sigma"]**2 /2/batch["r"] * (n_dist(-a1) - torch.exp(-batch["r"]*t) * (mt/St)**(2*batch["r"]/batch["sigma"]**2) * n_dist(-a3))
    
    def naf(self, batch, param):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std


HYPERCUBES["black_scholes_asian"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0]),
    "r": Hypercube(interval=[0.005, 0.08]),
    "sigma": Hypercube(interval=[0.1, 0.6]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}

class BSasian(Pde):
    params = ("t", "x", "r", "sigma", "K")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_asian"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        n = 250
        dt = t/n
        sqrt_dt = torch.sqrt(dt)
        dw =  torch.randn(
            batch["s"].shape + (n,), dtype=batch["s"].dtype, device=batch["s"].device
        )
        path = torch.ones(batch["s"].shape + (n+1,), dtype=batch["s"].dtype, device=batch["s"].device)
        path[:,:,0] = batch["x"][:,0].reshape(-1,1)
        for i in range(n):
            path[:,:,i+1] = path[:,:,i] * torch.exp(batch["r"] * dt - 0.5 * dt * batch["sigma"] ** 2 +  batch["sigma"] * sqrt_dt * dw[:,:,i])
        return torch.exp(-batch["r"] * self.hypercubes['t'].interval[1]) * torch.nn.ReLU()(
            torch.exp(batch["t"] * torch.log(batch["x"][:,1].reshape(-1,1)) + t * torch.mean(torch.log(path[:,:,1:]),dim=-1)) - batch["K"]
            )

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0
        """
        n = 250
        dt = batch["t"]/n
        sqrt_dt = torch.sqrt(dt)
        dw =  torch.randn(
            batch["s"].shape + (n,), dtype=batch["s"].dtype, device=batch["s"].device
        )
        path = torch.ones(batch["s"].shape + (n+1,), dtype=batch["s"].dtype, device=batch["s"].device)
        path[:,:,0] = batch["s"]
        for i in range(n):
            # path[:,:,i+1] = torch.abs(path[:,:,i] * (1 + batch["r"] * dt +  batch["sigma"] * sqrt_dt * dw[:,:,i]))
            path[:,:,i+1] = path[:,:,i] * torch.exp(batch["r"] * dt - 0.5 * dt * batch["sigma"] ** 2 +  batch["sigma"] * sqrt_dt * dw[:,:,i])
        x = torch.concat([path[:,:,-1], torch.exp(torch.mean(torch.log(path[:,:,1:]),dim=-1))], dim=1)
        return x

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * batch["s"]
    
    get_r, get_sigma = None, None

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        t, T = batch["t"], self.hypercubes["t"].interval[1]
        St, At = batch["x"][:,0].reshape(-1,1), batch["x"][:,1].reshape(-1,1)
        mu = (batch["r"] - batch["sigma"]**2/2) /2/T * (T-t)**2
        sig = batch["sigma"]/T * torch.sqrt((T-t)**3/3)
        d2 = (
                t/T * torch.log(At) + (1-t/T) * torch.log(St) + mu - torch.log(batch["K"])
            ) / sig
        d1 = d2 + sig
        return torch.exp(-batch["r"]*(T-t)) * (
            At ** (t/T) * St ** (1-t/T) * torch.exp(mu + sig**2 / 2) * n_dist(d1) - batch["K"] * n_dist(d2)
        )
    
    def naf(self, batch, param):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std


HYPERCUBES["black_scholes_basket"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0], dims=(10,)),
    "r": Hypercube(interval=[0.005, 0.08]),
    "sigma": Hypercube(interval=[0.1, 0.6], dims=(10,)),
    "rho": Hypercube(interval=[-0.1, 0.8]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}

class BSbasket(Pde):
    params = ("t", "x", "r", "sigma", "rho", "K")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_basket"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return True

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        #print("begin to calculate the sqrt of cov at ")
        #print(time.time())
        n = batch["sigma"].shape[-1]
        batch_size = batch["sigma"].shape[0]
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        # sqrt_cov = torch.linalg.cholesky(torch.stack([torch.full((batch["x"].shape[-1], batch["x"].shape[-1]), rho[0], dtype=batch["x"].dtype, device=batch["x"].device).fill_diagonal_(1) for rho in batch["rho"]]))
        #print("complete to calculate the sqrt of cov at ")
        #print(time.time())
        #print("begin to calculate the dw at ")
        #print(time.time())
        dw = torch.sqrt(t) * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device).unsqueeze(2)
        ).squeeze(2)
        #print("complete to calculate the dw at ")
        #print(time.time())
        #print("begin to calculate the sde at ")
        #print(time.time())
        sde = batch["x"] * torch.exp(
             batch["r"] * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        #print("complete to calculate the sde at ")
        #print(time.time())
        return torch.exp(-batch["r"] * self.hypercubes['t'].interval[1]) * torch.nn.ReLU()(torch.pow(torch.prod(sde, dim=1, keepdim=True), 1.0/sde.shape[-1]) - batch["K"])

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0
        """
        #print("begin to calculate the sqrt of cov at ")
        #print(time.time())
        n = batch["sigma"].shape[-1]
        batch_size = batch["sigma"].shape[0]
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        # sqrt_cov = torch.linalg.cholesky(torch.stack([torch.full((batch["s"].shape[-1], batch["s"].shape[-1]), rho[0], dtype=batch["s"].dtype, device=batch["s"].device).fill_diagonal_(1) for rho in batch["rho"]]))
        #print("complete to calculate the sqrt of cov at ")
        #print(time.time())
        #print("begin to calculate the dw at ")
        #print(time.time())
        dw = torch.sqrt(batch["t"]) * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device).unsqueeze(2)
        ).squeeze(2)
        #print("complete to calculate the dw at ")
        #print(time.time())
        #print("begin to calculate the sde at ")
        #print(time.time())
        sde = batch["s"] * torch.exp(
            batch["r"] * batch["t"] - 0.5 * batch["t"] * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        #print("complete to calculate the sde at ")
        #print(time.time())
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * torch.pow(torch.prod(batch["s"], dim=1, keepdim=True), 1.0/batch["s"].shape[-1])
    
    get_r, get_sigma = None, None

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        # t = self.hypercubes["t"].interval[1] - batch["t"]
        # n = batch["sigma"].shape[-1]
        # batch_size = batch["sigma"].shape[0]
        # #print("begin to calculate the sqrt of RHO at ")
        # #print(time.time())
        # RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        # RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        # #print(RHO.shape)
        # # RHO = torch.stack([torch.full((n,n), rho[0], dtype=batch["sigma"].dtype, device=batch["sigma"].device).fill_diagonal_(1) for rho in batch["rho"]])
        # #print("complete to calculate the sqrt of RHO at ")
        # #print(time.time())
        # #print("begin to calculate the sqrt of sig at ")
        # #print(time.time())
        # sig = 1/n * torch.sqrt(torch.sum(batch["sigma"].unsqueeze(2) * RHO * batch["sigma"].unsqueeze(1), (1,2))).reshape((-1,1))
        # #print("complete to calculate the sqrt of sig at ")
        # #print(time.time())
        # sigma_sqrtt = sig * torch.sqrt(t)
        # #print("begin to calculate the sqrt of F at ")
        # #print(time.time())
        # F = torch.pow(torch.prod(batch["s"], dim=1, keepdim=True), 1.0/batch["s"].shape[-1]) * torch.exp(t * (batch["r"] - 0.5 * torch.mean(batch["sigma"] ** 2, 1, keepdim=True) + 0.5 * sig ** 2))
        # #print("complete to calculate the sqrt of F at ")
        # #print(time.time())
        # _d =(
        #         torch.log(F / batch["K"])
        #          +  0.5 * t * sig ** 2
        #     ) / sigma_sqrtt
        # return torch.exp(-batch["r"]*t) * (F * n_dist(_d) - batch["K"] * n_dist(_d - sigma_sqrtt))

        t = self.hypercubes["t"].interval[1] - batch["t"]
        n = batch["sigma"].shape[-1] # the dimension of S_t
        batch_size = batch["sigma"].shape[0]
        #print("begin to calculate the sqrt of RHO at ")
        #print(time.time())
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        #print(RHO.shape)
        #print("complete to calculate the sqrt of RHO at ")
        #print(time.time())
        #print("begin to calculate the sqrt of sig at ")
        #print(time.time())
        sig_t = 1/n**2 * torch.sum(batch["sigma"].unsqueeze(2) * RHO * batch["sigma"].unsqueeze(1), (1,2)).reshape(-1,1)
        #print("complete to calculate the sqrt of sig at ")
        #print(time.time())
        sig = torch.mean(batch["sigma"]**2, dim=1, keepdim=True)
        #print("begin to calculate the sqrt of F at ")
        #print(time.time())
        F = torch.exp(torch.mean(torch.log(batch["x"]), dim=1, keepdim=True)) * torch.exp((batch["r"] - (sig - sig_t)/2 ) * t)
        #print("complete to calculate the sqrt of F at ")
        #print(time.time())
        d_p =(
                torch.log(F / batch["K"])
                 +  0.5 * t * sig_t
            ) / torch.sqrt(sig_t * t)
        return torch.exp(-batch["r"]*t) * (F * n_dist(d_p) - batch["K"] * n_dist(d_p - torch.sqrt(sig_t * t)))
    
    def naf(self, batch, param):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std


HYPERCUBES["black_scholes_basket_TI"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0], dims=(10,)),
    "r0": Hypercube(interval=[0.005, 0.08]),
    "r1": Hypercube(interval=[0.001, 0.004]),
    "r2": Hypercube(interval=[0, 0.01]),
    "sigma_bar": Hypercube(interval=[0.1, 0.6], dims=(10,)),
    "beta": Hypercube(interval=[0.01, 0.04], dims=(10,)),
    "rho": Hypercube(interval=[-0.1, 0.8]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}

class BSbasketTI(Pde):
    params = ("t", "x", "K", "rho", "r", "sigma")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_basket_TI"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return True
    
    @staticmethod
    def get_rmt(t0, t1, r0, r1, r2):
        return r0 * (t1 - t0) + 0.5 * r1 * (t1**2 - t0**2) + 1/3 * r2 * (t1**3 - t0**3)

    @staticmethod
    def get_s2mt(t0, t1, T, sigma_bar, beta):
        beta = 2 * beta
        return sigma_bar**2 * torch.exp(-beta*T) / beta * (torch.exp(beta * t1) - torch.exp(beta * t0))
    
    @staticmethod
    def get_scovmt(t0, t1, T, sigma_bar, beta):
        beta = torch.concat([beta[:,i].reshape(-1,1) + beta for i in range(beta.shape[-1])], dim=-1)
        sigma_bar = torch.concat([sigma_bar[:,i].reshape(-1,1) * sigma_bar for i in range(sigma_bar.shape[-1])], dim=-1)
        return sigma_bar * torch.exp(-beta*T) / beta * (torch.exp(beta * t1) - torch.exp(beta * t0))
        

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        #print("begin to calculate the sqrt of cov at ")
        #print(time.time())
        n = batch["s"].shape[-1]
        batch_size = batch["s"].shape[0]
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        # sqrt_cov = torch.linalg.cholesky(torch.stack([torch.full((batch["x"].shape[-1], batch["x"].shape[-1]), rho[0], dtype=batch["x"].dtype, device=batch["x"].device).fill_diagonal_(1) for rho in batch["rho"]]))
        #print("complete to calculate the sqrt of cov at ")
        #print(time.time())
        #print("begin to calculate the dw at ")
        #print(time.time())
        dw = torch.matmul(sqrt_cov, 
                                                torch.randn(batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device).unsqueeze(2)
        ).squeeze(2)
        s2mt = BSbasketTI.get_s2mt(batch["t"], self.hypercubes["t"].interval[1], self.hypercubes["t"].interval[1], batch["sigma_bar"], batch["beta"])
        rmt = BSbasketTI.get_rmt(batch["t"], self.hypercubes["t"].interval[1], batch["r0"], batch["r1"], batch["r2"])
        sde = batch["x"] * torch.exp(
            rmt - 0.5 * s2mt + torch.sqrt(s2mt) * dw
        )
        #print("complete to calculate the sde at ")
        #print(time.time())
        return torch.exp(-BSbasketTI.get_rmt(0, self.hypercubes["t"].interval[1], batch["r0"], batch["r1"], batch["r2"])) * torch.nn.ReLU()(torch.pow(torch.prod(sde, dim=1, keepdim=True), 1.0/sde.shape[-1]) - batch["K"])

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0
        """
        #print("begin to calculate the sqrt of cov at ")
        #print(time.time())
        n = batch["s"].shape[-1]
        batch_size = batch["s"].shape[0]
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        # sqrt_cov = torch.linalg.cholesky(torch.stack([torch.full((batch["s"].shape[-1], batch["s"].shape[-1]), rho[0], dtype=batch["s"].dtype, device=batch["s"].device).fill_diagonal_(1) for rho in batch["rho"]]))
        #print("complete to calculate the sqrt of cov at ")
        #print(time.time())
        #print("begin to calculate the dw at ")
        #print(time.time())
        dw = torch.matmul(sqrt_cov, 
                            torch.randn(batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device).unsqueeze(2)
        ).squeeze(2)
        s2mt = BSbasketTI.get_s2mt(0, batch["t"], 1, batch["sigma_bar"], batch["beta"])
        rmt = BSbasketTI.get_rmt(0, batch["t"], batch["r0"], batch["r1"], batch["r2"])
        sde = batch["s"] * torch.exp(
            rmt - 0.5 * s2mt + torch.sqrt(s2mt) * dw
        )
        #print("complete to calculate the sde at ")
        #print(time.time())
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * torch.pow(torch.prod(batch["s"], dim=1, keepdim=True), 1.0/batch["s"].shape[-1])
    
    @staticmethod
    def get_r(batch):
        """
        Get the r(t_0), ..., r(t_K)
        """
        num_timepoints = 20
        ts = torch.linspace(1/num_timepoints, 1, num_timepoints, dtype=batch["s"].dtype, device=batch["s"].device).repeat(batch["s"].shape[0], 1)
        return batch["r0"] + batch["r1"] * ts + batch["r2"] * ts ** 2

    @staticmethod
    def get_sigma(batch):
        """
        Get the sigma(t_0), ..., sigma(t_K)
        """
        num_timepoints = 20
        ts = torch.linspace(1/num_timepoints, 1, num_timepoints, dtype=batch["s"].dtype, device=batch["s"].device).repeat(batch["s"].shape[0], 1)
        return torch.concat(
            [batch["sigma_bar"][:,i].reshape(-1,1) * torch.exp(-batch["beta"][:,i].reshape(-1,1) * (1 - ts)) for i in range(batch["beta"].shape[1])], dim=-1)

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        n = batch["s"].shape[-1] # the dimension of S_t
        batch_size = batch["s"].shape[0]
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)

        sig_t = 1/n**2 * torch.sum(BSbasketTI.get_scovmt(batch["t"], self.hypercubes["t"].interval[1], self.hypercubes["t"].interval[1], batch["sigma_bar"], batch["beta"]) * RHO.reshape(RHO.shape[0],-1), dim=-1, keepdim=True) / t
        sig = torch.mean(BSbasketTI.get_s2mt(batch["t"], self.hypercubes["t"].interval[1], self.hypercubes["t"].interval[1], batch["sigma_bar"], batch["beta"]) / t, dim=1, keepdim=True)
        mask_sig_t, mask_sig = torch.isnan(sig_t), torch.isnan(sig)
        sig_t, sig = torch.where(mask_sig_t, 0, sig_t), torch.where(mask_sig, 0, sig)
        #print("begin to calculate the sqrt of F at ")
        #print(time.time())
        rmt = BSbasketTI.get_rmt(batch["t"], self.hypercubes["t"].interval[1], batch["r0"], batch["r1"], batch["r2"])
        F = torch.exp(torch.mean(torch.log(batch["x"]), dim=1, keepdim=True)) * torch.exp(rmt - (sig - sig_t)/2 * t)
        #print("complete to calculate the sqrt of F at ")
        #print(time.time())
        d_p =(
                torch.log(F / batch["K"])
                 +  0.5 * t * sig_t
            ) / torch.sqrt(sig_t * t)
        return torch.exp(-rmt) * (F * n_dist(d_p) - batch["K"] * n_dist(d_p - torch.sqrt(sig_t * t)))
    
    def naf(self, batch, param):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        elif param == "r":
            return (batch[param] - self.hypercubes["r0"].mean) /  self.hypercubes["r0"].std
        elif param == "sigma":
            return (batch[param] - self.hypercubes["sigma_bar"].mean) /  self.hypercubes["sigma_bar"].std
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std



HYPERCUBES["MJD"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0]),
    "r": Hypercube(interval=[0.005, 0.08]),
    "sigma": Hypercube(interval=[0.1, 0.6]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
    "lambda": Hypercube(interval=[10, 20]),
    "m": Hypercube(interval=[-0.02, 0.02]),
    "delta": Hypercube(interval=[0.01, 0.1]),
}


class MJD(Pde):
    params = ("t", "x", "r", "sigma", "lambda", "m", "delta", "K")

    def __init__(self, hypercubes=HYPERCUBES["MJD"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        dw = torch.sqrt(t) * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        muj = torch.exp(batch["m"] + 0.5 * batch["delta"] ** 2) - 1
        num_j = torch.poisson(
            t * batch["lambda"]
        )
        jumps = torch.normal(
            mean=num_j * batch["m"], std=torch.sqrt(num_j) * batch["delta"]
        )
        sde = batch["x"] * torch.exp(
            batch["r"] * t - batch["lambda"] * muj * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw + jumps
        )
        return torch.exp(-batch["r"] * self.hypercubes['t'].interval[1]) * torch.nn.ReLU()(sde - batch["K"])

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0
        """
        dw = torch.sqrt(batch["t"]) * torch.randn(
            batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device
        )
        muj = torch.exp(batch["m"] + 0.5 * batch["delta"] ** 2) - 1
        num_j = torch.poisson(
            batch["t"] * batch["lambda"]
        )
        jumps = torch.normal(
            mean=num_j * batch["m"], std=torch.sqrt(num_j) * batch["delta"]
        )
        sde = batch["s"] * torch.exp(
            batch["r"] * batch["t"] - batch["lambda"] * muj * batch["t"] - 0.5 * batch["t"] * batch["sigma"] ** 2 + batch["sigma"] * dw + jumps
        )
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * batch["s"]
    
    get_r, get_sigma = None, None

    @staticmethod
    def BS_call(r, sig, t, x, K):
        sigma_sqrtt = sig * torch.sqrt(t)
        _d = (
            (
                torch.log(x / K)
                + r * t +  0.5 * t * sig ** 2
            )
            / sigma_sqrtt
        )
        return x * n_dist(_d) - K * (torch.exp(-r*t) * n_dist(_d - sigma_sqrtt))

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        t = self.hypercubes["t"].interval[1] - batch["t"]
        muj = torch.exp(batch["m"] + 0.5 * batch["delta"] ** 2) - 1
        lambda_h = batch["lambda"] * torch.exp(batch["m"] + 0.5 * batch["delta"]**2)
        j, sig_j, r_j =0, batch["sigma"], batch["r"] - batch["lambda"] * muj
        w = torch.exp(-lambda_h*t)
        res = 0
        # for i in range(20):
        while True:
            new = self.BS_call(r_j, sig_j, t, batch["x"], batch["K"])
            if j > 100:
                print("The iteration exceeds the maximum iteration number.")
                break
            if torch.all(w * new / (res + 1e-28) < 1e-16):
                print("Finish the iteration.")
                break
            res = res + w * new
            j += 1
            w = w * lambda_h * t / j
            sig_j, r_j = torch.sqrt(batch["sigma"]**2 + j*batch["delta"]**2/t), batch["r"] - batch["lambda"] * muj + j * (batch["m"] + 0.5*batch["delta"]**2) / t
        res = torch.where(t == 0, torch.nn.ReLU()(batch["x"] - batch["K"]), res)
        return res

            
    
    def naf(self, batch, param):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std



HYPERCUBES["MJD_basket"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0], dims=(10,)),
    "r": Hypercube(interval=[0.005, 0.08]),
    "sigma": Hypercube(interval=[0.1, 0.6], dims=(10,)),
    "rho_b": Hypercube(interval=[-0.1, 0.8]),
    "lambda": Hypercube(interval=[10, 20]),
    "m": Hypercube(interval=[-0.02, 0.02], dims=(10,)),
    "delta": Hypercube(interval=[0.01, 0.1], dims=(10,)),
    "rho_j": Hypercube(interval=[-0.1, 0.8]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}

class MJDbasket(Pde):
    params = ("t", "x", "r", "sigma", "rho_b", "lambda", "m", "delta", "rho_j", "K")

    def __init__(self, hypercubes=HYPERCUBES["MJD_basket"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return True

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        # print("Begin to calculate payoff")
        t = self.hypercubes["t"].interval[1] - batch["t"]
        n = batch["sigma"].shape[-1]
        batch_size = batch["sigma"].shape[0]
        RHO = batch["rho_b"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        dw = torch.sqrt(t) * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device).unsqueeze(2)
        ).squeeze(2)

        muj = torch.exp(batch["m"] + 0.5 * batch["delta"] ** 2) - 1
        num_j = torch.poisson(
            t * batch["lambda"]
        )
        RHO = batch["rho_j"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        jumps = num_j * batch["m"] + torch.sqrt(num_j) * batch["delta"] * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device).unsqueeze(2)
        ).squeeze(2)

        sde = batch["x"] * torch.exp(
             batch["r"] * t - batch["lambda"] * muj * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw + jumps
        )
        return torch.exp(-batch["r"] * self.hypercubes['t'].interval[1]) * torch.nn.ReLU()(torch.pow(torch.prod(sde, dim=1, keepdim=True), 1.0/sde.shape[-1]) - batch["K"])

    @staticmethod
    def get_X(batch):
        """
        get the X from S_0
        """
        # print("Begin to calculate X")
        n = batch["sigma"].shape[-1]
        batch_size = batch["sigma"].shape[0]
        RHO = batch["rho_b"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        dw = torch.sqrt(batch["t"]) * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device).unsqueeze(2)
        ).squeeze(2)

        muj = torch.exp(batch["m"] + 0.5 * batch["delta"] ** 2) - 1
        num_j = torch.poisson(
            batch["t"] * batch["lambda"]
        )
        RHO = batch["rho_j"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        jumps = num_j * batch["m"] + torch.sqrt(num_j) * batch["delta"] * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device).unsqueeze(2)
        ).squeeze(2)

        sde = batch["s"] * torch.exp(
            batch["r"] * batch["t"] - batch["lambda"] * muj * batch["t"]  - 0.5 * batch["t"] * batch["sigma"] ** 2 + batch["sigma"] * dw + jumps
        )
        #print("complete to calculate the sde at ")
        #print(time.time())
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * torch.pow(torch.prod(batch["s"], dim=1, keepdim=True), 1.0/batch["s"].shape[-1])
    
    get_r, get_sigma = None, None

    @staticmethod
    def _u(tau, x, r, sig, sig_p, sig_t, K):
        print(r)
        Fp = x * torch.exp((- (sig_t**2 - sig_p**2) / 2) * tau)
        F = x * torch.exp((r - (sig_t**2 - sig**2) / 2) * tau)
        d_p = (torch.log(F) - torch.log(K) + 0.5 * sig ** 2 * tau) / (sig * torch.sqrt(tau))
        return Fp * n_dist(d_p) - K * torch.exp(-r*tau) * n_dist(d_p - sig * torch.sqrt(tau))

    @staticmethod
    def BS_call(t, x, K, r, sig):
        sigma_sqrtt = sig * torch.sqrt(t)
        _d = (
            (
                torch.log(x / K)
                + r * t +  0.5 * t * sig ** 2
            )
            / sigma_sqrtt
        )
        return x * n_dist(_d) - K * (torch.exp(-r*t) * n_dist(_d - sigma_sqrtt))

    @staticmethod
    def MJD_call(t, x, K, r, sig, _lambda, m, delta, threshold=1e-18, max_iter=100):
        """
        t: time to maturity
        x: spot price
        K: strike price
        r: risk-free rate
        sigma: volatility
        _lambda: jump intensity
        m: mean of jump size
        delta: standard deviation of jump size
        threshold: threshold for the iteration
        max_iter: maximum iteration number
        """
        muj = torch.exp(m + 0.5 * delta ** 2) - 1
        lambda_h = _lambda * torch.exp(m + 0.5 * delta**2)
        j, sig_j, r_j = 0, sig, r - _lambda * muj
        w = torch.exp(-lambda_h*t)
        res = 0
        while True:
            new = MJDbasket.BS_call(t, x, K, r_j, sig_j)
            if torch.all(w * new / (res+1e-30) < threshold):
                print("The iteration ends with completion of requirements.")
                break
            if j > max_iter:
                print("The iteration exceeds the maximum iteration number.")
                break
            res = res + w * new
            j += 1
            w = w * lambda_h * t / j
            sig_j, r_j = torch.sqrt(sig**2 + j*delta**2/t), r - _lambda * muj + j * (m + 0.5*delta**2) / t
        return res

    def solution(self, batch):
        """
        Outputs the exact solution.
        """
        print("Begin to calculate the solution.")
        t = self.hypercubes["t"].interval[1] - batch["t"]
        n = batch["sigma"].shape[-1] # the dimension of S_t
        batch_size = batch["sigma"].shape[0]

        RHO = batch["rho_b"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sig_h = torch.sqrt(
            1/n**2 * torch.sum(batch["sigma"].unsqueeze(2) * RHO * batch["sigma"].unsqueeze(1), (1,2)).reshape(-1,1)
        )
        RHO = batch["rho_j"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        delta_t = torch.sqrt(
            1/n**2 * torch.sum(batch["delta"].unsqueeze(2) * RHO * batch["delta"].unsqueeze(1), (1,2)).reshape(-1,1)
        )
        m_t = torch.mean(batch["m"], dim=1, keepdim=True)
        sig_t = torch.sqrt(torch.mean(batch["sigma"]**2, dim=1, keepdim=True))
        muj = torch.exp(batch["m"] + 0.5 * batch["delta"] ** 2) - 1
        mu_t = torch.mean(muj, dim=1, keepdim=True)

        x = torch.exp(torch.mean(torch.log(batch["x"]), dim=1, keepdim=True))

        res = self.MJD_call(t, x * torch.exp(0.5 * (sig_h**2 - sig_t**2)*t) * torch.exp(batch["lambda"]*(torch.exp(m_t + 0.5*delta_t**2)-1 - mu_t)*t),
                              batch["K"], batch["r"], sig_h, batch["lambda"], m_t, delta_t, threshold=1e-16, max_iter=1000)
        res = torch.where(t == 0, torch.nn.ReLU()(x - batch["K"]), res)
        return res

        # lambda_t = batch["lambda"] * torch.exp(m_t + 0.5 * delta_t**2)
        # j, sig_j, r_j =0, sig_h, batch["r"] - batch["lambda"] * mu_t
        # w = torch.exp(-lambda_t*t)
        # res = 0
        # while True:
        #     new = self._u(t, x, r_j, sig_j, sig_h, sig_t, batch["K"])
        #     if torch.all(w * new / (res + 1e-28) < 1e-18):
        #         break
        #     res = res + w * new
        #     print("res is ")
        #     print(res)
        #     j += 1
        #     w = w * lambda_t * t / j
        #     sig_j, r_j = torch.sqrt(sig_h**2 + j*delta_t**2/t), batch["r"] - batch["lambda"] * mu_t + j * (m_t + 0.5*delta_t**2) / t
        # return res
    
    def naf(self, batch, param):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std



PDES = {pde.__name__: pde for pde in Pde.get_subclasses()}
