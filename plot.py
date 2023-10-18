import torch
import matplotlib.pyplot as plt

def get_option_price_curve(model, K, num_path, config):
    device = torch.device("cuda")
    with torch.no_grad(): 
        data = {
            key: torch.full(((K+1)*num_path, 1), value, dtype=torch.float32, device=device) for key, value in config.items()
        }
        data.update(
            {
                "t": torch.linspace(model.pde.hypercubes["t"].interval[0], model.pde.hypercubes["t"].interval[1], K + 1, dtype=torch.float32, device=device).repeat(num_path).reshape(-1,1)
            }
        )
        batch = {
            key: torch.full((num_path, 1), value, dtype=torch.float32, device=device) for key, value in config.items()
        }
        x = torch.full((num_path,K+1), 1, dtype=torch.float32, device=device)
        for i in range(K+1):
            x[:,i] = batch["s"].squeeze(1).clone()
            dw = (1/K)** 0.5  * torch.randn(
                batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device
            )
            batch["s"] = batch["s"] * torch.exp(
                batch["r"] * 1/K - 0.5 * 1/K * batch["sigma"] ** 2 + batch["sigma"] * dw
            )
        if type(model.pde).__name__ == "BSr":
            data["x"] = x.reshape(-1,1)
        elif type(model.pde).__name__ == "BSlookback":
            mt = torch.concat([torch.min(x[:,:_i], dim=1).values.reshape(-1,1) for _i in range(1,K+2)], dim=1)
            data["x"] = torch.concat([x.reshape(-1,1), mt.reshape(-1,1)], dim=1)
        
        if model.pde.get_K is not None:
            data["K"] = model.pde.get_K(data)
        tensor = model.pde.normalize_and_flatten(data)
        y_pred = model.net.forward(tensor)
        y_true = model.pde.solution(data)
    return {
        "t": data["t"].reshape(num_path, K+1),
        "y_pred": y_pred.reshape(num_path, K+1),
        "y_true": y_true.reshape(num_path, K+1)
        }


def plot_mc_curves(t_test, y_pred, y_true=None):
    if t_test.device != "cpu":
        t_test= t_test.cpu()
    if y_pred.device != "cpu":
        y_pred= y_pred.cpu()
    if y_true.device != "cpu":
        y_true= y_true.cpu()
    samples = 16
    plt.figure(figsize=(13, 8))
    plt.plot(t_test[0:1,:].T, y_pred[0:1,:].T,'darkviolet',label='Learned solution')
    plt.plot(t_test[0:1,:].T, y_true[0:1,:].T,'--',color='lightseagreen',label='Exact solution')
    # plt.plot(t_test[0:1,-1], y_true[0:1,-1],'ko')

    plt.plot(t_test[1:samples,:].T, y_pred[1:samples,:].T,'darkviolet')
    plt.plot(t_test[1:samples,:].T, y_true[1:samples,:].T,'--',color='lightseagreen')
    # plt.plot(t_test[1:samples,-1], y_true[1:samples,-1],'ko')

    # plt.plot([0],y_true[0,0],'ks') # ,label='$Y_0 = u(0,X_0)$')

    plt.xlabel('time', fontdict={"size": 18})
    plt.ylabel('price', fontdict={"size": 18})
    #plt.title(fr'm={u_hat[-1]: 2f}, r={u_hat[0]: 2f}, $\theta$={u_hat[1]: 2f},$\kappa$={u_hat[2]: 2f}， $\sigma$={u_hat[3]: 2f}, $\rho$={u_hat[4]: 2f}')# , $\rho$={u_hat[2]: 2f}')
    plt.legend(prop={"size": 15})
    plt.show()
