import torch
from torch import nn

#Functions for approximating the Jacobian.
#Taken from https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/residual.py

class MemoryEfficientLogDetEstimator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        estimator_fn,
        gnet,
        x,
        n_power_series,
        vareps,
        coeff_fn,
        training,
        *g_params
    ):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(),
                    (x,) + g_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError("Provide training=True if using backward.")

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[: len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2 :]

            dg_x, *dg_params = torch.autograd.grad(
                g, [x] + g_params, grad_g, allow_unused=True
            )

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple(
                [g.mul_(dL) if g is not None else None for g in grad_params]
            )

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple(
                [
                    dg.add_(djac) if djac is not None else dg
                    for dg, djac in zip(dg_params, grad_params)
                ]
            )

        return (None, None, grad_x, None, None, None, None) + grad_params


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.0).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[
            0
        ]
        tr = torch.sum(vjp.reshape(x.shape[0], -1) * vareps.reshape(x.shape[0], -1), 1)
        delta = (-1) ** (k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1) ** k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(
        vjp_jac.reshape(x.shape[0], -1) * vareps.reshape(x.shape[0], -1), 1
    )
    return logdetgrad


def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError("g is required to be an instance of nn.Module.")

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn,
        gnet,
        x,
        n_power_series,
        vareps,
        coeff_fn,
        training,
        *list(gnet.parameters())
    )


# Helper functions


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)