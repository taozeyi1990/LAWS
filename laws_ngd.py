from pennylane import numpy as np

import pennylane as qml
from pennylane.utils import _flatten, unflatten
from pennylane.optimize.gradient_descent import GradientDescentOptimizer
from pennylane.optimize import QNGOptimizer


class LAWSNG(GradientDescentOptimizer):

    def __init__(self,
                 delta=0.01,
                 lookaround_optimizer=GradientDescentOptimizer,
                 lookaround_stepsize=0.02,
                 lookaround_loop=5,
                 gradient_method="default",
                 beta=0.9,
                 approx="block-diag",
                 lam=0.001,
                 **kwargs):
        super().__init__(delta)

        self.delta = delta
        self.lookaround_stepsize = lookaround_stepsize
        self.lookaround_loop = lookaround_loop
        self.gradient_method = gradient_method
        self.beta = beta
        self.approx = approx
        self.lam = lam
        self.lookaround_optim = lookaround_optimizer(
            lookaround_stepsize, **kwargs)

        self.gradient_samples = []

        if isinstance(self.lookaround_optim, QNGOptimizer):
            raise NotImplementedError(
                "Current Pennylane does not support QNGOptimizer as look around optimizer")

    def step_and_cost(self, qnode, *args, grad_fn=None, recompute_tensor=True, metric_tensor_fn=None, **kwargs):

        if len(args) > 1:
            raise NotImplementedError(
                "Current Pennylane does not support multi args NG.")

        new_args = prev_args = args[0]

        for _ in range(self.lookaround_loop):
            new_args = self.lookaround_optim.step(
                qnode, new_args, grad_fn=grad_fn, **kwargs)

            self.gradient_samples.append(prev_args - new_args)

            prev_args = new_args

        if recompute_tensor or self.metric_tensor is None:
            if metric_tensor_fn is None:

                metric_tensor_fn = qml.metric_tensor(qnode, approx=self.approx)

            _metric_tensor = metric_tensor_fn(new_args)
            # Reshape metric tensor to be square
            shape = qml.math.shape(_metric_tensor)
            size = qml.math.prod(shape[: len(shape) // 2])
            self.metric_tensor = qml.math.reshape(_metric_tensor, (size, size))
            # Add regularization
            self.metric_tensor = self.metric_tensor + self.lam * qml.math.eye(
                size, like=_metric_tensor
            )

        new_args = np.array(self.apply_grad(
            self.gradient_samples, new_args), requires_grad=True)

        forward = qnode(new_args)

        return new_args, forward

    def step(self, qnode, *args, grad_fn=None, recompute_tensor=True, metric_tensor_fn=None, **kwargs):

        new_args, _ = self.step_and_cost(
            qnode,
            *args,
            grad_fn=grad_fn,
            recompute_tensor=recompute_tensor,
            metric_tensor_fn=metric_tensor_fn,
            **kwargs,
        )
        return new_args

    def apply_grad(self, gradient_samples, args):

        if self.gradient_method == "default":
            smapled_gradient = sum(
                [g for g in gradient_samples])/self.lookaround_loop
        elif self.gradient_method == "ema":
            smapled_gradient= np.zeros_like(gradient_samples[0])
            for idx, g in enumerate(gradient_samples):
                smapled_gradient = self.beta*smapled_gradient + (1 -self.beta) * g
        else:
            raise NotImplementedError("No such implementation")

        x_warm_start_flat = np.array(list(_flatten(args)))
        x_new_flat = x_warm_start_flat - self.delta * \
            np.linalg.solve(self.metric_tensor, smapled_gradient)

        self.gradient_samples = []
        return x_new_flat
