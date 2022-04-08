from pennylane.utils import _flatten, unflatten
from pennylane import numpy as np
from pennylane.numpy import ndarray, tensor
from pennylane.optimize.gradient_descent import GradientDescentOptimizer
from pennylane.optimize import QNGOptimizer


class LAWS(GradientDescentOptimizer):

    def __init__(self, delta=0.01,
                 lookaround_optimizer=GradientDescentOptimizer,
                 lookaround_stepsize=0.02,
                 lookaround_loop=5,
                 sampling_method="lookahead",
                 gradient_method="default",
                 beta=0.9,
                 sampling_stepsize=0.5,
                 **kwargs):
        super().__init__(delta)

        self.delta = delta
        self.lookaround_stepsize = lookaround_stepsize
        self.lookaround_loop = lookaround_loop
        self.sampling_method = sampling_method
        self.gradient_method = gradient_method
        self.beta = beta
        self.sampling_stepsize = sampling_stepsize
        self.kwargs = kwargs
        self.lookaround_optim = lookaround_optimizer(
            lookaround_stepsize, **kwargs)

        if isinstance(self.lookaround_optim, QNGOptimizer):
            raise NotImplementedError(
                "Current Pennylane does not support QNGOptimizer as look around optimizer")

        self.gradient_samples = []

    def step_and_cost(self, objective_fn, *args, grad_fn=None, **kwargs):

        new_args = prev_args = args
    
        forward = None
        for _ in range(self.lookaround_loop):

            if self.sampling_method == "lookahead":
                
                new_args = self.lookaround_optim.step(
                    objective_fn, *new_args, grad_fn=grad_fn, **kwargs)

                single_lookaround_gradient = []
                for n, p in zip(new_args, prev_args):
                    if getattr(n, "requires_grad", True):
                        g = n - p
                        single_lookaround_gradient.append(g)
                self.gradient_samples.append(single_lookaround_gradient)
                prev_args = new_args

            elif self.sampling_method == "lookaheadrandomsmaple":
                kwargs['lookahead'] = True
                new_args = self.lookaround_optim.step(
                    objective_fn, *new_args, grad_fn=grad_fn, **kwargs)

                single_lookaround_gradient = []
                for n, p in zip(new_args, prev_args):
                    if getattr(n, "requires_grad", True):
                        g = n - p
                        single_lookaround_gradient.append(g)
                self.gradient_samples.append(single_lookaround_gradient)
                prev_args = new_args

            elif self.sampling_method == "lookmultiahead":
                kwargs['lookahead'] = True
                cur_args = self.lookaround_optim.step(
                    objective_fn, *new_args, grad_fn=grad_fn, **kwargs)

                single_lookaround_gradient = []
                for n, p in zip(cur_args, new_args):
                    if getattr(n, "requires_grad", True):
                        g = p - n
                        single_lookaround_gradient.append(g)
                self.gradient_samples.append(single_lookaround_gradient)
            else:
                raise NotImplementedError("No such implementation")

        new_args = self.apply_grad(self.gradient_samples, new_args)

        if forward is None:
            forward = objective_fn(*new_args, **kwargs)

        if len(new_args) == 1:
            return new_args[0], forward

        return new_args, forward

    def step(self, objective_fn, *args, grad_fn=None, **kwargs):

        new_args, _ = self.step_and_cost(
            objective_fn,
            *args,
            grad_fn=grad_fn,
            **kwargs,
        )

        return new_args

    def apply_grad(self, gradient_samples, args):

        if self.gradient_method == "default":
            smapled_gradient = [sum(i) for i in zip(*gradient_samples)]
        elif self.gradient_method == "ema":
            smapled_gradient = []
            if self.beta != 0:
                it = len(gradient_samples)
                for i in range(len(gradient_samples[0])):
                    tmp = np.zeros_like(gradient_samples[0][i])
                    for j in range(it):
                        #tmp += self.beta**(it - j -1) * gradient_samples[j][i]
                        tmp = self.beta*tmp + \
                            (1 - self.beta) * gradient_samples[j][i]
                    #tmp*= (1 - self.beta)
                    smapled_gradient.append(tmp)
        else:
            raise NotImplementedError("No such implementation")

        if self.sampling_method == "lookmultiahead":
            for smaple in smapled_gradient:
                smaple /= self.lookaround_loop

        args_new = list(args)
        trained_index = 0

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                if self.sampling_method == "lookahead" or "lookaheadrandomsmaple":
                    x_warm_start_flat = _flatten(arg)
                elif self.sampling_method == "lookmultiahead":
                    x_warm_start_flat = _flatten(arg) - self.sampling_stepsize * _flatten(smapled_gradient[trained_index])
                else:
                    raise NotImplementedError("No such implementation")
                grad_flat = _flatten(smapled_gradient[trained_index])
                trained_index += 1
                x_new_flat = [x - self.delta *
                              g for x, g in zip(x_warm_start_flat, grad_flat)]
                args_new[index] = unflatten(x_new_flat, args[index])

                if isinstance(arg, ndarray):
                    # Due to a bug in unflatten, input PennyLane tensors
                    # are being unwrapped. Here, we cast them back to PennyLane
                    # tensors.
                    # TODO: remove when the following is fixed:
                    # https://github.com/PennyLaneAI/pennylane/issues/966
                    args_new[index] = args_new[index].view(tensor)
                    args_new[index].requires_grad = True

        self.gradient_samples = []

        return args_new
