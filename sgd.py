import warnings

from pennylane._grad import grad as get_gradient
from pennylane.utils import _flatten, unflatten
from pennylane.numpy import ndarray, tensor


class GradientDescentOptimizer:

    def __init__(self, stepsize=0.01):
        self.stepsize = stepsize
        print("Using SGD from Zeyi Implentation")

    @property
    def _stepsize(self):
        warnings.warn(
            "'_stepsize' is deprecated. Please use 'stepsize' instead.",
            UserWarning,
            stacklevel=2,
        )

        return self.stepsize

    @_stepsize.setter
    def _stepsize(self, stepsize):
        warnings.warn(
            "'_stepsize' is deprecated. Please use 'stepsize' instead.",
            UserWarning,
            stacklevel=2,
        )

        self.stepsize = stepsize

    def update_stepsize(self, stepsize):
        r"""Update the initialized stepsize value :math:`\eta`.
        This allows for techniques such as learning rate scheduling.
        Args:
            stepsize (float): the user-defined hyperparameter :math:`\eta`
        """
        warnings.warn(
            "'update_stepsize' is deprecated. Stepsize value can be updated using "
            "the 'stepsize' attribute.",
            UserWarning,
            stacklevel=2,
        )

        self.stepsize = stepsize

    def step_and_cost(self, objective_fn, *args, grad_fn=None, **kwargs):

        g, forward = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args)

        if forward is None:
            forward = objective_fn(*args, **kwargs)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward
        return new_args, forward

    def step(self, objective_fn, *args, grad_fn=None, **kwargs):

        g, _ = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args)
        #print("new_args", new_args)
        # unwrap from list if one argument, cleaner return
        # if len(new_args) == 1:
        #     print("nnew_args", new_args[0])
        #     return new_args[0]
        
        return new_args

    @staticmethod
    def compute_grad(objective_fn, args, kwargs, grad_fn=None):
        g = get_gradient(objective_fn) if grad_fn is None else grad_fn
        grad = g(*args, **kwargs)
        forward = getattr(g, "forward", None)

        num_trainable_args = 0
        for arg in args:
            if getattr(arg, "requires_grad", True):
                num_trainable_args += 1

        if num_trainable_args == 1:
            #print("here")
            grad = (grad,)
        #print(grad)
        return grad, forward

    def apply_grad(self, grad, args):

        args_new = list(args)
        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                x_flat = _flatten(arg)
                grad_flat = _flatten(grad[trained_index])
                trained_index += 1

                x_new_flat = [e - self.stepsize * g for g, e in zip(grad_flat, x_flat)]

                args_new[index] = unflatten(x_new_flat, args[index])

                if isinstance(arg, ndarray):
                    # Due to a bug in unflatten, input PennyLane tensors
                    # are being unwrapped. Here, we cast them back to PennyLane
                    # tensors.
                    # TODO: remove when the following is fixed:
                    # https://github.com/PennyLaneAI/pennylane/issues/966
                    args_new[index] = args_new[index].view(tensor)
                    args_new[index].requires_grad = True

        return args_new