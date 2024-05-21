"""Convenience functions built on top of `make_vjp`."""

import numpy as np

from .core import make_vjp
from .util import subval

def grad(fun, argnum=0):
    """Constructs gradient function.

    Given a function fun(x), returns a function fun'(x) that returns the
    gradient of fun(x) wrt x.

    Args:
      fun: single-argument function. ndarray -> ndarray.
      argnum: integer. Index of argument to take derivative wrt.

    Returns:
      gradfun: function that takes same args as fun(), but returns the gradient
        wrt to fun()'s argnum-th argument.
    """
    def gradfun(*args, **kwargs):
        # Replace args[argnum] with x. Define a single-argument function to
        # compute derivative wrt.
        # NOTE(alionkun) 注意上面的注释写到 fun 是一个 single-argument 函数，应该是不对的
        # NOTE(alionkun) fun 可以是任意输入形式的函数，但这里只支持对其中一个位置参数求偏导
        # NOTE(alionkun) unary 表示一元， subval()将第argnum个位置参数替换为新函数的唯一输入x，对fun来说没有变化
        unary_fun = lambda x: fun(*subval(args, argnum, x), **kwargs)

        # Construct vector-Jacobian product
        # NOTE(alionkun) 接着上面，最后用的还是同一个参数
        vjp, ans = make_vjp(unary_fun, args[argnum])
        # NOTE(alionkun) ones_like()意味着用户不能指定输入的梯度
        return vjp(np.ones_like(ans))
    return gradfun
