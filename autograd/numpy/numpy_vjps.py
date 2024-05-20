"""Vector-Jacobian products for NumPy functions.

This library consists of implementations of vector-Jacobian products (vjps, gradients)
for functions implemented in numpy. Each function-argument index pair is
provided a gradient function registered with defvjp().

Notice that vjps are implemented with (autograd-wrapped) functions as well. This
is the magic that allows one to compute gradients-of-gradients!
"""
from __future__ import absolute_import
import numpy as onp
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.tracer import primitive
from autograd.core import defvjp

# ----- Binary ufuncs -----

defvjp(anp.add,         lambda g, ans, x, y : unbroadcast(x, g),
                        lambda g, ans, x, y : unbroadcast(y, g))
defvjp(anp.multiply,    lambda g, ans, x, y : unbroadcast(x, y * g),
                        lambda g, ans, x, y : unbroadcast(y, x * g))
defvjp(anp.subtract,    lambda g, ans, x, y : unbroadcast(x, g),
                        lambda g, ans, x, y : unbroadcast(y, -g))
defvjp(anp.divide,      lambda g, ans, x, y : unbroadcast(x,   g / y),
                        lambda g, ans, x, y : unbroadcast(y, - g * x / y**2))
defvjp(anp.true_divide, lambda g, ans, x, y : unbroadcast(x,   g / y),
                        lambda g, ans, x, y : unbroadcast(y, - g * x / y**2))
defvjp(anp.power,
    lambda g, ans, x, y: unbroadcast(x, g * y * x ** anp.where(y, y - 1, 1.)),
    lambda g, ans, x, y: unbroadcast(y, g * anp.log(replace_zero(x, 1.)) * x ** y))

def replace_zero(x, val):
    """Replace all zeros in 'x' with 'val'."""
    return anp.where(x, x, val)

def unbroadcast(target, g, broadcast_idx=0):
    """Remove broadcasted dimensions by summing along them.

    When computing gradients of a broadcasted value, this is the right thing to
    do when computing the total derivative and accounting for cloning.
    """
    # NOTE(alionkun) 以 fun = add(x, y) 为例， vjp of x 是 lambda g, ans, x, y : unbroadcast(x, g)
    # NOTE(alionkun) x和y均有可能在add中被broadcast，这里假设x=1, y=[1,2,3]，也就是x被broadcast了
    # NOTE(alionkun) 在计算梯度的时候就需要将broadcast考虑进来，通过对比x和ans或者x和g的维度即可判断x是否被broadcast
    # NOTE(alionkun) 假设x被广播过，那么g的维度就会比x的维度多，通过不断从axis=0，即高纬度对g进行reduce sum，最终将g和x的维度对齐（这里x的g一定是维度对齐的）
    # NOTE(alionkun) 广播一定是在高纬度扩展
    # NOTE(alionkun) ** 例子1：x=1, y=[1,2,3]
    # NOTE(alionkun) 则 g.shape=ans.shape=(3,)，假设g=[0.1, 0.1, 0.2]，则 vjp of x 就是 0.4, 其shape=(,)=x.shape
    # NOTE(alionkun) ** 例子2：x=[1,2,3], y=[[1,2,3],[4,5,6]]
    # NOTE(alionkun) 则 g.shape=ans.shape=(2,3)，假设g=[[0.1, 0.2, 0.3], [0.1,0.1, 0.1]]，则 vjp of x 就是 [0.2, 0.3, 0.4]，其shape=(3,)=x.shape
    while anp.ndim(g) > anp.ndim(target):
        g = anp.sum(g, axis=broadcast_idx)
    # NOTE(alionkun) 如果x包含size=1的维度，情况会是如何？
    # NOTE(alionkun) ** 例子3： x=[[1,2,3]], y=[[1,2,3],[4,5,6]]，假设g=[[0.1, 0.2, 0.3], [0.1,0.1, 0.1]]
    # NOTE(alionkun) 经过上面的处理，到这里 g=[[0.1, 0.2, 0.3], [0.1,0.1, 0.1]]
    # NOTE(alionkun) target.shape=(1,3)而g.shape=(2,3)，显然需要继续对齐维度
    # NOTE(alionkun) 对齐的方法显然就是在target.size=1的维度，对g进行reduce sum，并保持keepdims=True
    # NOTE(alionkun) 之后即可得到g.shape=(1,3)，和target.shape完全对齐
    for axis, size in enumerate(anp.shape(target)):
        if size == 1:
            g = anp.sum(g, axis=axis, keepdims=True)
    # TODO(alionkun) 这是干嘛的
    if anp.iscomplexobj(g) and not anp.iscomplex(target):
        g = anp.real(g)
    return g

# ----- Simple grads -----

defvjp(anp.negative, lambda g, ans, x: -g)
defvjp(anp.exp,    lambda g, ans, x: ans * g)
defvjp(anp.log,    lambda g, ans, x: g / x)
defvjp(anp.tanh,   lambda g, ans, x: g / anp.cosh(x) **2)
defvjp(anp.sinh,   lambda g, ans, x: g * anp.cosh(x))
defvjp(anp.cosh,   lambda g, ans, x: g * anp.sinh(x))

defvjp(anp.where, None,
       lambda g, ans, c, x=None, y=None: anp.where(c, g, anp.zeros(g.shape)),
       lambda g, ans, c, x=None, y=None: anp.where(c, anp.zeros(g.shape), g))

defvjp(anp.reshape, lambda g, ans, x, shape, order=None:
       anp.reshape(g, anp.shape(x), order=order))

# ----- Dot grads -----

def _dot_vjp_0(g, ans, lhs, rhs):
  if max(anp.ndim(lhs), anp.ndim(rhs)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if anp.ndim(lhs) == 0:
    return anp.sum(rhs * g)
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 1:
    return g * rhs
  if anp.ndim(lhs) == 2 and anp.ndim(rhs) == 1:
    return g[:, None] * rhs
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 2:
    return anp.dot(rhs, g)
  return anp.dot(g, rhs.T)

def _dot_vjp_1(g, ans, lhs, rhs):
  if max(anp.ndim(lhs), anp.ndim(rhs)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if anp.ndim(rhs) == 0:
    return anp.sum(lhs * g)
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 1:
    return g * lhs
  if anp.ndim(lhs) == 2 and anp.ndim(rhs) == 1:
    return anp.dot(g, lhs)
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 2:
    return lhs[:, None] * g
  return anp.dot(lhs.T, g)

defvjp(anp.dot, _dot_vjp_0, _dot_vjp_1)
