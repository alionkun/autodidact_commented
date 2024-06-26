"""Vector-Jacobian Products, Backpropagation.

Construct vector-Jacobian product of a DAG of computation. With this library,
one can,

- Construct vector-Jacobian product for any single-input single-output function.
  (make_vjp)
- Register vector-Jacobian product functions for any primitive function and
  argument index.
"""
from collections import defaultdict
from itertools import count
import numpy as np

from .tracer import trace, Node
from .util import toposort

def make_vjp(fun, x):
    """Make function for vector-Jacobian product.

    Args:
      fun: single-arg function. Jacobian derived from this.
      x: ndarray. Point to differentiate about.

    Returns:
      vjp: single-arg function. vector -> vector-Jacobian[fun, x] product.
      end_value: end_value = fun(start_node)

    """
    # NOTE(alionkun) start_node表示计算图的起点节点，也就是x对应的计算图的节点
    start_node = Node.new_root()
    end_value, end_node = trace(start_node, fun, x)
    if end_node is None:
        def vjp(g): return np.zeros_like(x)
    else:
        def vjp(g): return backward_pass(g, end_node)
    return vjp, end_value

def backward_pass(g, end_node):
    """Backpropagation.

    Traverse computation graph backwards in topological order from the end node.
    For each node, compute local gradient contribution and accumulate.
    """
    outgrads = {end_node: g}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        fun, value, args, kwargs, argnums = node.recipe
        for argnum, parent in zip(argnums, node.parents):
            # Lookup vector-Jacobian product (gradient) function for this
            # function/argument.
            vjp = primitive_vjps[fun][argnum]

            # Compute vector-Jacobian product (gradient) contribution due to
            # parent node's use in this function.
            parent_grad = vjp(outgrad, value, *args, **kwargs)

            # Save vector-Jacobian product (gradient) for upstream nodes.
            # Sum contributions with all others also using parent's output.
            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)
    return outgrad

def add_outgrads(prev_g, g):
    """Add gradient contributions together."""
    if prev_g is None:
        return g
    return prev_g + g

primitive_vjps = defaultdict(dict)
# NOTE(alionkun) VJP实际上就是TensorFlow中的OP的grad_fn，这里就是注册 grad_fn 
def defvjp(fun, *vjps, **kwargs):
    # NOTE(alionkun) 在TensorFlow中，grad_fn直接返回Loss关于每个函数参数的梯度，因此函数参数的顺序是固定的
    # NOTE(alionkun) 在defvjp中我们可以看到也是类似的
    """Register vector-Jacobian product functions.

    Let fun(x, y, ...) = ans be a function. We wish to register a
    vector-Jacobian product for each of fun's arguments. That is, functions

      vjp_x(g, ans, x, y, ...) = g df/dx
      vjp_y(g, ans, x, y, ...) = g df/dy
      ...

    This function registers said callbacks.

    Args:
      fun: function for which one wants to define vjps for.
      *vjps: functions. vector-Jacobian products. One per argument to fun().
      **kwargs: additional keyword arugments. Only 'argnums' is used.
    """
    # NOTE(alionkun) 如果指定了argnums作为fun的参数数量，则使用之
    # NOTE(alionkun) 否则赋予一个count()，这是一个无限序列生成器，默认从0开始，step=1
    # NOTE(alionkun) 这个参数貌似没见使用
    argnums = kwargs.get('argnums', count())
    for argnum, vjp in zip(argnums, vjps):
        primitive_vjps[fun][argnum] = vjp
