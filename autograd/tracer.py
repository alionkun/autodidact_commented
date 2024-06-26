"""Tracing utilities.


This library provides functions for constructing a computation graph. With this
library, one can,

- Build a computation graph. (trace)
- Register wrapper types for unwrapped values based on type(). (Box.register)
- Build functions that can deal with wrapped values. (primitive,
  notrace_primitive)
- Box values. (new_box)
"""
from collections import defaultdict
from contextlib import contextmanager

from .util import subvals, wraps

def trace(start_node, fun, x):
    with trace_stack.new_trace() as trace_id:
        # Wrap 'x' in a box.
        start_box = new_box(x, trace_id, start_node)

        # Apply fun() to boxed value. This will carry the value throughout the
        # comutation as well as the box.
        end_box = fun(start_box)

        if isbox(end_box) and end_box._trace_id == start_box._trace_id:
            # Extract final value (== fun(x)) and its node in the computation
            # graph.
            return end_box._value, end_box._node
        else:
            # Output seems independent of input
            return end_box, None

class Node(object):
    """A node in a computation graph."""
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        """

        Args:
          value: output of fun(*args, **kwargs)
          fun: wrapped numpy that was applied.
          args: all (unboxed) positional arguments.
          kwargs: dict of additional keyword args.
          parent_argnums: integers corresponding to positional indices of boxed
            values.
          parents: Node instances corresponding to parent_argnums.
        """
        self.parents = parents
        self.recipe = (fun, value, args, kwargs, parent_argnums)

    def initialize_root(self):
        # NOTE(alionkun) 计算图的起始节点是没有上游节点的
        self.parents = []
        # NOTE(alionkun) `lambda x: x`表示起始节点也就是直接输出计算图的输入
        self.recipe = (lambda x: x, None, (), {}, [])

    @classmethod
    # NOTE(wilkenslin) 为什么搞一个new_root函数，使用__init__()无法满足需求吗
    # NOTE(wilkenslin) 另外root是否叫做leaf更合适？
    # NOTE(wilkenslin) 看来需要看autograd的原作才行，教学版有关键细节丢失的问题
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        # NOTE(alionkun) 唯一的子类 ArrayBox 没有重写该函数，root节点就是这个样，并且唯一的调用处没有使用任何参数
        root.initialize_root(*args, **kwargs)
        return root

def primitive(f_raw):
    """Wraps a function so that its gradient (vjp) can be specified and its
    invocation can be recorded."""
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        # Fetch boxed arguments with largest trace_id.  This ensures that the
        # computational graph being constructed only consists of other nodes
        # from the same call to trace().
        boxed_args, trace_id = find_top_boxed_args(args)
        if boxed_args:
            # Replace some elements of args with corresponding unboxed values.
            argvals = subvals(args, [(argnum, box._value) for argnum, box in boxed_args])
            # Get nodes for each boxed argument.
            parents = tuple(box._node for _, box in boxed_args)

            # Get argument indices for each boxed argument.
            argnums = tuple(argnum for argnum, _ in boxed_args)

            # Calculate result of applying original numpy function.
            #
            # Note that we use a recursive call here in order to also augment
            # outer calls to trace() with lower trace_ids. See TraceStack's
            # docstring for details.
            ans = f_wrapped(*argvals, **kwargs)

            # Create a new node
            node = Node(ans, f_wrapped, argvals, kwargs, argnums, parents)
            return new_box(ans, trace_id, node)
        else:
            return f_raw(*args, **kwargs)
    return f_wrapped

def notrace_primitive(f_raw):
    """Wrap a raw numpy function by discarding boxes.

    Results are not boxed. Unboxing is a signal that the f_raw() is
    non-differentiable with respect to its arguments. Consider the computation,

    ```
    x = 1.5
    y = np.floor(x) + x
    ```

    What is the derivative of y wrt x? Autograd says 1. as np.floor has zero
    derivative near x=1.5.
    """
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        # Extract np.ndarray values from boxed values.
        argvals = map(getval, args)

        # Call original function. Note that f_raw()'s arguments may still be
        # boxed, but with a lower trace_id. 因为可能有多层包装，因为可以求高阶梯度
        return f_raw(*argvals, **kwargs)
    return f_wrapped

def find_top_boxed_args(args):
    """Finds boxed arguments with largest trace_id.

    Equivalent to finding the largest trace_id of any argument, keeping args
    with the same, and dropping the remainder.

    Args:
      args: Arguments to function wrapped by primitive().

    Returns:
      top_boxes: List of (index, boxed argument). Arguments have same, largest
        trace_id.
      top_trace_id: trace_id of all elements in top_boxes.
    """
    top_trace_id = -1
    top_boxes = []
    for argnum, arg in enumerate(args):
        if isbox(arg):
            if arg._trace_id > top_trace_id:
                top_boxes = [(argnum, arg)]
                top_trace_id = arg._trace_id
            elif arg._trace_id == top_trace_id:
                top_boxes.append((argnum, arg))
    return top_boxes, top_trace_id

class TraceStack(object):
    """Tracks number of times trace() has been called.

    This is critical to ensure calling grad() on a function that also calls
    grad() resolves correctly. For example,

    ```
    def f(x):
      def g(y):
        return x * y
      return grad(g)(x)

    y = grad(f)(5.)
    ```

    First, grad(f)(5.) wraps 5. in a Box and calls f(Box(5)). Then, grad(g)(x)
    wraps Box(5) again and calls g(Box(Box(5)). When computing grad(g), we want
    to treat x=Box(5) as fixed -- it's not a direct argument to g(). How does
    Autograd know that x is fixed, when all it can see is
    np.multipy(Box(5.), Box(Box(5.))? Because the second argument has a larger
    trace_id than the former!
    """
    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self):
        """Increment trace depth."""
        self.top += 1
        yield self.top
        self.top -= 1

trace_stack = TraceStack()

# NOTE(alionkun) 参考https://matt-graham.github.io/slides/ad/index.html#/5/7
# NOTE(alionkun) Box机制用于包装真是的value和计算图的node
class Box(object):
    """Boxes a value within a computation graph."""

    # Type -> subclasses of Box. Types may be instances of Box. Subclasses must
    # take same arguments for __init__().
    type_mappings = {}

    # Non-Box types that can be boxed. Box的所有子类
    types = set()

    def __init__(self, value, trace_id, node): # Box 和 Node 的关系是什么？
        self._value = value
        self._node = node
        self._trace_id = trace_id

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self._value))

    @classmethod
    def register(cls, value_type):
        """Register a class as a Box for type 'value_type'.

        Should be called immediately after declaration.

        Args:
          cls: Inherits from Box. Type to box values of type 'value_type'.
          value_type: Type to be boxed.
        """
        # NOTE(alionkun) types是一个set集合，所以相当于登记了所有Box的子类
        # NOTE(alionkun) type_mappings使得可以通过普通类型找到对应的Box类型
        Box.types.add(cls)
        Box.type_mappings[value_type] = cls

        # TODO(alionkun) 不好懂啊
        # The Box implementation for a Box type is itself. Why? Imagine a nested
        # call to grad(). One doesn't want the inner Box's computation graph to
        # interact with the outer Box's.
        Box.type_mappings[cls] = cls


box_type_mappings = Box.type_mappings # keys=[<class 'numpy.ndarray'>, <class 'autograd.numpy.numpy_boxes.ArrayBox'>, <class 'float'>, <class 'numpy.float64'>, <class 'numpy.float32'>, <class 'numpy.float16'>, <class 'complex'>, <class 'numpy.complex64'>, <class 'numpy.complex128'>]，没有 Box 类型。

def new_box(value, trace_id, node):
    """Box an unboxed value.

    Args:
      value: unboxed value
      trace_id: int. Trace stack depth. 栈的深度 == 梯度的阶数 ？？？
      node: Node corresponding to this boxed value.

    Returns:
      Boxed value.
    """
    try:
        return box_type_mappings[type(value)](value, trace_id, node) # 目前都是 autograd.numpy.numpy_boxes.ArrayBox 类型
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

box_types = Box.types # 目前只有 autograd.numpy.numpy_boxes.ArrayBox 一个类型

# If True, the value is Box.
isbox  = lambda x: type(x) in box_types  # almost 3X faster than isinstance(x, Box)

# Get value from a Box.
# NOTE(alionkun) 通过递归来实现多层嵌套的 unboxing 
getval = lambda x: getval(x._value) if isbox(x) else x
