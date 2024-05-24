"""Utility functions."""


def subvals(x, ivs):
    """Replace the i-th value of x with v.

    Args:
      x: iterable of items.
      ivs: list of (int, value) pairs.

    Returns:
      x modified appropriately.
    """
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

def subval(x, i, v):
    """Replace the i-th value of x with v. x是一个位置参数列表，使用v替换x中第i个参数"""
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

def toposort(end_node): # 以结束节点（loss）为起点，反向遍历计算图，得到反向图的节点拓扑序，也是 backprop 的计算顺序
    child_counts = {} # 统计每个节点的孩子节点数量
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents) # 能够进入 stack 是因为在孩子节点的 parents 列表，所以 stack 中的节点都对应着一个孩子节点

    childless_nodes = [end_node] # childless 中的节点的孩子节点都找到了，从梯度计算的角度看，也就是该节点可以结算梯度了。
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

def wraps(fun, namestr="{fun}", docstr="{doc}", **kwargs):
    """Decorator for a function wrapping another.

    Used when wrapping a function to ensure its name and docstring get copied
    over. // 只是为了拷贝 name 和 doc 两个属性，应用场景是被装饰的函数本身也是一个装饰器

    Args:
      fun: function to be wrapped
      namestr: Name string to use for wrapped function.
      docstr: Docstring to use for wrapped function.
      **kwargs: additional string format values.

    Return:
      Wrapped function.
    """
    def _wraps(f):
        try:
            f.__name__ = namestr.format(fun=get_name(fun), **kwargs)
            f.__doc__ = docstr.format(fun=get_name(fun), doc=get_doc(fun), **kwargs)
        finally:
            return f
    return _wraps

def wrap_nary_f(fun, op, argnum):
    namestr = "{op}_of_{fun}_wrt_argnum_{argnum}"
    docstr = """\
    {op} of function {fun} with respect to argument number {argnum}. Takes the
    same arguments as {fun} but returns the {op}.
    """
    return wraps(fun, namestr, docstr, op=get_name(op), argnum=argnum)

get_name = lambda f: getattr(f, '__name__', '[unknown name]')
get_doc  = lambda f: getattr(f, '__doc__' , '')
