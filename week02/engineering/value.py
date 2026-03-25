import math
import graphviz


class Value:

    def __init__(self, data=0.0, _prev=None, _op="", grad=0.0, label=""):
        self.data = data
        self._prev = _prev if _prev is not None else set()
        self._op = _op
        self.grad = grad
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        new_data = Value(self.data + other.data, {self, other}, "+")

        def _backward():
            self.grad += 1.0 * new_data.grad
            other.grad += 1.0 * new_data.grad

        new_data._backward = _backward
        return new_data

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        new_data = Value(self.data * other.data, {self, other}, "*")

        def _backward():
            self.grad += other.data * new_data.grad
            other.grad += self.data * new_data.grad

        new_data._backward = _backward
        return new_data

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        new_data = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * new_data.grad

        new_data._backward = _backward

        return new_data

    def __pow__(self, other):
        new_data = Value(
            self.data**other,
            {
                self,
            },
            f"**{other}",
        )

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * new_data.grad

        new_data._backward = _backward
        return new_data

    def exp(self):
        x = self.data
        new_data = Value(
            math.exp(x),
            {
                self,
            },
            "e",
        )

        def _backward():
            self.grad += new_data.data * new_data.grad

        new_data._backward = _backward
        return new_data

    def __truediv__(self, other):
        return self * (other**-1)

    def backward(self):
        # for the L (root)
        self.grad = 1.0

        topo = top_sort(self)

        for node in reversed(topo):
            node._backward()


def top_sort(root):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(root)
    return topo


def trace(v: Value):
    nodes = set()
    edges = set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)

            if v._prev is not None:
                for pr in v._prev:
                    edges.add((pr, v))
                    build(pr)

    build(v)
    return nodes, edges


def draw_dot(root: Value, fn="") -> graphviz.Digraph:
    dot = graphviz.Digraph(
        filename=fn, format="svg", graph_attr={"rankdir": "LR"}
    )  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))

        label_text = getattr(n, "label", "")

        # for any value in the graph, create a rectangular ('record') node
        dot.node(
            name=uid,
            label=f"{{ {label_text} | data: {n.data:.4f} | grad: {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
