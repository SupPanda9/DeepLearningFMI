import graphviz
from value import Value, trace

EPSILON = 0.001


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        filename="04_result", format="svg", graph_attr={"rankdir": "LR"}
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


def main() -> None:
    a = Value(2.0)
    a.label = "a"
    # dl / da = (dl / de) (de / da)
    # de / da = (a * b)'_a = b = -3.0
    # dl / da = 5.0 * (-3.0) = -15.0
    a.grad = -15.0

    b = Value(-3.0)
    b.label = "b"
    # dl / db = (dl / de) (de / db)
    # de / db = (a * b)'_b = a = 2.0
    # dl / db = 5.0 * 2.0 = 10.0
    b.grad = 10.0

    c = Value(10.0)
    c.label = "c"
    # dl / dc = (dl / dd) (dd / dc)
    # dd / dc = (e + c)'_c = 1
    # dl / dc = 5.0 * 1.0 = 5.0
    c.grad = 5.0

    e = a * b
    e.label = "e"
    # dl / de = (dl / dd) (dd / de)
    # dd / de = (e + c)'_e = 1
    # dl / de = 5.0 * 1.0 = 5.0
    e.grad = 5.0

    d = e + c
    d.label = "d"
    # dl / dd = (f*d)'_d, dl / dd = f = 5.0000
    d.grad = 5.0

    f = Value(5.0)
    f.label = "f"
    # dl / df = (f*d)'_d, dl / df = d = 4.0000
    f.grad = 4.0

    L = d * f
    L.label = "L"
    # dl/dl = 1
    L.grad = 1

    print(round(manual_der(a, L)) == a.grad)
    print(round(manual_der(b, L)) == b.grad)
    print(round(manual_der(c, L)) == c.grad)
    print(round(manual_der(e, L)) == e.grad)
    print(round(manual_der(d, L)) == d.grad)
    print(round(manual_der(f, L)) == f.grad)
    print(round(manual_der(L, L)) == L.grad)


def manual_der(node: Value, root: Value):
    original = node.data

    topo_sort = []
    visited = set()

    def build_topology(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topology(child)
            topo_sort.append(v)

    build_topology(root)

    def forward():
        for v in topo_sort:
            if v is node:
                continue
            if v._op == "+":
                left, right = tuple(v._prev)
                v.data = left.data + right.data
            elif v._op == "*":
                left, right = tuple(v._prev)
                v.data = left.data * right.data

    forward()
    L_original = root.data

    node.data = original + EPSILON
    forward()
    L_eps = root.data

    node.data = original
    forward()

    return (L_eps - L_original) / EPSILON


if __name__ == "__main__":
    main()
