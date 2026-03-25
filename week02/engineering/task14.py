import graphviz
from value import Value, trace


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        filename="14_result", format="svg", graph_attr={"rankdir": "LR"}
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

    x1 = Value(2.0)
    x1.label = "x1"
    x2 = Value(0.0)
    x2.label = "x2"

    w1 = Value(-3.0)
    w1.label = "w1"
    w2 = Value(1.0)
    w2.label = "w2"

    b = Value(6.8813735870195432)
    b.label = "b"

    x1w1 = x1 * w1
    x1w1.label = "x1*w1"

    x2w2 = x2 * w2
    x2w2.label = "x2*w2"

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"

    logit = x1w1x2w2 + b
    logit.label = "logit"

    L = logit.tanh()
    L.label = "L"

    # derivative of L with respect to itself is 1
    # dL / dL = 1
    L.grad = 1.0

    # derivative of tanh: d/dx tanh(x) = 1 - tanh^2(x)
    # dL/dlogit = (dL/dL) * (1 - tanh(logit)**2)
    # logit.grad = 1.0 * (1 - L.data**2) = 1 - (0.707106**2) ~= 0.5
    logit.grad = 0.5

    # (+) distributes the gradient to both inputs
    # L = logit = x1w1x2w2 + b
    # x1w1x2w2.grad = dL/dx1w1x2w2 = (dL/dlogit) * (dlogit/dx1w1x2w2) = 0.5 * 1.0
    x1w1x2w2.grad = 0.5
    # b.grad = dL/db = (dL/dlogit) * (dlogit/db) = 0.5 * 1.0
    b.grad = 0.5

    # x1w1x2w2 = x1w1 + x2w2 addition
    # x1w1.grad = 0.5
    x1w1.grad = 0.5
    # x2w2.grad = 0.5
    x2w2.grad = 0.5

    # multiplication (*) derivative is the other input
    # x2w2 = x2 * w2
    # x2.grad = dL/dx2 = (dL/dx2w2) * (dx2w2/dx2) = 0.5 * w2.data = 0.5 * 1.0
    x2.grad = 0.5
    # w2.grad = dL/dw2 = (dL/dx2w2) * (dx2w2/dw2) = 0.5 * x2.data = 0.5 * 0.0
    w2.grad = 0.0

    # x1w1 = x1 * w1
    # x1.grad = dL/dx1 = (dL/dx1w1) * (dx1w1/dx1) = 0.5 * w1.data = 0.5 * -3.0
    x1.grad = -1.5
    # w1.grad = dL/dw1 = (dL/dx1w1) * (dx1w1/dw1) = 0.5 * x1.data = 0.5 * 2.0
    w1.grad = 1.0

    draw_dot(L).render(directory="./week02/graphviz_output", view=True)


if __name__ == "__main__":
    main()
