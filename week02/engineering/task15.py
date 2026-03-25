import graphviz
from value import Value, trace


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        filename="15_result", format="svg", graph_attr={"rankdir": "LR"}
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

    x = Value(10.0)
    x.label = "x"
    y = Value(5.0)
    y.label = "y"

    z = x + y
    z.label = "z"

    z.backward()

    draw_dot(z).render(directory="./week02/graphviz_output", view=True)


if __name__ == "__main__":
    main()
