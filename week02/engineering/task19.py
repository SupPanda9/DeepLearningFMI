from value import Value, trace
import graphviz

def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        filename="19_result", format="svg", graph_attr={"rankdir": "LR"}
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
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')

    x1w1 = x1 * w1
    x1w1.label = "x1w1"
    x2w2 = x2 * w2
    x2w2.label = "x1w1"

    x1w1_x2w2 = x1w1 + x2w2
    x1w1_x2w2.label = "x1w1 + x2w2"

    logit = x1w1_x2w2 + b
    logit.label = "logit"

    e = (2 * logit).exp()
    e.label = "e"
    
    L = (e - 1) / (e + 1)
    L.label = 'L'

    L.backward()

    draw_dot(L).render(directory="./week02/graphviz_output", view=True)

if __name__ == "__main__":
    main()