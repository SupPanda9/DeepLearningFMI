from value import Value


def main() -> None:
    a = Value(2.0, grad=-15.0)
    b = Value(-3.0, grad=10.0)
    c = Value(10.0, grad=5.0)
    f = Value(5.0, grad=4.0)

    e = a * b
    d = e + c
    L = d * f
    print(f"Old L = {L.data}")

    learning_rate = 0.01

    for v in [a, b, c, f]:
        v.data -= learning_rate * v.grad

    e_new = a * b
    d_new = e_new + c
    L_new = d_new * f
    print(f"New L = {L_new.data}")


if __name__ == "__main__":
    main()
