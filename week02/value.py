class Value:
    def __init__(self, data=0.0, _prev=None):
        self.data = data
        self._prev = _prev

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        new_data = self.data + other.data
        return Value(new_data, {self, other})

    def __mul__(self, other):
        new_data = self.data * other.data
        return Value(new_data, {self, other})
