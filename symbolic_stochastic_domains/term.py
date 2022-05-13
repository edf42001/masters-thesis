class Term:
    """
    Terms are things like on(x, y), inhand-nil, table(x)
    They can be inverted and have multiple arguments
    """

    def __init__(self, name="", negated=False, **kwargs):
        self.name = name
        self.negated = negated

        self.kwargs = kwargs

    def get_negated(self) -> bool:
        return self.negated

    def true(self) -> bool:
        return not self.negated

    def get_unique_id(self) -> str:
        """
        Returns a string representation of this term which will
        be unique due to the name and arguments of the term
        """
        ret = self.name

        if self.kwargs:
            ret += "(" + ", ".join(self.kwargs.values()) + ")"

        return ret

    def __repr__(self) -> str:
        """The name but with a tilde if it is negated"""

        ret = self.get_unique_id()

        if self.negated:
            ret = "~" + ret

        return ret
