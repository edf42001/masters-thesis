class Rule:
    """
    A rule consists of a set of probabilistic strips operators. Also probably a context and deictic references,
    but lets start here

    For example:
    0.6: ~on(x, y)
    0.4: no_change
    """

    def __init__(self):
        self.rule = []
