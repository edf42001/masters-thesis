from symbolic_stochastic_domains.term import Term


class State:
    """A state is a collection of terms, negated or not"""

    def __init__(self):
        self.terms = []

        # For quick lookup of term changes
        # Only updated when we want to compare terms
        self.term_dict = dict()

    def add_term(self, term):
        self.terms.append(term)

    def add_terms(self, terms):
        self.terms.extend(terms)

    def update_term_dict(self):
        # Creates a dictionary with strings as keys to identify each term and the truth value of the term
        self.term_dict = {t.get_unique_id(): not t.get_negated() for t in self.terms}

    def __repr__(self):
        return str(self.terms)

    def __getitem__(self, item) -> Term:
        # If accessing with int, return in order, otherwise, do a lookup in the dictionary
        if isinstance(item, int):
            return self.terms[item]
        else:
            return self.term_dict[item]

    def __contains__(self, item):
        return item in self.term_dict

    @staticmethod
    def get_changed_terms(s1, s2):
        """
        Given a start state and end state, returns which terms changed to either true or negated
        Assumes all the terms exist in both states, only negation status can change
        """

        # Make sure dicts are up to date
        s1.update_term_dict()
        s2.update_term_dict()

        # If the value of the term changed, return it in the dict with its name and new value
        return {term:s2[term] for term in s1.term_dict if s1[term] != s2[term]}
