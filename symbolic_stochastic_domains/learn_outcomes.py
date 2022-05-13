def learn_outcomes(examples, outcomes):
    """
    Learns a minimal set of outcomes by combining and dropping outcomes until
    everything is explained as small as possible
    """

    # Step one: Use add operator to "pick a pair of non-contradictory outcomes in the set and create
    # a new outcome that is their conjunction"

    # Step 2:
    # drops an outcome from the set. Outcomes can only be dropped if they were overlapping
    # with other outcomes on every example they cover, otherwise the outcome set would not remain proper
    print(examples)
    print(outcomes)

    # To cover an example means if you apply the outcome it explains the example
    # i.e. if you apply heads(c1), heads(c2) to HT, you get HH, which explains a result of HH
    # To test for this, every change from s1 -> s2 must be listed in the outcome

    # An outcome is removed if every state change it covers is already covered by other things
    # For the no change case, find every example it covers. Then, see if it is covered by another outcome
    # If this is true for all, then it can be removed

    # Add combines non contradictory outcomes into a new one. For example, heads(c1) and tails(c1) contradict

    return outcomes