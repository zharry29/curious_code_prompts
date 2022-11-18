import get_most_likely_ending_index

class Passage:
    """Given a context and 4 different endings, output the most likely ending"""
    def __init__(self, context, possible_endings):
        self.context = context # The context
        self.possible_endings = possible_endings # The 4 endings
    def best_ending(self):
        # Choose the most likely ending to the context
        return get_most_likely_ending_index(self.context, self.possible_endings)

passage = Passage(
    context = "{ctx}"
    possible_endings = [
            "{ending0}",
            "{ending1}",
            "{ending2}",
            "{ending3}"
        ]
)
assert(passage.best_ending ==$ {label})