import get_most_likely_ending_index

class Passage:
    def __init__(self, context, possible_endings):
        self.context = context
        self.possible_endings = possible_endings
    def best_ending(self):
        return get_most_likely_ending_index(self.context, self.possible_endings)

passage0 = Passage(
    context = "{ctx}"
    possible_endings = [
            "{ending0}",
            "{ending1}",
            "{ending2}",
            "{ending3}"
        ]
)
assert(passage0.best_ending == {label})