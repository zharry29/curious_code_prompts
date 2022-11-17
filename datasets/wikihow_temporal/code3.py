import order_steps

class Event:
    """Given a goal and two steps, predict the correct order to do the two steps to achieve the goal."""
    def __init__(self, goal, step0, step1):
        self.goal = goal # The goal that someone is trying to accomplish
        self.step0 = step0 # One of the steps that needs to be taken
        self.step1 = step1 # Another one of the steps that needs to be taken
    def get_order_of_steps(self):
        # Output a list of correct order of those two steps to be taken
        return order_steps(self.goal, self.step0, self.step1)

event{x} = Event(
    goal = "{goal}"
    step0 = "{step0}"
    step1 = "{step1}"
)
assert(event{x}.get_order_of_steps == [input{label_first}, input{label_after}])