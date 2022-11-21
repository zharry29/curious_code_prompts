import find_goal

class Event:
    """Given a step and four possible goals, predict which goal the step is logically trying to achieve."""
    def __init__(self, step, goal0, goal1, goal2, goal3):
         # The step someone takes to achieve some goal
        self.step = step
        # The four possible goals
        self.goal0 = goal0
        self.goal1 = goal1
        self.goal2 = goal2
        self.goal3 = goal3
    def get_goal_from_step(self):
        # Output the correct goal that the step is trying to accomplish
        return find_goal(self.step, [self.goal0, self.goal1, self.goal2, self.goal3])

event = Event(
    step = "{step}",
    goal0 = "{goal0}",
    goal1 = "{goal1}",
    goal2 = "{goal2}",
    goal3 = "{goal3}"
)
assert(event.get_goal_from_step == goal${label})