"""Given a goal and two steps, predict the correct order to do the two steps to achieve the goal."""

# The goal that someone is trying to accomplish
goal = "{goal}"
# One of the steps that needs to be taken
step0 = "{step0}" 
# Another one of the steps that needs to be taken
step1 = "{step1}" 
# The list of correct order of those two steps to be taken
order_of_execution = $[input{label_first}, input{label_after}]