import question_answering

class Context:
    """Answer the question depending on the context"""
    def __init__(self, context):
        self.context = context # The context
        self.question = question # The question

    def get_answer(self):
        # If you can't find the answer, please respond "unanswerable"
        answer = question_answering(self.question, self.context)
        return answer if len(answer) > 0 else "unanswerable"

context = Context(
    context = "{context}",
    question = "{question}"
)
assert(context.get_answer ==<\split> "{answer}")