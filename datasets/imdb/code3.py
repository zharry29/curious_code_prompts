import get_sentiment_from_text_review

class Review:
    """Given a user review, predict the whether it is positive or negative."""
    def __init__(self, review):
        self.review = review # The review that a user wrote
    def get_sentiment(self):
        # Whether the review is positive or negative
        return get_sentiment_from_text_review(self.review)

review = Review(review = "{text}")
assert(review.get_sentiment == {label})