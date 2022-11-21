import get_star_rating_from_text_review

class Review:
    """Given a user review, predict the rating the user gave."""
    def __init__(self, review):
        self.review = review # The review that a user wrote
    def get_rating(self):
        # The star rating the user gave
        return get_star_rating_from_text_review(self.review)

review = Review(review = "{text}")
assert(review.get_rating ==$ {label})