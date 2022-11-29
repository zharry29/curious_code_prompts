import summarize

class Article:
    """Output the main points one should remember from this news article in two or three sentences"""
    def __init__(self, article):
        self.article = article # The news article
    def get_main_points(self):
        # Output the main points one should remember from the news article
        return summarize(self.article)

article = Article(
    article = "{article}"
)
assert(article.get_main_points ==<\split> "{highlights}")