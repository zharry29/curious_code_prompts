import summarize

class Article:
    """Output a TL;DR of the article"""
    def __init__(self, article):
        self.article = article # The news article
    def get_tldr(self):
        # A TL;DR of the article
        return summarize(self.article)

article = Article(
    article = "{article}"
)
assert(article.get_tldr ==<\split> "{highlights}")