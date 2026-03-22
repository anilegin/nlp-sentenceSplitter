from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2


class TfidfFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        text_col="centered_context",
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        lowercase=False,
        use_chi2=False,
        k_best=5000
    ):
        self.text_col = text_col
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.lowercase = lowercase
        self.use_chi2 = use_chi2
        self.k_best = k_best

        self.vectorizer_ = None
        self.selector_ = None

    def fit(self, X, y=None):
        texts = X[self.text_col].fillna("").astype(str)

        self.vectorizer_ = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            lowercase=self.lowercase
        )

        X_tfidf = self.vectorizer_.fit_transform(texts)

        if self.use_chi2:
            if y is None:
                raise ValueError("y is required when use_chi2=True")

            k = min(self.k_best, X_tfidf.shape[1])
            self.selector_ = SelectKBest(score_func=chi2, k=k)
            self.selector_.fit(X_tfidf, y)

        return self

    def transform(self, X):
        if self.vectorizer_ is None:
            raise RuntimeError("fit must be called before transform")

        texts = X[self.text_col].fillna("").astype(str)
        X_tfidf = self.vectorizer_.transform(texts)

        if self.selector_ is not None:
            X_tfidf = self.selector_.transform(X_tfidf)

        return X_tfidf

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self):
        if self.vectorizer_ is None:
            raise RuntimeError("fit must be called before get_feature_names_out")

        feature_names = self.vectorizer_.get_feature_names_out()

        if self.selector_ is not None:
            mask = self.selector_.get_support()
            feature_names = feature_names[mask]

        return feature_names