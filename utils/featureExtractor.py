import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.quote_chars = {'"', "'", "”", "’", "«", "»", "‹", "›"}
        self.closing_brackets = {")", "]", "}"}
        self.punct_chars = {".", "?", "!", ",", ";", ":"}

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _safe_last_char(text):
        return text[-1] if text else ""

    @staticmethod
    def _safe_first_char(text):
        return text[0] if text else ""

    @staticmethod
    def _count_trailing_set(text, valid_chars):
        count = 0
        for ch in reversed(text):
            if ch in valid_chars:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _count_leading_set(text, valid_chars):
        count = 0
        for ch in text:
            if ch in valid_chars:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _is_title_token(token):
        return int(bool(token) and token[0].isupper() and token[1:].islower())

    def transform(self, X):
        df = X.copy()

        left = df["left_context"].fillna("")
        right = df["right_context"].fillna("")
        punct = df["punct"].fillna("")
        token_before = df["token_before"].fillna("")
        token_after = df["token_after"].fillna("")

        prev_char = left.apply(self._safe_last_char)
        next_char = right.apply(self._safe_first_char)

        out = pd.DataFrame(index=df.index)

        # punctuation identity
        out["is_period"] = (punct == ".").astype(int)
        out["is_question"] = (punct == "?").astype(int)
        out["is_exclamation"] = (punct == "!").astype(int)
        out["is_newline_candidate"] = (punct == "\n").astype(int)

        # immediate char context
        out["prev_is_digit"] = prev_char.str.isdigit().astype(int)
        out["next_is_digit"] = next_char.str.isdigit().astype(int)
        out["prev_is_alpha"] = prev_char.str.isalpha().astype(int)
        out["next_is_alpha"] = next_char.str.isalpha().astype(int)
        out["prev_is_space"] = prev_char.apply(lambda x: int(x.isspace()) if x else 0)
        out["next_is_space"] = next_char.apply(lambda x: int(x.isspace()) if x else 0)
        out["prev_is_upper"] = prev_char.str.isupper().astype(int)
        out["next_is_upper"] = next_char.str.isupper().astype(int)
        out["prev_is_lower"] = prev_char.str.islower().astype(int)
        out["next_is_lower"] = next_char.str.islower().astype(int)
        out["next_is_newline"] = (next_char == "\n").astype(int)

        # token-level features
        out["token_before_len"] = token_before.str.len()
        out["token_after_len"] = token_after.str.len()
        out["token_before_is_short"] = (token_before.str.len() <= 3).astype(int)
        out["token_before_is_upper"] = token_before.str.isupper().astype(int)
        out["token_before_is_title"] = token_before.apply(self._is_title_token)
        out["token_after_is_upper"] = token_after.str.isupper().astype(int)
        out["token_after_is_title"] = token_after.apply(self._is_title_token)
        out["token_before_has_digit"] = token_before.str.contains(r"\d", regex=True).astype(int)
        out["token_after_has_digit"] = token_after.str.contains(r"\d", regex=True).astype(int)

        # useful patterns
        out["is_decimal_pattern"] = (
            prev_char.str.isdigit() & next_char.str.isdigit()
        ).astype(int)

        out["is_possible_abbrev"] = (
            (punct == ".") &
            (token_before.str.len() <= 4) &
            (token_before.str.contains(r"^[A-Za-z]+$", regex=True))
        ).astype(int)

        out["next_token_starts_upper"] = token_after.apply(
            lambda x: int(bool(x) and x[0].isupper())
        )

        # quotes / brackets after punctuation
        out["leading_quotes_after"] = right.apply(
            lambda x: self._count_leading_set(x, self.quote_chars)
        )
        out["leading_closing_brackets_after"] = right.apply(
            lambda x: self._count_leading_set(x, self.closing_brackets)
        )
        out["has_quote_after"] = (out["leading_quotes_after"] > 0).astype(int)
        out["has_closing_bracket_after"] = (out["leading_closing_brackets_after"] > 0).astype(int)

        # punctuation clusters
        out["prev_is_punct"] = prev_char.isin(self.punct_chars).astype(int)
        out["next_is_punct"] = next_char.isin(self.punct_chars).astype(int)
        out["trailing_punct_before"] = left.apply(
            lambda x: self._count_trailing_set(x, self.punct_chars)
        )
        out["leading_punct_after"] = right.apply(
            lambda x: self._count_leading_set(x, self.punct_chars)
        )

        # newline / layout cues
        out["newline_in_left_10"] = left.apply(lambda x: int("\n" in x[-10:]))
        out["newline_in_right_10"] = right.apply(lambda x: int("\n" in x[:10]))
        out["double_newline_in_right_20"] = right.apply(lambda x: int("\n\n" in x[:20]))

        return out