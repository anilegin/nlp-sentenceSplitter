from pathlib import Path
import re

import pandas as pd


class SentenceSplitPreprocessor:
    def __init__(self, eos_token="<EOS>", candidate_punct=None, window=50):
        self.eos_token = eos_token
        self.candidate_punct = candidate_punct or {".", "?", "!"}

        self.window = window

        # these may appear after the real sentence-ending punctuation
        self.trailing_closers = {'"', "'", "”", "’", "»", "›", ")", "]", "}"}

    @staticmethod
    def read_text(file_path):
        return Path(file_path).read_text(encoding="utf-8")

    def _find_boundary_punct(self, clean_chars):
        # walk back over spaces and closing symbols to find the real eos punctuation
        j = len(clean_chars) - 1

        while j >= 0 and clean_chars[j].isspace():
            j -= 1

        while j >= 0 and clean_chars[j] in self.trailing_closers:
            j -= 1

        if j >= 0 and clean_chars[j] in self.candidate_punct:
            return j

        return None

    @staticmethod
    def _get_token_before(text, idx):
        j = idx - 1
        while j >= 0 and re.match(r"[\w'’-]", text[j], flags=re.UNICODE):
            j -= 1
        return text[j + 1:idx]

    @staticmethod
    def _get_token_after(text, idx):
        j = idx + 1
        while j < len(text) and text[j].isspace():
            j += 1

        start = j
        while j < len(text) and re.match(r"[\w'’-]", text[j], flags=re.UNICODE):
            j += 1

        return text[start:j]

    def extract_clean_text_and_boundaries(self, raw_text):
        clean_chars = []
        boundary_punct_indices = set()
        non_punct_boundary_indices = []

        i = 0
        while i < len(raw_text):
            if raw_text.startswith(self.eos_token, i):
                boundary_idx = self._find_boundary_punct(clean_chars)

                if boundary_idx is not None:
                    boundary_punct_indices.add(boundary_idx)
                else:
                    # boundary exists, but not attached to . ? !
                    non_punct_boundary_indices.append(len(clean_chars))

                # remove eos completely from model-visible text
                i += len(self.eos_token)
            else:
                clean_chars.append(raw_text[i])
                i += 1

        clean_text = "".join(clean_chars)
        return clean_text, boundary_punct_indices, non_punct_boundary_indices

    def extract_candidates(self, clean_text, boundary_punct_indices, doc_id):
        rows = []

        for idx, ch in enumerate(clean_text):
            if ch not in self.candidate_punct:
                continue

            left_start = max(0, idx - self.window)
            right_end = min(len(clean_text), idx + self.window + 1)

            rows.append({
                "doc_id": doc_id,
                "char_idx": idx,
                "punct": ch,
                "left_context": clean_text[left_start:idx],
                "right_context": clean_text[idx + 1:right_end],
                "centered_context": clean_text[left_start:right_end],
                "token_before": self._get_token_before(clean_text, idx),
                "token_after": self._get_token_after(clean_text, idx),
                "label": 1 if idx in boundary_punct_indices else 0,
            })

        return rows

    def process_document(self, raw_text, doc_id):
        clean_text, boundary_punct_indices, non_punct_boundary_indices = self.extract_clean_text_and_boundaries(raw_text)
        rows = self.extract_candidates(clean_text, boundary_punct_indices, doc_id)
        df = pd.DataFrame(rows)

        return {
            "df": df,
            "clean_text": clean_text,
            "boundary_punct_indices": boundary_punct_indices,
            "non_punct_boundary_indices": non_punct_boundary_indices,
        }

    def process_file(self, file_path):
        raw_text = self.read_text(file_path)
        doc_id = Path(file_path).stem
        return self.process_document(raw_text, doc_id)