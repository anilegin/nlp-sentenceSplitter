from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, f1_score, accuracy_score

from utils.preprocessing import SentenceSplitPreprocessor
from utils.featureExtractor import FeatureExtractor


class SentenceDataset(Dataset):
    def __init__(self, texts, num_features, labels, char2idx, max_len):
        self.texts = texts.fillna("").values
        self.num_features = torch.tensor(num_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        seq = [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in text]

        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]

        return torch.tensor(seq, dtype=torch.long), self.num_features[idx], self.labels[idx]


class CharCNNWithFeatures(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_numeric_features,
        embed_dim=64,
        num_filters=128,
        kernel_sizes=(3, 5, 7),
        hidden_dim=128,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        cnn_dim = num_filters * len(kernel_sizes)

        self.num_proj = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Linear(cnn_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_char, x_num):
        emb = self.embedding(x_char)
        emb = emb.transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            h = torch.max(h, dim=2).values
            conv_outs.append(h)

        h_char = torch.cat(conv_outs, dim=1)
        h_num = self.num_proj(x_num)

        h = torch.cat([h_char, h_num], dim=1)
        logits = self.head(h).squeeze(1)
        return logits


class BiLSTMWithFeatures(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_numeric_features,
        embed_dim=64,
        lstm_hidden_dim=128,
        lstm_layers=1,
        hidden_dim=128,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.num_proj = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Linear((2 * lstm_hidden_dim) + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_char, x_num):
        emb = self.embedding(x_char)
        lstm_out, _ = self.bilstm(emb)
        h_char = lstm_out.mean(dim=1)

        h_num = self.num_proj(x_num)
        h = torch.cat([h_char, h_num], dim=1)
        logits = self.head(h).squeeze(1)
        return logits


def get_predictions(model, data_loader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_char, x_num, yb in data_loader:
            x_char = x_char.to(device)
            x_num = x_num.to(device)

            logits = model(x_char, x_num)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def reconstruct_with_eos(clean_text, pred_df, eos_token="<EOS>"):
    boundary_indices = set(
        pred_df.loc[pred_df["ensemble_pred"] == 1, "char_idx"].tolist()
    )

    output_chars = []
    for i, ch in enumerate(clean_text):
        output_chars.append(ch)
        if i in boundary_indices:
            output_chars.append(eos_token)

    return "".join(output_chars)


def evaluate_sentence_splitter(
    language,
    raw_file_path,
    models_root=None,
    batch_size=128,
    output_dir=None,
    eos_token="<EOS>",
    window=50,
):
    language = language.lower()
    if language not in ["english", "italian"]:
        raise ValueError("language must be 'english' or 'italian'")

    if models_root is None:
        default_root = Path("/content/nlp-sentenceSplitter/models")
        models_root = default_root if default_root.exists() else Path("./models")
    else:
        models_root = Path(models_root)

    if output_dir is None:
        default_out = Path("/content/nlp-sentenceSplitter/results")
        output_dir = default_out if default_out.parent.exists() else Path("./results")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    lang_dir = models_root / language

    charcnn_path = lang_dir / f"{language}_charcnn_final.pt"
    bilstm_path = lang_dir / f"{language}_bilstm_final.pt"
    char2idx_path = lang_dir / f"{language}_char2idx.pkl"
    scaler_path = lang_dir / f"{language}_scaler.pkl"
    meta_info_path = lang_dir / f"{language}_meta.pkl"
    meta_clf_path = lang_dir / f"{language}_meta_clf.pkl"
    meta_thr_path = lang_dir / f"{language}_meta_threshold.pkl"

    preprocessor = SentenceSplitPreprocessor(eos_token=eos_token, window=window)
    processed = preprocessor.process_file(raw_file_path)

    eval_df = processed["df"]
    clean_text = processed["clean_text"]

    has_labels = "label" in eval_df.columns and eval_df["label"].notna().any()

    char2idx = joblib.load(char2idx_path)
    scaler = joblib.load(scaler_path)
    meta_info = joblib.load(meta_info_path)
    meta_clf = joblib.load(meta_clf_path)
    meta_threshold = joblib.load(meta_thr_path)

    max_len = meta_info["max_len"]
    vocab_size = meta_info["vocab_size"]
    num_numeric_features = meta_info["num_numeric_features"]

    extractor = FeatureExtractor()
    X_eval_feat = extractor.transform(eval_df)
    X_eval_num = scaler.transform(X_eval_feat)

    if "label" in eval_df.columns:
        eval_labels = eval_df["label"].fillna(0).astype(int).values
    else:
        eval_labels = np.zeros(len(eval_df), dtype=int)

    eval_dataset = SentenceDataset(
        texts=eval_df["centered_context"],
        num_features=X_eval_num,
        labels=eval_labels,
        char2idx=char2idx,
        max_len=max_len
    )

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    charcnn_model = CharCNNWithFeatures(
        vocab_size=vocab_size,
        num_numeric_features=num_numeric_features,
        embed_dim=64,
        num_filters=128,
        kernel_sizes=(3, 5, 7),
        hidden_dim=128,
        dropout=0.3
    )

    bilstm_model = BiLSTMWithFeatures(
        vocab_size=vocab_size,
        num_numeric_features=num_numeric_features,
        embed_dim=64,
        lstm_hidden_dim=128,
        lstm_layers=1,
        hidden_dim=128,
        dropout=0.3
    )

    charcnn_model.load_state_dict(torch.load(charcnn_path, map_location=device))
    bilstm_model.load_state_dict(torch.load(bilstm_path, map_location=device))

    charcnn_model.to(device).eval()
    bilstm_model.to(device).eval()

    charcnn_probs, charcnn_preds, labels_1 = get_predictions(
        charcnn_model, eval_loader, device
    )
    bilstm_probs, bilstm_preds, labels_2 = get_predictions(
        bilstm_model, eval_loader, device
    )

    X_meta_eval = np.column_stack([
        charcnn_probs,
        bilstm_probs,
        np.abs(charcnn_probs - bilstm_probs),
        (charcnn_probs + bilstm_probs) / 2,
        np.maximum(charcnn_probs, bilstm_probs),
        np.minimum(charcnn_probs, bilstm_probs),
    ])

    meta_probs = meta_clf.predict_proba(X_meta_eval)[:, 1]
    meta_preds = (meta_probs > meta_threshold).astype(int)

    pred_df = eval_df.copy()
    pred_df["charcnn_prob"] = charcnn_probs
    pred_df["bilstm_prob"] = bilstm_probs
    pred_df["ensemble_prob"] = meta_probs
    pred_df["ensemble_pred"] = meta_preds

    predicted_text = reconstruct_with_eos(clean_text, pred_df, eos_token=eos_token)

    raw_name = Path(raw_file_path).stem
    pred_csv_path = output_dir / f"{language}_{raw_name}_candidate_predictions.csv"
    split_txt_path = output_dir / f"{language}_{raw_name}_predicted_split.txt"

    pred_df.to_csv(pred_csv_path, index=False)
    with open(split_txt_path, "w", encoding="utf-8") as f:
        f.write(predicted_text)

    metrics = None
    if has_labels:
        metrics = {
            "charcnn_accuracy": accuracy_score(labels_1, charcnn_preds),
            "charcnn_f1": f1_score(labels_1, charcnn_preds),
            "bilstm_accuracy": accuracy_score(labels_1, bilstm_preds),
            "bilstm_f1": f1_score(labels_1, bilstm_preds),
            "ensemble_accuracy": accuracy_score(labels_1, meta_preds),
            "ensemble_f1": f1_score(labels_1, meta_preds),
            "ensemble_report": classification_report(labels_1, meta_preds, digits=4),
        }

    return {
        "language": language,
        "raw_file_path": str(raw_file_path),
        "models_root": str(models_root),
        "output_dir": str(output_dir),
        "has_labels": has_labels,
        "metrics": metrics,
        "pred_df": pred_df,
        "predicted_text": predicted_text,
        "candidate_predictions_path": str(pred_csv_path),
        "predicted_text_path": str(split_txt_path),
    }