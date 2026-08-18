import os
import math
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from models.base import BaseTagger
from taxonomy.taxonomy import Taxonomy

# -------------------------------------------------------------------
#                   MULTICLASS CLASSIFIER MODEL
# -------------------------------------------------------------------
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F


class MHSA(nn.Module):
    def __init__(self, emb_dim, kqv_dim, num_heads=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.kqv_dim = kqv_dim
        self.num_heads = num_heads

        self.w_k = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_q = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_v = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_out = nn.Linear(kqv_dim * num_heads, emb_dim)

    def forward(self, query, key, value):
        b, t = query.shape
        e = self.kqv_dim
        h = self.num_heads

        keys = self.w_k(key).view(b, h, e)
        values = self.w_v(value).view(b, h, e)
        queries = self.w_q(query).view(b, h, e)

        dot = queries @ keys.transpose(2, 1)
        dot = dot / (e ** 0.5)
        dot = F.softmax(dot, dim=2)
        out = dot @ values

        out = out.contiguous().view(b, h * e)
        return self.w_out(out)


class MulticlassClassifier(nn.Module):
    def __init__(self, bert_model_path):
        super(MulticlassClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            bert_model_path,
            output_hidden_states=True,
            output_attentions=False
        )

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(576, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.act = nn.ReLU()

        self.multi_head_attention = MHSA(1024, 64, 16)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=1024,
            num_heads=4,
            batch_first=True
        )

    def forward(self, tokens, masks, taxonomy_vectors):
        outputs = self.bert(tokens, attention_mask=masks)[2]

        # last layer hidden states
        output_1 = outputs[-1].permute(1, 0, 2)
        output_1 = torch.mean(output_1, dim=0)

        pooled_output = outputs[-1]  # shape: (batch, seq, 768)

        x = self.fc1(pooled_output)
        
        cos_label = nn.CosineSimilarity(dim=1, eps=1e-5)

        # SAME TARGET-CURRENT logic as original code:
        targets_curr_batch = []
        for index_1, input_x in enumerate(x):
            distance = cos_label(torch.mean(input_x, dim=0).reshape(1, -1), taxonomy_vectors)
            distances, indices = torch.topk(distance, 1, largest=True)

            target_distances = (
                F.normalize(taxonomy_vectors[indices], p=2, dim=1)
                - F.normalize(taxonomy_vectors, p=2, dim=1)
            ).pow(2).sum(1)

            distances, indices = torch.topk(target_distances, 1, largest=False)
            targets_curr_batch.append(taxonomy_vectors[indices].reshape(1, 1, 1024))

        targets_batch = torch.cat(targets_curr_batch, dim=0)

        attn_output, _ = self.multihead_attn(targets_batch, x, x)
        x = torch.sum(attn_output, dim=1)
        return x


# ===================================================================
#                           TAGREC CLASS
# ===================================================================

class TagRec(BaseTagger):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.taxonomy: Taxonomy = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = MulticlassClassifier(bert_model_path="bert-base-uncased").to(self.device).eval()
    
    def set_taxonomy(self, taxonomy: Taxonomy): 
        self.taxonomy = taxonomy
        cleaned_taxonomy = self.__clean_taxonomy()
        self.taxonomy_vectors = torch.tensor(
            SentenceTransformer("bert-large-nli-stsb-mean-tokens").encode(cleaned_taxonomy),
            dtype=torch.float32,
            device=self.device
        )
        
    def __clean_taxonomy(self):
        cleaned_taxonomy = []
        for leaf in self.taxonomy.leaf_nodes:
            path = leaf.get_path()
            cleaned_taxonomy.append("  ".join(path))
        return cleaned_taxonomy
    
    # -----------------------------------------------------------
    # LOAD MODEL + ARTIFACTS
    # -----------------------------------------------------------
    def load(self, path: str = "models/weights/tagrec.zip"):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        print("✅ TagRec loaded successfully.")

    # -----------------------------------------------------------
    # PREDICT TOP-K TAXONOMY LABELS
    # -----------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        top_k: int = 3,
        batch_size: int = 32,
        show_scores: bool = False
    ) -> pd.DataFrame:
        """
        Fast batched inference version.
        Returns long-form dataframe: top_k × input_rows
        Keeps all original columns from input df.
        """

        if self.taxonomy is None:
            raise ValueError("Taxonomy not set. Please set the taxonomy before prediction.")

        results = []
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

        num_batches = math.ceil(len(df) / batch_size)

        for b in range(num_batches):
            batch_df = df.iloc[b * batch_size : (b + 1) * batch_size]
            batch_texts = batch_df["target"].tolist()

            # -------- Tokenize --------
            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            # -------- Model forward --------
            with torch.no_grad():
                batch_vecs = self.model(
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    self.taxonomy_vectors
                )  # (batch, 1024)

            # -------- Cosine similarity --------
            sims = cos(
                batch_vecs.unsqueeze(1),
                self.taxonomy_vectors.unsqueeze(0)
            )  # (batch, num_taxonomy)

            top_scores, top_idx = torch.topk(sims, top_k, dim=1)

            # -------- Build output rows --------
            for i in range(len(batch_df)):
                row_meta = batch_df.iloc[i].to_dict()

                for k in range(top_k):
                    label = self.taxonomy.leaf_nodes[int(top_idx[i, k].item())]
                    score = float(top_scores[i, k].item())

                    out_row = dict(row_meta)  # keep ALL original columns
                    out_row["predicted_taxonomy"] = label

                    if show_scores:
                        out_row["prediction_score"] = score

                    results.append(out_row)

        return pd.DataFrame(results)


