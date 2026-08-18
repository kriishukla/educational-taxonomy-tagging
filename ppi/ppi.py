import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import torch
from scipy.stats import t, norm


class LPPI:
    """
    Local Prediction-Powered Inference (LPPI)
    - Uses local k-NN calibration around each test point.
    - Uses Satterthwaite formula for degrees of freedom.
    - Uses t CI for small df, z CI for large df.
    """

    def __init__(self, 
                 k=10, 
                 confidence=0.90, 
                 embedder_model="all-MiniLM-L6-v2", 
        ):
        """
        Initializes LPPI.

        Args:
            k (int): number of neighbor gold points for local calibration.
            confidence (float): confidence level for CI.
            model_name (str): SBERT embedding model.
        """
        self.k = k
        self.alpha = 1 - confidence
        self.embedder_model = embedder_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model '{embedder_model}' on {self.device}...")
        self.embedder = SentenceTransformer(embedder_model, device=self.device)
        self.knn = None          # will hold NearestNeighbors model
        self.gold_df = None     # will store gold dataframe
        self.score_cols = []     # list of score columns
        self.gold_embeddings = None


    # ----------------------------------------------------------------------
    # FIT: builds KNN on gold embeddings & computes residuals
    # ----------------------------------------------------------------------
    def fit(self,
            gold_df: pd.DataFrame,
            target_text: str = "target",
            score_cols: list[str] = [],
        ):
        """
        Fit LPPI on gold data.

        gold_df columns:
            target_text
            <score>_gold
            <score>_llm

        Args:
            target_text (str): column for embeddings
            gold_df (DataFrame): gold dataset containing texts and scores
            score_cols (list[str]): base names of score columns
        """
        print("Fitting LPPI...")

        self.score_cols = score_cols
        self.gold_df = gold_df.copy()

        # Compute residuals for each score column
        for score_col in score_cols:
            gold_col = f"{score_col}_gold"
            llm_col = f"{score_col}_llm"

            if gold_col not in self.gold_df:
                raise ValueError(f"Missing column: {gold_col}")
            if llm_col not in self.gold_df:
                raise ValueError(f"Missing column: {llm_col}")

            self.gold_df[f"residual_{score_col}"] = (
                self.gold_df[llm_col] - self.gold_df[gold_col]
            )

        # --- embedding gold texts ---
        print("Embedding gold text...")
        self.gold_embeddings = self.embedder.encode(
            self.gold_df[target_text].tolist(), show_progress_bar=True
        )

        # --- KNN ---
        self.knn = NearestNeighbors(n_neighbors=self.k, metric="cosine")
        self.knn.fit(self.gold_embeddings)

        print("LPPI Fit complete.")


    # ----------------------------------------------------------------------
    # RECTIFY: compute local delta_hat, SE, CI for one test point
    # ----------------------------------------------------------------------
    def __rectify(self, neighbor_residuals):
        """
        Performs local rectification for one (test) point.

        Uses:
        - delta_hat = mean residual
        - SE = sqrt(var / k)
        - df from Satterthwaite
        - t or z CI

        Args:
            neighbor_residuals: array-like residuals of nearest gold points

        Returns:
            (delta_hat, se, ci_low, ci_high)
        """
        r = np.array(neighbor_residuals)
        m = len(r)

        delta_hat = r.mean()

        # sample variance (ddof=1)
        if m > 1:
            s2 = r.var(ddof=1)
        else:
            # no variance available
            s2 = 0.0

        se = np.sqrt(s2 / m) if s2 > 0 else 0.0

        # df (Satterthwaite)
        if s2 > 0 and m > 1:
            df = (s2 / m) ** 2 / ((s2 ** 2) / ((m ** 2) * (m - 1)))
        else:
            df = np.inf

        # t or z
        if df > 30:       # large df → z-score
            crit = norm.ppf(1 - self.alpha / 2)
        else:
            crit = t.ppf(1 - self.alpha / 2, df)

        ci_size = crit * se
        return delta_hat, se, ci_size

    # ----------------------------------------------------------------------
    # CALIBRATE: rectify LLM predictions on new data
    # ----------------------------------------------------------------------
    def calibrate(self, df, target_text = "target", clip_range=None):
        """
        calibrate rectified scores.

        df columns:
            target_text
            <score>_llm

        Returns:
            DataFrame with:
                y_rect_<score>
                ci_low_<score>
                ci_high_<score>
        """
        if self.knn is None:
            raise RuntimeError("Must call fit() before predict().")

        print("Embedding unlabeled text...")
        test_embeddings = self.embedder.encode(
            df[target_text].tolist(), show_progress_bar=True
        )

        print("Finding nearest gold neighbors...")
        distances, idx = self.knn.kneighbors(test_embeddings)

        out = df.copy()

        # create result columns
        for score_col in self.score_cols:
            out[f"{score_col}_rectified"] = np.nan
            out[f"{score_col}_ci_size"] = np.nan


        print("Applying local rectification...")
        for i in range(len(df)):
            gold_rows = self.gold_df.iloc[idx[i]]

            for score_col in self.score_cols:
                residuals = gold_rows[f"residual_{score_col}"].values

                delta_hat, se, ci_size = self.__rectify(residuals)

                f_x0 = df.iloc[i][f"{score_col}_llm"]
                y_rect = f_x0 - delta_hat

                out.at[i, f"{score_col}_rectified"] = np.clip(y_rect, clip_range[0], clip_range[1]) if clip_range is not None else y_rect
                out.at[i, f"{score_col}_ci_size"] = ci_size
        return out
