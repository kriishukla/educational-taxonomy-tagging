import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr


class RectificationEvaluator:
    """
    Evaluate LLM rectification performance against gold scores.
    """

    def __init__(self, df: pd.DataFrame, score_columns: list[str]):
        self.df = df.copy()
        self.score_columns = score_columns

    # ---------------------------
    # helpers
    # ---------------------------
    def _cols(self, prefix):
        return {
            "llm": f"{prefix}_llm",
            "rect": f"{prefix}_rectified",
            "gold": f"{prefix}_gold",
            "ci": f"{prefix}_ci_size",
        }

    def _validate_columns(self, cols):
        missing = [c for c in cols.values() if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    # ---------------------------
    # core metrics
    # ---------------------------
    @staticmethod
    def _mae(a, b):
        return mean_absolute_error(a, b)

    @staticmethod
    def _rmse(a, b):
        return root_mean_squared_error(a, b)

    @staticmethod
    def _spearman(a, b):
        return spearmanr(a, b).correlation

    # ---------------------------
    # rectification metrics
    # ---------------------------
    @staticmethod
    def _improvement_rate(llm, rect, gold):
        return np.mean(np.abs(rect - gold) < np.abs(llm - gold))

    @staticmethod
    def _directional_correctness(llm, rect, gold):
        move_direction = np.sign(rect - llm)
        true_direction = np.sign(gold - llm)
        return np.mean(move_direction == true_direction)

    @staticmethod
    def _confidence_weighted_error(rect, gold, ci):
        ci_safe = np.clip(ci, 1e-8, None)
        return np.mean(np.abs(rect - gold) / ci_safe)

    # ---------------------------
    # evaluation per score
    # ---------------------------
    def evaluate_score(self, prefix: str) -> dict:

        cols = self._cols(prefix)
        self._validate_columns(cols)

        data = self.df[list(cols.values())].dropna()

        llm = data[cols["llm"]].values
        rect = data[cols["rect"]].values
        gold = data[cols["gold"]].values
        ci = data[cols["ci"]].values

        mae_llm = self._mae(gold, llm)
        mae_rect = self._mae(gold, rect)

        rmse_llm = self._rmse(gold, llm)
        rmse_rect = self._rmse(gold, rect)

        results = {
            "MAE_llm": mae_llm,
            "MAE_rectified": mae_rect,
            "MAE_delta": mae_llm - mae_rect,

            "RMSE_llm": rmse_llm,
            "RMSE_rectified": rmse_rect,
            "RMSE_delta": rmse_llm - rmse_rect,

            "Spearman_llm": self._spearman(llm, gold),
            "Spearman_rectified": self._spearman(rect, gold),
            "Spearman_gain":
                self._spearman(rect, gold)
                - self._spearman(llm, gold),

            "Improvement_rate":
                self._improvement_rate(llm, rect, gold),

            "Error_reduction_%":
                (mae_llm - mae_rect) / mae_llm
                if mae_llm > 0 else np.nan,

            "Directional_correctness":
                self._directional_correctness(llm, rect, gold),

            "Confidence_weighted_error":
                self._confidence_weighted_error(rect, gold, ci),
        }

        return results

    # ---------------------------
    # evaluate all scores
    # ---------------------------
    def evaluate_all(self) -> pd.DataFrame:
        rows = {}

        for prefix in self.score_columns:
            rows[prefix] = self.evaluate_score(prefix)

        return pd.DataFrame(rows).T