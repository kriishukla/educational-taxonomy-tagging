import json
import pandas as pd
from typing import Optional
from llm.message import TAXONOMY_JUDGE_MESSAGE
from llm.lm import LM
from taxonomy.taxonomy import TaxonomyNode

class LLMJudge:
    def __init__(self, model: str, temperature: float = 0.15):
        self.llm = LM(model=model, temperature=temperature)

    def __build_messages(self, text: str, taxonomy: dict):
        """
        Builds message payload for taxonomy judging.
        """
        return TAXONOMY_JUDGE_MESSAGE + [
            {
                "role": "user",
                "content": f"""
Text:
{text}

Taxonomy:
{json.dumps(taxonomy, ensure_ascii=False, indent=2)}
"""
            }
        ]

    def __safe_json_load(self, response: str) -> dict:
        """
        Safely parse LLM JSON output.
        """
        try:
            return json.loads(response)
        except Exception:
            # fallback: try extracting JSON block
            try:
                start = response.index("{")
                end = response.rindex("}") + 1
                return json.loads(response[start:end])
            except Exception:
                return {}

    def judge_taxonomy_predictions(
        self,
        predictions: pd.DataFrame,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Judge taxonomy validity using LLM.

        Args:
            predictions (pd.DataFrame):
                - target (str)
                - predicted_taxonomy (dict or json str)

            top_n (int, optional):
                Only evaluate first N rows

        Returns:
            pd.DataFrame with additional columns:
                <taxonomy_level>_validity_llm
        """

        df = predictions.copy()

        if top_n is not None:
            df = df.head(top_n)

        all_rows_outputs = []

        for idx, row in df.iterrows():
            print(f"Judging row {idx}...")
            text = row["target"]
            taxonomy_raw = row["predicted_taxonomy"]

            # ensure taxonomy is dict
            if isinstance(taxonomy_raw, TaxonomyNode):
                taxonomy_dict = taxonomy_raw.to_json()

            messages = self.__build_messages(text, taxonomy_dict)
            response = self.llm(messages)
            scores = self.__safe_json_load(response)

            # build row result
            row_output = {}

            for level in taxonomy_dict.keys():
                col_name = f"{level}_validity_llm"
                row_output[col_name] = scores.get(level, None)

            all_rows_outputs.append(row_output)

        validity_df = pd.DataFrame(all_rows_outputs, index=df.index)

        df = pd.concat([df, validity_df], axis=1)

        return df