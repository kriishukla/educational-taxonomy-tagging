# utils/data_loader.py
import re
import pandas as pd
from typing import List, Optional

def clean_text(text: str) -> str:
    """
    Clean raw text by removing HTML tags, unicode artifacts, and other noise.
    This function is generic and can be applied to questions, paragraphs,
    descriptions, or any free-form text field.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub('<[^>]*>', ' ', text)
    text = re.sub(' +', ' ', text)        
    text = re.sub('\xa0', '', text)
    text = re.sub('nan', '', text)
    text = re.sub(u'\u2004', '', text)
    text = re.sub(u'\u2009', '', text)
    text = re.sub('&nbsp', '', text)
    text = re.sub('&ndash', '', text)
    text = re.sub('\r', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('MathType@.*', '', text)
    text = re.sub('&thinsp', '', text)
    text = re.sub('&times', '', text)
    text = re.sub('\u200b', '', text)
    text = re.sub('&rarr;;;', '', text)

    return text.strip()

def load_text_dataframe(csv_path: str, text_column: str):
    """
    Load a CSV file and clean the specified text column.
    
    Args:
        csv_path (str): Path to the input CSV file.
        text_column (str): Column name containing text to clean.

    Returns:
        pd.DataFrame: DataFrame with cleaned text.
    """
    df = pd.read_csv(csv_path)

    # Validate column existence
    if text_column not in df.columns:
        raise ValueError(
            f"CSV must contain a '{text_column}' column. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean the column inline
    df[text_column] = df[text_column].astype(str).apply(clean_text)

    return df


def load_target_text(
    csv_path: str,
    target_column: str,
    keep_columns: Optional[List[str]] = None,
    top_n: Optional[int] = None
):
    """
    Load a CSV file and return a dataframe with text entries
    from the specified target column plus additional columns
    kept unchanged.

    Args:
        csv_path (str): Path to the input CSV file.
        target_column (str): Column name containing text.
        keep_columns (List[str], optional): Additional column names
            to keep unchanged in the final DataFrame.
        top_n (int, optional): Number of top rows to return based on the order in the CSV.
    Returns:
        pd.DataFrame: DataFrame with 'target' and kept columns.
    """
    df = pd.read_csv(csv_path)

    keep_columns = keep_columns or []

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(
            f"CSV must contain a '{target_column}' column. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate keep columns
    missing = [c for c in keep_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Rename target column
    df = df.rename(columns={target_column: "target"})

    # Select final columns
    final_columns = ["target"] + keep_columns

    if top_n is not None:
        df = df.head(top_n)

    return df[final_columns].astype({"target": str})


def load_taxonomy(taxonomy_file: str):
    """
    Load a taxonomy text file and clean each line inline.
    Hierarchies like 'Physics >> Mechanics >> Rotational Motion'
    are flattened into: 'Physics Mechanics Rotational Motion'
    """
    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    cleaned = []
    for line in raw_lines:
        line = ' '.join(line.split(">>"))  # remove hierarchy separators
        cleaned.append(line.strip())

    return cleaned

