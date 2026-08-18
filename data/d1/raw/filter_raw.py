import pandas as pd
from typing import Set

def filter_and_shuffle_csv(
    taxonomy_path: str,
    csv_path: str,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Filters a CSV based on chapters present in a taxonomy file,
    prints missing chapters, and returns a shuffled DataFrame.
    """

    # ---------- Read taxonomy ----------
    taxonomy_chapters: Set[str] = set()

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Ignore first line
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(">>")]
        if len(parts) >= 2:
            chapter = parts[1].lower()
            taxonomy_chapters.add(chapter)

    # ---------- Read CSV ----------
    df = pd.read_csv(csv_path)

    if "chapter" not in df.columns:
        raise ValueError("CSV must contain a 'chapter' column")

    # Normalize chapter column
    df["chapter_normalized"] = df["chapter"].str.strip().str.lower()

    # ---------- Filter ----------
    filtered_df = df[df["chapter_normalized"].isin(taxonomy_chapters)].copy()

    # ---------- Find missing taxonomy chapters ----------
    csv_chapters = set(filtered_df["chapter_normalized"].unique())
    missing_chapters = taxonomy_chapters - csv_chapters

    # ---------- Print missing ----------
    if missing_chapters:
        print("Chapters present in taxonomy but NOT in CSV:")
        for ch in sorted(missing_chapters):
            print(f"- {ch}")
    else:
        print("All taxonomy chapters are present in the CSV.")

    # ---------- Cleanup & Shuffle ----------
    filtered_df.drop(columns=["chapter_normalized"], inplace=True)
    shuffled_df = filtered_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return shuffled_df


df1 = filter_and_shuffle_csv(
    taxonomy_path="../../../taxonomy/PCv3.txt",
    csv_path="test_raw.csv"
)

df2 = filter_and_shuffle_csv(
    taxonomy_path="../../../taxonomy/PCv3.txt",
    csv_path="train_raw.csv"
)

df3 = filter_and_shuffle_csv(
    taxonomy_path="../../../taxonomy/PCv3.txt",
    csv_path="val_raw.csv"     
)

df1.to_csv("test.csv", index=False)
df2.to_csv("questions.csv", index=False)
df3.to_csv("lppi.csv", index=False)

