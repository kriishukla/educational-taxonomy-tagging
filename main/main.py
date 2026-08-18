from utils.data_loader import load_target_text
from taxonomy.taxonomy import Taxonomy
from models.tagrec import TagRec
from llm.llmJudge import LLMJudge
from ppi.ppi import LPPI
import pandas as pd



def main(epochs: int):
    
    questions = load_target_text("data/d1/train.csv", target_column = "eng", top_n=100)
    taxonomy = Taxonomy("taxonomy/PCv3.txt")

    tagger = TagRec()
    tagger.set_taxonomy(taxonomy)
    tagger.load()

    llm_judge = LLMJudge(model="openai/gpt-oss-120b")

    lppi = LPPI()
    lppi_gold = pd.read_pickle("data/d1/lppi_gold1.pkl")
    lppi.fit(lppi_gold, score_cols=["Subject_validity", "Chapter_validity", "Topic_validity"])
    
    for epoch in range(epochs):
        tagrec_predictions = tagger.predict(questions, top_k=3)
        tagrec_predictions_score_llm = llm_judge.judge_taxonomy_predictions(tagrec_predictions)
        lppi.calibrate(tagrec_predictions_score_llm)

if __name__ == "__main__":
    main(epochs=1)