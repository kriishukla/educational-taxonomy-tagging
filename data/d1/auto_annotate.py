import pandas as pd
from llm.lm import LM
import copy


TAXONOMY_PREDICT_MESSAGE = [
{
    "role": "system",
    "content": '''
You are an expert semantic classifier specializing in taxonomy selection and content categorization.

===========
OBJECTIVE
===========
Your task is to identify the single most appropriate taxonomy for a given question.

===========
INPUT
===========
You will be given:

1. A piece of question belonging to the JEE syllabus.
2. A LIST of taxonomies represented as an array of strings.

Each taxonomy string represents a complete hierarchical classification path:  
Subject >> Chapter >> Topic

===========
CLASSIFICATION TASK
===========
- Analyze the given question.
- Compare the conceptual relevance, domain consistency, and semantic correctness of the question with each taxonomy in the list.
- Select the ONE taxonomy that best matches the question.

===========
OUTPUT
===========
Return ONLY the single most appropriate taxonomy.

- The output must EXACTLY MATCH one of the taxonomy strings from the input list.

- Do NOT include explanations or extra text.
- Output ONLY the selected taxonomy string.
- Do NOT invent or modify taxonomy values.
- Use semantic judgment, not keyword overlap alone.

===========
IMPORTANT RULES
===========
- If none of the taxonomies match the question or the question is invalid/unclear, output ONLY "NO".
'''
},

{
    "role": "user",
    "content": '''
Question:
The most suitable material to be used as the core of an electromagnet is soft iron.

Taxonomy:
[
"Physics >> Magnetism and Matter >> The bar magnet",
"Physics >> Magnetism and Matter >> Magnetic properties of materials",
"Physics >> Magnetism and Matter >> Permanent magnets and electromagnets",
]
'''
},
{
    "role": "assistant",
    "content": "Physics >> Magnetism and Matter >> Permanent magnets and electromagnets"
},

{
    "role": "user",
    "content": '''
Question:
The element present in traces in insulin is: A. Iron B. cobalt c. zinc D. Magnesium

Taxonomy:
[
"Chemistry >> Amines >> Nomenclature",
"Chemistry >> Amines >> Preparation of amines",
]
'''
},
{
    "role": "assistant",
    "content": "NO"
}
]


def predict_taxonomy(annotator, question, taxonomies):
    # make copy of base message
    messages = copy.deepcopy(TAXONOMY_PREDICT_MESSAGE)

    messages.append({
        "role": "user",
        "content": f'''Question:
{question}

Taxonomy:
{taxonomies}
'''
    })

    taxonomy = annotator(messages)
    return taxonomy.strip()

def annotate_csv_with_taxonomy(csv_path, taxonomy_txt_path, top_n=None, output_path=None):
    annotator = LM(model="openai/gpt-oss-120b", temperature=0)
    # Read csv
    df = pd.read_csv(csv_path)

    # Read taxonomy file
    with open(taxonomy_txt_path, 'r', encoding='utf-8') as f:
        taxonomy_lines = f.readlines()

    # Ignore first line
    taxonomy_lines = [line.strip() for line in taxonomy_lines[1:] if line.strip()]

    # Split into columns for filtering
    taxonomy_df = pd.DataFrame([
        {
            "taxonomy": line,
            "Subject": line.split(">>")[0].strip(),
            "Chapter": line.split(">>")[1].strip(),
            "Topic": line.split(">>")[2].strip()
        }
        for line in taxonomy_lines
    ])

    # Limit to top_n rows if provided
    if top_n is not None:
        df = df.head(top_n)

    annotated_rows = []

    for i, row in df.iterrows():
        q_id = row['q_id']
        question = row['eng']
        chapter = row['chapter']

        # filter taxonomy options by chapter
        options = taxonomy_df[taxonomy_df['Chapter'].str.lower() == chapter.lower()]['taxonomy'].tolist()

        if not options:
            print("0 taxonomies found for chapter:", chapter)
            continue

            # try expect continue 
        try:   
            taxonomy_choice = predict_taxonomy(annotator, question, options)
        except Exception as e:
            print(f"error predicting taxonomy for q_id {q_id}: {e}")
            continue

        if taxonomy_choice == "NO":
            continue

        annotated_rows.append({
            "q_id": q_id,
            "eng": question,
            "class": row.get("class"),
            "chapter": chapter,
            "taxonomy": taxonomy_choice
        })
        
        print(f"[{i}] {taxonomy_choice}")

    annotated_df = pd.DataFrame(annotated_rows)

    # Save output csv if required
    if output_path:
        annotated_df.to_csv(output_path, index=False)

    return annotated_df


annotate_csv_with_taxonomy(
    csv_path="lppi.csv",
    taxonomy_txt_path="../../taxonomy/PCv3.txt",
    top_n=660,  
    output_path="lppi_gold.csv")