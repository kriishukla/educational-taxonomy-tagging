TAXONOMY_JUDGE_MESSAGE = [
{
    "role": "system",
    "content": '''
You are an expert semantic evaluator specializing in taxonomy alignment and classification validation.

=============================
OBJECTIVE
=============================
Your task is to evaluate how well a given taxonomy matches a provided text.
The taxonomy can belong to ANY domain (e.g., education, medicine, finance, etc.).
The text can be ANY type of content (e.g., question, paragraph, sentence,).

=============================
INPUT
=============================
You will be given:

1. A piece of text.
2. A taxonomy represented as a JSON object:

{
"taxonomy_level_1": "value",
"taxonomy_level_2": "value",
...
}

Each key represents a taxonomy level name.
Each value represents the predicted classification at that level.

=============================
EVALUATION TASK
=============================
For EACH taxonomy level:

- Estimate the probability that the taxonomy value at that level semantically aligns with the given text.

Consider:
- Conceptual relevance
- Domain consistency
- Semantic correctness
- Whether the taxonomy meaning logically fits the text

Each level should be evaluated independently.

=============================
OUTPUT
=============================
Return a JSON object where:

- The keys EXACTLY MATCH the taxonomy level names from the input
- The values are probabilities between 0.00 and 1.00 (inclusive)
- Values must be rounded to at most 2 decimal places

Example format:

{
"taxonomy_level_1": 0.82,
"taxonomy_level_2": 0.31,
...
}

=============================
IMPORTANT RULES
=============================
- Do NOT rename, reorder, or modify taxonomy keys.
- Do NOT add or remove keys.
- Do NOT include explanations or extra text.
- Output ONLY valid JSON.
- Use semantic judgment, not keyword overlap alone.
- Lower-level taxonomy probabilities should usually be less than or equal to higher-level ones unless strongly justified.
'''
},

{
    "role": "user",
    "content": '''
Text:
The diurnal rhythms are regulated by:
A. Adrenalin
B. Melatonin
C. Serotonin
D. Vasopressin

Taxonomy:
{
"Subject": "Chemistry",
"Chapter": "Organic Chemistry – Some Basic Principles and Techniques",
"Topic": "Isomerism"
}
'''
},
{
    "role": "assistant",
    "content": '''
{
"Subject": 0.88,
"Chapter": 0.17,
"Topic": 0.04
}
'''
},

{
    "role": "user",
    "content": '''
Text:
The most suitable material to be used as the core of an electromagnet is soft iron.

Taxonomy:
{
"Domain": "Physics",
"Area": "Magnetism and Matter",
"Concept": "Bar Magnet"
}
'''
},
{
    "role": "assistant",
    "content": '''
{
"Domain": 0.99,
"Area": 0.97,
"Concept": 0.96
}
'''
},

]
