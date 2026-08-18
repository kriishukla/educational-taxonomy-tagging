import pandas as pd

# Load the CSV file
file_path = "jeeq.csv"  # Adjust if different
df = pd.read_csv(file_path)

print(df.columns.tolist())

# Filter for Physics and Chemistry only
df_physics_chemistry = df[df['subject'].isin(['Physics', 'Chemistry'])].drop(columns=['subject']).rename(columns={'question': 'questions'})

# Filter for PCM (Physics, Chemistry, Maths)
df_pcm = df[df['subject'].isin(['Physics', 'Chemistry', 'Maths'])].drop(columns=['subject']).rename(columns={'question': 'questions'})

# Save new CSV files
physics_chemistry_file = "jeeq_PC.csv"
pcm_file = "jeeq_PCM.csv"

df_physics_chemistry.to_csv(physics_chemistry_file, index=False)
df_pcm.to_csv(pcm_file, index=False)