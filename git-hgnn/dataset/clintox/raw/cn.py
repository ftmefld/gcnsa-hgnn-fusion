import pandas as pd

# Read the txt file (already comma-separated)
df = pd.read_csv("filtered_smiles_ClinTox.csv.txt")

# Save as CSV
df.to_csv("clintoxfilter.csv", index=False)
