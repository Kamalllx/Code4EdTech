import pandas as pd

# Read the Excel file
df = pd.read_excel(r"C:\Users\Dell\Desktop\Projects\Randomhack\draft2\Theme 1 - Sample Data\Key (Set A and B).xlsx")

print("DataFrame info:")
print(df.info())
print("\nDataFrame shape:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nFirst few rows:")
print(df.head())

print("\nSample data from each column:")
for col in df.columns:
    print(f"\n{col}:")
    sample_values = df[col].head(5).tolist()
    for i, val in enumerate(sample_values):
        print(f"  Row {i}: '{val}' (type: {type(val).__name__})")