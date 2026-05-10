import pandas as pd
from scipy.stats import pearsonr
import os

# This finds the folder where PC.py is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "AI_vs_Human_Scores.csv")

# Now it will find the file if it's in the same folder as PC.py
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
    
    # Run your correlation
    r_value, p_value = pearsonr(df['AI_Score'], df['Recruiter_Score'])
    print(f"Correlation (r): {r_value:.4f}")
    
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    print("Make sure you moved AI_vs_Human_Scores.csv into the 'Data' folder.")