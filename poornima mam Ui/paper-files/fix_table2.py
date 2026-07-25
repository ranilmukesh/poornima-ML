import os
import re

analysis_path = r"d:\poornima sukumar mam files\poornima mam Ui\paper-files\generate_analysis.py"
with open(analysis_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace table 2 numeric conversion
old_t2 = """        for col in ['PreBLHBA1C', 'PreBLFBS', 'PreRBMI', 'PreRwaist', 'Diabetic_Duration']:
            if col in df_master.columns:
                mean_o = df_master[col].mean()
                sd_o = df_master[col].std()
                missing_pct = df_master[col].isnull().mean() * 100"""

new_t2 = """        for col in ['PreBLHBA1C', 'PreBLFBS', 'PreRBMI', 'PreRwaist', 'Diabetic_Duration']:
            if col in df_master.columns:
                s_o = pd.to_numeric(df_master[col], errors='coerce')
                mean_o = s_o.mean()
                sd_o = s_o.std()
                missing_pct = df_master[col].isnull().mean() * 100"""

content = content.replace(old_t2, new_t2)

with open(analysis_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed Table 2")
