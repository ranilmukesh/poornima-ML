import os
import re

analysis_path = r"d:\poornima sukumar mam files\poornima mam Ui\paper-files\generate_analysis.py"
with open(analysis_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace add_cont
old_add_cont = """        def add_cont(name, col):
            if col in df_master.columns:
                mean_all = df_master[col].mean()
                sd_all = df_master[col].std()
                if 'PostRgroupname' in df_master.columns:
                    mean_y = df_master[col][yoga_mask].mean()
                    sd_y = df_master[col][yoga_mask].std()
                    mean_c = df_master[col][ctrl_mask].mean()
                    sd_c = df_master[col][ctrl_mask].std()"""

new_add_cont = """        def add_cont(name, col):
            if col in df_master.columns:
                s_all = pd.to_numeric(df_master[col], errors='coerce')
                mean_all = s_all.mean()
                sd_all = s_all.std()
                if 'PostRgroupname' in df_master.columns:
                    s_y = pd.to_numeric(df_master[col][yoga_mask], errors='coerce')
                    s_c = pd.to_numeric(df_master[col][ctrl_mask], errors='coerce')
                    mean_y = s_y.mean()
                    sd_y = s_y.std()
                    mean_c = s_c.mean()
                    sd_c = s_c.std()"""

content = content.replace(old_add_cont, new_add_cont)

with open(analysis_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed add_cont")
