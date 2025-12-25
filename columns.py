"""Column Selection and Data Processing Pipeline

This module processes raw diabetes datasets by:
1. Selecting important columns from the raw data
2. Applying data cleaning and preprocessing
3. Feature engineering and categorical encoding
"""

import os
import pandas as pd
from data_prep import process_csv_files_enriched

input_files = [
    "raw_data/nmbfinalDiabetes (4).csv",
    "raw_data/nmbfinalnewDiabetes (3).csv", 
    "raw_data/PrePostFinal (3).csv",
]

output_dir = "temp_processed"

os.makedirs(output_dir, exist_ok=True)

columns_to_keep = [
    'PostBLHBA1C','PostRgroupname', 'PreBLAge', 'PreRgender', 'PreRarea', 'PreRmaritalstatus',
    'PreReducation', 'PreRpresentoccupation', 'PreRcurrentworking', 'PreRdiafather',
    'PreRdiamother', 'PreRdiabrother', 'PreRdiasister', 'PreRtobuse', 'PreRtobdaily',
    'PreRtobcurrent', 'PreRtobdailyuse', 'PreRtobaverage', 'PreRtobsmoking',
    'PreRstouse', 'PreRstodaily', 'PreRstocurrent', 'PreRstodailyuse', 'PreRstoyear',
    'PreRalcoholuse', 'PreRalyear', 'PreRalthreemonths', 'PreRalthreemonthsuse',
    'PreRallastmonth', 'PreRallastmonthuse', 'PreRlastmonthavg', 'PreRdrinktype',
    'PreRalmlperday', 'PreRsleepquality', 'PreRstworkvalue', 'PreRstfamilyvalue',
    'PreRsthealthvalue', 'PreRstfinancialvalue', 'PreRstsocialvalue',
    'PreRmildactivityduration', 'PreRmoderateduration', 'PreRvigorousduration', 'PreRskipbreakfast', 'PreRlessfiber', 'PreRlessfruit',
    'PreRlessvegetable', 'PreRmilk', 'PreRmeat', 'PreRfriedfood', 'PreRpopcorn',
    'PreRsweet', 'PreRdrink', 'PreRstaplefood', 'PreRheight', 'PreRweight', 'PreRhip',
    'PreRwaist', 'PreBLPPBS', 'PreBLFBS', 'PreBLHBA1C', 'PreBLCHOLESTEROL',
    'PreBLTRIGLYCERIDES','PreRsystolicfirst',
    'PreRdiastolicfirst',
    'postblage',
    'PreRdiaage',
    'Diabetic_Duration',
    'Duration_Status', #Preblcholestrolldl doesnt exist in apollo , so dont use
    'current_smoking', #remove the proccessed
    'current_alcohol' #remove the proccessed(cols used for feature engineering only)
]

if __name__ == "__main__":
    process_csv_files_enriched(
        input_files,
        output_dir=output_dir,
        columns_to_keep=columns_to_keep,
        suffix="_selected_columns_cleaned_processed",
        overwrite=True,
        enrich_with_features=True,
        top_feature_percent=0.5
    )