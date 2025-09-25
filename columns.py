import os
import pandas as pd
from data_prep import process_csv_files_enriched

# The paths to the input CSV files (relative paths) - these are the raw data files
input_files = [
    r"raw data\nmbfinalDiabetes (4).csv",
    r"raw data\nmbfinalnewDiabetes (3).csv",
    r"raw data\PrePostFinal (3).csv",
]

# Output directory (absolute path)
output_dir = r"C:\Users\maadh\OneDrive\Desktop\PhobosQ\poornima-ML\cleaned data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# The list of columns to be saved to the new file
# I have cleaned the list you provided by removing asterisks, duplicates, and extra text.
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
    'PreRmildactivityduration', 'PreRmoderateduration', 'PreRvigorousduration',
    'PhysicalActivity_Optimal', 'PreRskipbreakfast', 'PreRlessfiber', 'PreRlessfruit',
    'PreRlessvegetable', 'PreRmilk', 'PreRmeat', 'PreRfriedfood', 'PreRpopcorn',
    'PreRsweet', 'PreRdrink', 'PreRstaplefood', 'PreRheight', 'PreRweight', 'PreRhip',
    'PreRwaist', 'PreBLPPBS', 'PreBLFBS', 'PreBLHBA1C', 'PreBLCHOLESTEROL',
    'PreBLTRIGLYCERIDES', 'PreBLCHOLESTEROLLDL','PreRsystolicfirst',
    'PreRsystolicsecond',
    'PostRsystolicfirst',
    'PostRsystolicsecond',
    'PreRdiastolicfirst',
    'PreRdiastolicsecond',
    'PostRdiastolicfirst',
    'PostRdiastolicsecond',
    'postblage',
    'PreRdiaage',
    'systolic',
    'diastolic',
    'Diabetic_Duration(years)',
    'Duration_Status',
    'current_smoking',
    'current_alcohol'
]

# Process files with enrichment and encoding
process_csv_files_enriched(
    input_files,
    output_dir=output_dir,
    columns_to_keep=columns_to_keep,
    suffix="_selected_columns_cleaned_processed",
    overwrite=True,
    enrich_with_features=True,
    top_feature_percent=0.5
)