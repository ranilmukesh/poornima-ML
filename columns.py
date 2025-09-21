import os
import pandas as pd

# The paths to the input CSV files (relative paths)
input_files = [
    r"prepared data\nmbfinalDiabetes (4)_selected_columns_cleaned.csv",
    r"prepared data\nmbfinalnewDiabetes (3)_selected_columns_cleaned.csv",
    r"prepared data\PrePostFinal (3)_selected_columns_cleaned.csv",
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
]

for input_csv_path in input_files:
    # Get just the filename without extension for output naming
    base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]
    output_csv_path = os.path.join(output_dir, f"{base_filename}_processed.csv")
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(input_csv_path)}")
    print(f"{'='*80}")
    
    try:
        # Read the original CSV file
        df = pd.read_csv(input_csv_path)

        # Select only the specified columns
        # We will check which of the desired columns are actually in the CSV
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        
        if not existing_columns:
            print("None of the specified columns were found in the CSV file.")
        else:
            # Create a new DataFrame with only the existing columns
            new_df = df[existing_columns]

            # Save the new DataFrame to a new CSV file
            new_df.to_csv(output_csv_path, index=False)

            print(f"✓ Successfully created: {os.path.basename(output_csv_path)}")
            print(f"📊 Summary: {len(existing_columns)} included / {len(missing_columns)} missing / {len(columns_to_keep)} total")
            print(f"📁 Saved to: {output_csv_path}")
            
            # Show which columns were missing for this specific file
            if missing_columns:
                print(f"\n❌ MISSING columns in {os.path.basename(input_csv_path)} ({len(missing_columns)}):")
                for i, col in enumerate(missing_columns, 1):
                    print(f"  {i:2d}. {col}")
            else:
                print(f"\n✅ All {len(columns_to_keep)} requested columns found in this file!")

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_csv_path}' was not found.")
        print(f"   Make sure the file exists in the current directory or adjust the path.")
    except Exception as e:
        print(f"❌ An error occurred while processing '{input_csv_path}': {e}")

print(f"\n{'='*80}")
print("🎉 Processing completed for all files.")
print(f"📁 All output files saved to: {output_dir}")
print(f"{'='*80}")