"""
Poornima-ML: Diabetes Dataset Preprocessing
===========================================

Main execution script to process raw diabetes datasets into ML-ready format.

Usage:
    python run_pipeline.py

This will process all three datasets:
- nmbfinalDiabetes (4).csv
- nmbfinalnewDiabetes (3).csv  
- PrePostFinal (3).csv

Output:
    Processed files will be saved in 'cleaned data/' folder with:
    - Feature selection and enrichment
    - Categorical encoding
    - Missing value imputation
    - 80 carefully selected columns ready for ML
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from data_prep import process_csv_files_enriched
except ImportError as e:
    print(f"Error importing data_prep module: {e}")
    print("Make sure data_prep.py is in the same directory and contains the required functions.")
    sys.exit(1)

def main():
    """Main function to run the complete data processing pipeline."""
    
    print("🚀 Starting Poornima-ML Data Processing Pipeline")
    print("=" * 60)
    
    # Define input files (raw data)
    input_files = [
        r"raw data\nmbfinalDiabetes (4).csv",
        r"raw data\nmbfinalnewDiabetes (3).csv",
        r"raw data\PrePostFinal (3).csv",
    ]
    
    # Define output directory
    output_dir = r"cleaned data"
    
    # Define columns to keep (80 selected columns)
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
    
    # Check if input files exist
    missing_files = []
    for file_path in input_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Error: The following input files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure the raw data files are in the 'raw data/' folder.")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Input files: {len(input_files)} CSV files")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Target columns: {len(columns_to_keep)} selected columns")
    print(f"🔧 Features: Enrichment with top 50% important features enabled")
    print()
    
    try:
        # Run the processing pipeline
        result_paths = process_csv_files_enriched(
            input_files,
            output_dir=output_dir,
            columns_to_keep=columns_to_keep,
            suffix="_selected_columns_cleaned_processed",
            overwrite=True,
            enrich_with_features=True,
            top_feature_percent=0.5
        )
        
        print()
        print("✅ Processing completed successfully!")
        print(f"📊 Generated {len(result_paths)} processed files:")
        for path in result_paths:
            print(f"   - {os.path.basename(path)}")
        
        print()
        print("🎉 Your data is now ready for machine learning!")
        print("📈 The processed files contain:")
        print("   • 80 carefully selected columns")
        print("   • Encoded categorical variables (Male=1, Female=0, Urban=1, Rural=0)")
        print("   • Missing values filled using feature relationships")
        print("   • Derived health indicators (current_smoking, current_alcohol)")
        print("   • Calculated metrics (BMI, diabetes duration, etc.)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("Please check the error message above and ensure all dependencies are installed.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)