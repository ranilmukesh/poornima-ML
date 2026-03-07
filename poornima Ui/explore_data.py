import pandas as pd
import os

files = [
    r"d:\poornima sukumar mam files\poornima Ui\apolloCombined.csv",
    r"d:\poornima sukumar mam files\poornima Ui\ApolloFormat_nmbfinalDiabetes (4).csv",
    r"d:\poornima sukumar mam files\poornima Ui\ApolloFormat_nmbfinalnewDiabetes (3).csv",
    r"d:\poornima sukumar mam files\poornima Ui\ApolloFormat_PrePostFinal (3).csv",
]

DESIRED_COLS = [
    'PostBLAge', 'PreBLGender', 'PreRarea', 'PreRmaritalstatus',
    'PreReducation', 'PreRpresentoccupation', 'PreRdiafather', 'PreRdiamother',
    'PreRdiabrother', 'PreRdiasister', 'current_smoking', 'current_alcohol',
    'PreRsleepquality', 'PreRmildactivity', 'PreRmildactivityduration',
    'PreRmoderate', 'PreRmoderateduration', 'PreRvigorous', 'PreRvigorousduration',
    'PreRskipbreakfast', 'PreRlessfruit', 'PreRlessvegetable', 'PreRmilk',
    'PreRmeat', 'PreRfriedfood', 'PreRsweet', 'PreRwaist', 'PreRBMI',
    'PreRsystolicfirst', 'PreRdiastolicfirst', 'PreBLPPBS', 'PreBLFBS',
    'PreBLHBA1C', 'PreBLCHOLESTEROL', 'PreBLTRIGLYCERIDES',
    'Diabetic_Duration', 'PostRgroupname', 'PostBLHBA1C'
]

TARGET = 'PostBLHBA1C'

for fpath in files:
    fname = os.path.basename(fpath)
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(fpath, low_memory=False)
        print(f"Shape: {df.shape}")
        print(f"All columns: {list(df.columns)}")
        
        # Check which desired columns exist
        found = [c for c in DESIRED_COLS if c in df.columns]
        missing = [c for c in DESIRED_COLS if c not in df.columns]
        print(f"\nDesired cols FOUND ({len(found)}): {found}")
        print(f"Desired cols MISSING ({len(missing)}): {missing}")
        
        # Check target
        if TARGET in df.columns:
            print(f"\nTarget '{TARGET}' stats:")
            print(f"  dtype: {df[TARGET].dtype}")
            print(f"  non-null: {df[TARGET].notna().sum()}")
            print(f"  null: {df[TARGET].isna().sum()}")
            print(f"  describe:\n{df[TARGET].describe()}")
        
        # Show dtypes and unique values for found columns
        print(f"\nColumn details for found columns:")
        for col in found:
            nunique = df[col].nunique()
            dtype = df[col].dtype
            sample_vals = df[col].dropna().unique()[:5]
            print(f"  {col}: dtype={dtype}, nunique={nunique}, nulls={df[col].isna().sum()}, samples={sample_vals}")
    
    except Exception as e:
        print(f"ERROR: {e}")

print("\n\nDONE!")
