from typing import Optional
import os
import pandas as pd
import re
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np


GENDER_MAP = {
    "male": "Male",
    "m": "Male",
    "ma le": "Male",
    "female": "Female",
    "f": "Female",
    "fe male": "Female",
    "transgender": "Transgender",
    "trans": "Transgender",
}

AREA_MAP = {
    "urban": "Urban",
    "rural": "Rural",
}

EDUCATION_ORDINAL = {
    "no formal schooling": 0,
    "up to primary school": 1,
    "up to high school": 2,
    "up to intermediate": 3,
    "up to university": 4,
    "university completed or higher": 5,
    "specify if any": 5,
}

OCCUPATION_GROUP = {
    "Homemaker": "Homemaker",
    "Professional/Executive/Big business": "Professional",
    "Clerical/medium business": "Clerical",
    "Self-employed/ skilled": "Self-employed",
    "Unskilled/ landless laborer": "Unskilled",
    "Retired": "Retired",
    "Unemployed(able to work)": "Unemployed",
    "Unemployed(unable to work)": "Unemployed",
    "Others(specify if any)": "Other",
}

SLEEP_QUALITY_ORDINAL = {
    "very good": 3,
    "fairly good": 2,
    "fairly bad": 1,
    "very bad": 0,
}

DIETARY_HABITS_ORDINAL = {
    # For bad habits (lower is better): skip breakfast, less fiber, less fruit, less vegetables, meat, fried food, popcorn, sweet, drink
    "rarely/ never": 2,  # Good (rarely do bad things)
    "sometimes": 1,      # Bad (sometimes do bad things) 
    "usually/often": 0,  # Very bad (usually do bad things)
}

MILK_CONSUMPTION_ORDINAL = {
    # For good habits (higher is better): milk consumption
    "usually/often": 2,  # Good (usually consume milk)
    "sometimes": 1,      # Moderate (sometimes consume milk)
    "rarely/ never": 0,  # Bad (rarely consume milk)
}


def _clean_text_series(s: pd.Series) -> pd.Series:
    """Coerce to string, strip surrounding whitespace and lower-case.

    Keeps NaN as NaN.
    """
    s = s.copy()
    # preserve NaN
    mask_na = s.isna()
    s = s.astype(str).str.strip()
    s.loc[mask_na] = pd.NA
    return s


def normalize_gender(s: pd.Series) -> pd.Series:
    """Normalize `PreRgender` values.

    - trims and lowercase inputs
    - maps common variants to 'Male','Female','Transgender'
    - unknown or missing -> 'Missing'

    Returns a pandas Series of cleaned string categories.
    """
    s2 = _clean_text_series(s).str.lower()

    def _map(x: Optional[str]):
        if pd.isna(x) or x in ("nan", "none", "na", ""):
            return "Missing"
        x = x.replace(" ", "") if isinstance(x, str) else x
        return GENDER_MAP.get(x, "Other")

    return s2.map(_map).astype("category")


def map_area(s: pd.Series, as_numeric: bool = False) -> pd.Series:
    """Normalize `PreRarea`.

    - maps to canonical 'Urban' or 'Rural'
    - missing -> 'Missing' (or -1 if as_numeric)
    - when as_numeric=True returns int: Urban=1, Rural=0, Missing=-1
    """
    s2 = _clean_text_series(s).str.lower()
    mapped = s2.map(lambda x: AREA_MAP.get(x, None) if pd.notna(x) else None)
    if as_numeric:
        return mapped.map({"Urban": 1, "Rural": 0}).fillna(-1).astype(int)
    return mapped.fillna("Missing").astype("category")


def clean_marital_status(s: pd.Series, group_rare: bool = True, rare_threshold: float = 0.01) -> pd.Series:
    """Clean `PreRmaritalstatus` text and optionally group rare categories.

    - fixes common typos (e.g., 'Seperated' -> 'Separated')
    - groups categories whose relative frequency < rare_threshold into 'Other'
    - missing -> 'Missing'
    """
    s2 = _clean_text_series(s)
    # basic corrections
    corrections = {
        "divorcee/seperated": "Divorcee/Separated",
        "divorcee/separated": "Divorcee/Separated",
        "widow/widower": "Widowed",
    }
    s3 = s2.str.lower().str.replace(r"\s+", " ", regex=True)
    s3 = s3.map(lambda x: corrections.get(x, x) if pd.notna(x) else x)
    s3 = s3.fillna("Missing")

    if group_rare:
        freqs = s3.value_counts(normalize=True)
        rare = set(freqs[freqs < rare_threshold].index)
        s3 = s3.map(lambda x: "Other" if x in rare else x)

    return s3.astype("category")


def education_to_ordinal(s: pd.Series, unknown_value: int = -1) -> pd.Series:
    """Map `PreReducation` to an ordinal integer scale.

    Uses the mapping in EDUCATION_ORDINAL. Unknown or missing values are set to
    `unknown_value` (default -1).
    """
    s2 = _clean_text_series(s).str.lower()
    mapped = s2.map(lambda x: EDUCATION_ORDINAL.get(x, None) if pd.notna(x) else None)
    return mapped.fillna(unknown_value).astype(int)


def group_occupation(s: pd.Series) -> pd.Series:
    """Group `PreRpresentoccupation` into broader buckets.

    Returns grouped categorical strings. Unmapped or missing values -> 'Other'/'Missing'.
    """
    s2 = _clean_text_series(s)
    # use a case-insensitive map
    inv_map = {k.lower(): v for k, v in OCCUPATION_GROUP.items()}
    def _map(x: Optional[str]):
        if pd.isna(x):
            return "Missing"
        return inv_map.get(x.lower(), "Other")

    return s2.map(_map).astype("category")


def sleepquality_to_ordinal(s: pd.Series, unknown_value: int = -1) -> pd.Series:
    """Map `PreRsleepquality` to an ordinal integer scale.

    Uses the mapping in SLEEP_QUALITY_ORDINAL. Unknown or missing values are set to
    `unknown_value` (default -1).
    """
    s2 = _clean_text_series(s).str.lower()
    mapped = s2.map(lambda x: SLEEP_QUALITY_ORDINAL.get(x, None) if pd.notna(x) else None)
    return mapped.fillna(unknown_value).astype(int)


def map_activity_duration_to_minutes(s: pd.Series, unknown_value: int = -1) -> pd.Series:
    """Convert activity duration string ranges to average minutes.

    - 'at least 10 mins' -> 10
    - '10 - 30mins' -> 20
    - '30mins - 1hr' -> 45
    - '1hr - 1.5hrs' -> 75
    - '>1.5hrs' -> 105
    - Missing/unknown -> `unknown_value` (default -1)
    """
    s2 = _clean_text_series(s).str.lower().str.replace(" ", "")

    # Mapping from cleaned string to minutes
    duration_map = {
        "atleast10mins": 10,
        "10-30mins": 20,
        "30mins-1hr": 45,
        "1hr-1.5hrs": 75,
        ">1.5hrs": 105,
    }

    def _map(x: Optional[str]):
        if pd.isna(x):
            return None
        return duration_map.get(x, None)

    mapped = s2.map(_map)
    return mapped.fillna(unknown_value).astype(int)


def dietary_habits_to_ordinal(s: pd.Series, is_milk: bool = False, unknown_value: int = -1) -> pd.Series:
    """Map dietary habit columns to ordinal integer scale.

    For bad habits (skip breakfast, less fiber, less fruit, less vegetables, meat, fried food, popcorn, sweet, drink):
    - 'Rarely/ Never' -> 2 (Good - rarely do bad things)
    - 'Sometimes' -> 1 (Bad - sometimes do bad things)
    - 'Usually/Often' -> 0 (Very bad - usually do bad things)

    For good habits (milk consumption):
    - 'Usually/Often' -> 2 (Good - usually consume milk)
    - 'Sometimes' -> 1 (Moderate - sometimes consume milk)  
    - 'Rarely/ Never' -> 0 (Bad - rarely consume milk)

    Args:
        s: pandas Series with dietary habit values
        is_milk: if True, uses milk consumption mapping (higher is better)
        unknown_value: value for missing/unknown entries (default -1)
    """
    s2 = _clean_text_series(s).str.lower()
    
    if is_milk:
        mapped = s2.map(lambda x: MILK_CONSUMPTION_ORDINAL.get(x, None) if pd.notna(x) else None)
    else:
        mapped = s2.map(lambda x: DIETARY_HABITS_ORDINAL.get(x, None) if pd.notna(x) else None)
    
    return mapped.fillna(unknown_value).astype(int)


def calculate_bmi(height_cm: pd.Series, weight_kg: pd.Series) -> pd.Series:
    """Calculate BMI from height (cm) and weight (kg).
    
    BMI = weight (kg) / (height (m))^2
    Returns NaN for missing values or invalid measurements.
    """
    # Convert height from cm to meters
    height_m = height_cm / 100
    
    # Calculate BMI, handle division by zero and negative values
    bmi = weight_kg / (height_m ** 2)
    
    # Set invalid BMI values to NaN (e.g., negative weights/heights, extreme values)
    bmi = bmi.where((height_cm > 0) & (weight_kg > 0) & (bmi < 100) & (bmi > 10))
    
    return bmi


def calculate_waist_hip_ratio(waist_cm: pd.Series, hip_cm: pd.Series) -> pd.Series:
    """Calculate waist-to-hip ratio (WHR).
    
    WHR = waist circumference / hip circumference
    Returns NaN for missing values or invalid measurements.
    """
    whr = waist_cm / hip_cm
    
    # Set invalid WHR values to NaN (e.g., negative measurements, extreme ratios)
    whr = whr.where((waist_cm > 0) & (hip_cm > 0) & (whr < 3) & (whr > 0.3))
    
    return whr


def yesno_to_binary(s: pd.Series) -> pd.Series:
    """Convert common Yes/No (and variants) to nullable Int64 1/0.

    Recognizes variants like: 'Yes', 'No', '1', '0', 'Yes(1)', 'No(0)', 'Y', 'N'.
    Unknown/missing values are returned as <NA> using pandas nullable Int64 dtype.
    """
    s2 = _clean_text_series(s)

    def _map(x: Optional[str]):
        if pd.isna(x) or str(x).lower() in ("nan", "none", "na", ""):
            return pd.NA
        # remove surrounding whitespace and common punctuation from lowercased string
        x_clean = re.sub(r"[^a-z0-9]", "", str(x).lower())
        # explicit digit
        if x_clean == "1":
            return 1
        if x_clean == "0":
            return 0
        if x_clean.startswith("yes") or x_clean == "y" or x_clean.endswith("1"):
            return 1
        if x_clean.startswith("no") or x_clean == "n" or x_clean.endswith("0"):
            return 0
        return pd.NA

    mapped = s2.map(_map)
    # return pandas nullable integer dtype so NA is preserved
    return mapped.astype("Int64")


def create_current_smoking(df: pd.DataFrame) -> pd.Series:
    """Create derived 'current_smoking' variable from tobacco/smoking columns.
    
    Returns 1 if any of the smoking-related columns has 'Yes' or any non-'No' value,
    0 if all are 'No' or missing, <NA> if all columns are missing.
    
    Smoking-related columns checked:
    - PreRtobuse, PreRtobdaily, PreRtobcurrent, PreRtobdailyuse, PreRtobaverage, 
      PreRtobsmoking, PreRstouse, PreRstodaily, PreRstocurrent, PreRstodailyuse, PreRstoyear
    """
    smoking_cols = [
        "PreRtobuse", "PreRtobdaily", "PreRtobcurrent", "PreRtobdailyuse", "PreRtobaverage", 
        "PreRtobsmoking", "PreRstouse", "PreRstodaily", "PreRstocurrent", "PreRstodailyuse", "PreRstoyear"
    ]
    
    # Get only columns that exist in the DataFrame
    existing_cols = [col for col in smoking_cols if col in df.columns]
    
    if not existing_cols:
        # No smoking columns present, return all NA
        return pd.Series([pd.NA] * len(df), dtype="Int64")
    
    # For each row, check if any smoking column indicates "Yes" or has any non-"No" value
    result = pd.Series([pd.NA] * len(df), dtype="Int64")
    
    # Combine all smoking columns into a single frame for vectorized operations
    smoke_df = df[existing_cols].copy()
    
    # Replace non-binary columns with binary representation
    if "PreRtobaverage" in smoke_df.columns:
        smoke_df["PreRtobaverage"] = smoke_df["PreRtobaverage"].apply(lambda x: 1 if pd.notna(x) else 0)
    if "PreRtobsmoking" in smoke_df.columns:
        smoke_df["PreRtobsmoking"] = smoke_df["PreRtobsmoking"].apply(lambda x: 1 if pd.notna(x) else 0)
    if "PreRstoyear" in smoke_df.columns:
        smoke_df["PreRstoyear"] = smoke_df["PreRstoyear"].apply(lambda x: 1 if pd.notna(x) else 0)

    # Convert all other columns to binary 1/0
    for col in smoke_df.columns:
        smoke_df[col] = yesno_to_binary(smoke_df[col])

    # Any '1' in a row means current smoker
    is_smoker = (smoke_df == 1).any(axis=1)
    # Any non-NA value means we have info
    has_info = smoke_df.notna().any(axis=1)

    # Set result: 1 if smoker, 0 if not smoker but has info, else NA
    result[is_smoker] = 1
    result[~is_smoker & has_info] = 0
    
    return result


def create_current_alcohol(df: pd.DataFrame) -> pd.Series:
    """Create derived 'current_alcohol' variable from alcohol consumption columns.
    
    Returns 1 if any of the alcohol-related columns has 'Yes' or any non-'No' value,
    0 if all are 'No' or missing, <NA> if all columns are missing.
    
    Alcohol-related columns checked:
    - PreRalcoholuse, PreRalyear, PreRalthreemonths, PreRalthreemonthsuse, 
      PreRallastmonth, PreRallastmonthuse, PreRlastmonthavg, PreRdrinktype, PreRalmlperday
    """
    alcohol_cols = [
        "PreRalcoholuse", "PreRalyear", "PreRalthreemonths", "PreRalthreemonthsuse", 
        "PreRallastmonth", "PreRallastmonthuse", "PreRlastmonthavg", "PreRdrinktype", "PreRalmlperday"
    ]
    
    # Get only columns that exist in the DataFrame
    existing_cols = [col for col in alcohol_cols if col in df.columns]
    
    if not existing_cols:
        # No alcohol columns present, return all NA
        return pd.Series([pd.NA] * len(df), dtype="Int64")

    # For each row, check if any alcohol column indicates "Yes" or has any non-"No" value
    result = pd.Series([pd.NA] * len(df), dtype="Int64")

    # Combine all alcohol columns into a single frame for vectorized operations
    alcohol_df = df[existing_cols].copy()

    # For non-binary columns, any non-null value is treated as a "yes" (1)
    for col in ["PreRalyear", "PreRalthreemonthsuse", "PreRallastmonthuse", "PreRlastmonthavg", "PreRdrinktype", "PreRalmlperday"]:
        if col in alcohol_df.columns:
            alcohol_df[col] = alcohol_df[col].apply(lambda x: 1 if pd.notna(x) else 0)

    # Convert all other columns to binary 1/0
    for col in alcohol_df.columns:
        alcohol_df[col] = yesno_to_binary(alcohol_df[col])

    # Any '1' in a row means current alcohol user
    is_drinker = (alcohol_df == 1).any(axis=1)
    # Any non-NA value means we have info
    has_info = alcohol_df.notna().any(axis=1)

    # Set result: 1 if drinker, 0 if not drinker but has info, else NA
    result[is_drinker] = 1
    result[~is_drinker & has_info] = 0
    
    return result


def calculate_met_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the total MET score from activity duration columns.

    This function calculates the weekly MET score for mild, moderate, and
    vigorous activities and sums them into a new column:
    'Physical activity_total_METSCORE_value'. It also creates a
    'PhysicalActivity_Optimal' column.

    MET multipliers:
    - Mild: 1.5
    - Moderate: 4.9
    - Vigorous: 6.8
    """
    df = df.copy()
    
    # Map activity duration strings to average minutes
    for col in ("PreRmildactivityduration", "PreRmoderateduration", "PreRvigorousduration"):
        if col in df.columns:
            df[f"{col}_mins"] = map_activity_duration_to_minutes(df[col])

    # Calculate MET score for each activity type, treating unknown values (-1) as 0
    mild_met = 0
    if "PreRmildactivityduration_mins" in df.columns:
        mild_mins = df["PreRmildactivityduration_mins"].replace(-1, 0)
        mild_met = 1.5 * mild_mins * 7

    moderate_met = 0
    if "PreRmoderateduration_mins" in df.columns:
        moderate_mins = df["PreRmoderateduration_mins"].replace(-1, 0)
        moderate_met = 4.9 * moderate_mins * 7

    vigorous_met = 0
    if "PreRvigorousduration_mins" in df.columns:
        vigorous_mins = df["PreRvigorousduration_mins"].replace(-1, 0)
        vigorous_met = 6.8 * vigorous_mins * 7
        
    df["Physical activity_total_METSCORE_value"] = mild_met + moderate_met + vigorous_met
    
    # Create 'PhysicalActivity_Optimal' column
    df["PhysicalActivity_Optimal"] = "No"
    df.loc[df["Physical activity_total_METSCORE_value"] >= 1100, "PhysicalActivity_Optimal"] = "Yes"
    
    return df


def clean_pre_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to run all cleaners on a DataFrame with diabetic duration logic."""
    df = df.copy()
    
    if "PreRgender" in df.columns:
        df["PreRgender"] = normalize_gender(df["PreRgender"])
    if "PreRarea" in df.columns:
        df["PreRarea"] = map_area(df["PreRarea"], as_numeric=False)
    if "PreRmaritalstatus" in df.columns:
        df["PreRmaritalstatus"] = clean_marital_status(df["PreRmaritalstatus"])
    if "PreReducation" in df.columns:
        df["PreReducation_ord"] = education_to_ordinal(df["PreReducation"]) 
    if "PreRpresentoccupation" in df.columns:
        df["PreRpresentoccupation_grp"] = group_occupation(df["PreRpresentoccupation"])
    if "PreRsleepquality" in df.columns:
        df["PreRsleepquality_ord"] = sleepquality_to_ordinal(df["PreRsleepquality"])
    
    # Map activity durations to minutes
    for col in ("PreRmildactivityduration", "PreRmoderateduration", "PreRvigorousduration"):
        if col in df.columns:
            df[f"{col}_mins"] = map_activity_duration_to_minutes(df[col])

    # Convert dietary habits to ordinal scores
    bad_habit_cols = [
        "PreRskipbreakfast", "PreRlessfiber", "PreRlessfruit", "PreRlessvegetable", 
        "PreRmeat", "PreRfriedfood", "PreRpopcorn", "PreRsweet", "PreRdrink"
    ]
    for col in bad_habit_cols:
        if col in df.columns:
            df[f"{col}_ord"] = dietary_habits_to_ordinal(df[col], is_milk=False)
    
    # Handle milk consumption separately (good habit)
    if "PreRmilk" in df.columns:
        df["PreRmilk_ord"] = dietary_habits_to_ordinal(df["PreRmilk"], is_milk=True)

    # Calculate BMI from height and weight
    if "PreRheight" in df.columns and "PreRweight" in df.columns:
        df["PreRBMI"] = calculate_bmi(df["PreRheight"], df["PreRweight"])

    # Calculate waist-to-hip ratio
    if "PreRwaist" in df.columns and "PreRhip" in df.columns:
        df["PreWaisttoHipRatio"] = calculate_waist_hip_ratio(df["PreRwaist"], df["PreRhip"])

    # Add systolic for Pre (for nmbfinaldiabetes and nmbfinalnewdiabetes)
    if "PreRsystolicfirst" in df.columns and "PreRsystolicsecond" in df.columns:
        df["systolic"] = df[["PreRsystolicfirst", "PreRsystolicsecond"]].max(axis=1)

    # Add systolic for Post (for prepostfinal)
    elif "PostRsystolicfirst" in df.columns and "PostRsystolicsecond" in df.columns:
        df["systolic"] = df[["PostRsystolicfirst", "PostRsystolicsecond"]].max(axis=1)

    # Add diastolic for Pre (for nmbfinaldiabetes and nmbfinalnewdiabetes)   
    if "PreRdiastolicfirst" in df.columns and "PreRdiastolicsecond" in df.columns:
        df["diastolic"] = df[["PreRdiastolicfirst", "PreRdiastolicsecond"]].max(axis=1)

    # Add diastolic for Post (for prepostfinal)   
    elif "PostRdiastolicfirst" in df.columns and "PostRdiastolicsecond" in df.columns:
        df["diastolic"] = df[["PostRdiastolicfirst", "PostRdiastolicsecond"]].max(axis=1)
    
    # ===== DIABETIC DURATION CALCULATION - FIXED =====
    # Initialize default values for all files
    df["Diabetic_Duration(years)"] = 0
    df["Duration_Status"] = "newly diagnosed"
    
    # Check what columns we have and apply appropriate logic
    if "PreBLAge" in df.columns and "PreRdiaage" in df.columns:
        # nmbfinaldiabetes type: PreBLAge - PreRdiaage
        duration = df["PreBLAge"] - df["PreRdiaage"]
        valid_mask = (df["PreBLAge"].notna() & df["PreRdiaage"].notna() & 
                      (duration >= 0) & (duration <= 80))
        df.loc[valid_mask, "Diabetic_Duration(years)"] = duration[valid_mask]
        df["postblage"] = df["PreBLAge"]  # Create for consistency
        
    elif "PostBLAge" in df.columns and "PostRdiaage" in df.columns:
        # prepostfinal type: PostBLAge - PostRdiaage  
        duration = df["PostBLAge"] - df["PostRdiaage"]
        valid_mask = (df["PostBLAge"].notna() & df["PostRdiaage"].notna() & 
                      (duration >= 0) & (duration <= 80))
        df.loc[valid_mask, "Diabetic_Duration(years)"] = duration[valid_mask]
        df["postblage"] = df["PostBLAge"]  # Create for consistency
        
    # If neither condition met, keep default values (duration=0, newly diagnosed)
    
    # Create Duration_Status based on calculated duration
    df["Duration_Status"] = "newly diagnosed"
    df.loc[df["Diabetic_Duration(years)"] > 0.6, "Duration_Status"] = "known diabetes"
    # ===== END DIABETIC DURATION CALCULATION =====

    # Convert family diabetes indicators from Yes/No to 1/0 (nullable int)
    for col in ("PreRdiafather", "PreRdiamother", "PreRdiabrother", "PreRdiasister", "PreRcurrentworking"):
        if col in df.columns:
            df[col] = yesno_to_binary(df[col])
    
    # Convert individual tobacco/smoking columns to binary
    smoking_binary_cols = ["PreRtobuse", "PreRtobdaily", "PreRtobcurrent", "PreRtobdailyuse", 
                          "PreRstouse", "PreRstodaily", "PreRstocurrent", "PreRstodailyuse"]
    for col in smoking_binary_cols:
        if col in df.columns:
            df[col] = yesno_to_binary(df[col])
    
    # Convert individual alcohol consumption columns to binary where applicable
    alcohol_binary_cols = ["PreRalcoholuse", "PreRalthreemonths", "PreRallastmonth"]
    for col in alcohol_binary_cols:
        if col in df.columns:
            df[col] = yesno_to_binary(df[col])
    
    # Create derived current_smoking variable
    df["current_smoking"] = create_current_smoking(df)
    
    # Create derived current_alcohol variable
    df["current_alcohol"] = create_current_alcohol(df)

    # Calculate total MET score for physical activity
    df = calculate_met_score(df)
    
    return df


def select_important_features(df: pd.DataFrame, target_columns: list, top_percent: float = 0.5) -> list:
    """Select top percentage of most important features for predicting target columns.

    Uses mutual information and random forest feature importance to rank features.
    Returns list of top feature column names.
    """
    # Prepare data for feature selection
    df_copy = df.copy()

    # Convert categorical columns to numeric for feature selection
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}

    for col in categorical_cols:
        if col not in target_columns:  # Don't encode target columns yet
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le

    # Fill NaN values temporarily for feature selection
    df_copy = df_copy.fillna(df_copy.median(numeric_only=True))

    # Get all columns except targets
    feature_cols = [col for col in df_copy.columns if col not in target_columns]

    if not feature_cols:
        return []

    # Calculate feature importance using multiple methods
    importance_scores = {}

    for target in target_columns:
        if target not in df_copy.columns:
            continue

        y = df_copy[target]

        # Skip if target has too few unique values or is mostly NaN
        if y.nunique() < 2 or y.isna().sum() > len(y) * 0.8:
            continue

        # Use mutual information for continuous/categorical targets
        if y.dtype in ['float64', 'int64']:
            try:
                mi_scores = mutual_info_regression(df_copy[feature_cols], y)
            except:
                mi_scores = np.zeros(len(feature_cols))
        else:
            try:
                # Convert target to numeric for MI calculation
                y_encoded = LabelEncoder().fit_transform(y.astype(str))
                mi_scores = mutual_info_classif(df_copy[feature_cols], y_encoded)
            except:
                mi_scores = np.zeros(len(feature_cols))

        # Use Random Forest importance
        rf_scores = np.zeros(len(feature_cols))
        try:
            if y.dtype in ['float64', 'int64']:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                y_encoded = LabelEncoder().fit_transform(y.astype(str))
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

            rf.fit(df_copy[feature_cols], y if y.dtype in ['float64', 'int64'] else y_encoded)
            rf_scores = rf.feature_importances_
        except:
            pass

        # Combine scores
        combined_scores = (mi_scores + rf_scores) / 2

        for i, col in enumerate(feature_cols):
            if col not in importance_scores:
                importance_scores[col] = []
            importance_scores[col].append(combined_scores[i])

    # Average importance across all targets
    avg_importance = {}
    for col in feature_cols:
        if col in importance_scores and importance_scores[col]:
            avg_importance[col] = np.mean(importance_scores[col])
        else:
            avg_importance[col] = 0

    # Sort by importance and select top percentage
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    top_n = max(1, int(len(sorted_features) * top_percent))

    return [col for col, score in sorted_features[:top_n]]


def enrich_columns_with_features(df: pd.DataFrame, target_columns: list, important_features: list) -> pd.DataFrame:
    """Enrich target columns by imputing missing values using important features.

    Uses KNN imputation with the important features to predict missing values in target columns.
    """
    df_enriched = df.copy()

    # Prepare features for imputation
    feature_cols = [col for col in important_features if col in df_enriched.columns]

    if not feature_cols:
        return df_enriched

    # Create feature matrix
    X = df_enriched[feature_cols].copy()

    # Handle categorical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Fill missing values in features first
    X = X.fillna(X.median(numeric_only=True))

    # Impute each target column
    for target_col in target_columns:
        if target_col not in df_enriched.columns:
            continue

        y = df_enriched[target_col].copy()

        # Skip if no missing values
        if y.isna().sum() == 0:
            continue

        # Prepare target for imputation
        if y.dtype in ['object', 'category']:
            # For categorical targets, use classification approach
            y_encoded = LabelEncoder().fit_transform(y.astype(str))
            y_filled = y_encoded.copy().astype(float)

            # Mark missing values
            missing_mask = df_enriched[target_col].isna()
            y_filled[missing_mask] = np.nan

            # Use KNN to impute
            if missing_mask.sum() > 0:
                imputer = KNNImputer(n_neighbors=min(5, len(X)-1))
                X_with_target = X.copy()
                X_with_target['target'] = y_filled

                imputed = imputer.fit_transform(X_with_target)
                imputed_target = imputed[:, -1]

                # Round to nearest integer for categorical
                imputed_target = np.round(imputed_target).astype(int)

                # Decode back to original categories
                le_target = LabelEncoder()
                non_missing_y = y.dropna()
                le_target.fit(non_missing_y)

                # Handle out-of-bounds predictions
                unique_labels = le_target.classes_
                imputed_target = np.clip(imputed_target, 0, len(unique_labels)-1)

                df_enriched.loc[missing_mask, target_col] = le_target.inverse_transform(imputed_target[missing_mask])

        else:
            # For numeric targets, use regression approach
            y_filled = y.copy().astype(float)

            if y_filled.isna().sum() > 0:
                imputer = KNNImputer(n_neighbors=min(5, len(X)-1))
                X_with_target = X.copy()
                X_with_target['target'] = y_filled

                imputed = imputer.fit_transform(X_with_target)
                df_enriched[target_col] = imputed[:, -1]

    return df_enriched


def encode_categorical_columns(df: pd.DataFrame, file_basename: str) -> pd.DataFrame:
    """Encode categorical columns to numeric values based on file-specific mappings."""

    df_encoded = df.copy()

    # Gender encoding mappings
    gender_mappings = {
        "nmbfinalDiabetes (4)": {"Male": 1, "Female": 0, "Transgender": 2, "Other": pd.NA, "Missing": pd.NA},
        "nmbfinalnewDiabetes (3)": {"Male": 1, "Female": 0, "Transgender": pd.NA, "Other": pd.NA, "Missing": pd.NA},
        "PrePostFinal (3)": {"Male": 1, "Female": 0, "Transgender": pd.NA, "Other": pd.NA, "Missing": pd.NA}
    }

    # Area encoding mappings
    area_mappings = {
        "nmbfinalDiabetes (4)": {"Urban": 1, "Rural": 0, "Missing": pd.NA},
        "nmbfinalnewDiabetes (3)": {"Urban": 1, "Rural": 0, "Missing": pd.NA},
        "PrePostFinal (3)": {"Urban": 1, "Rural": 0, "Missing": pd.NA}
    }

    # Determine which mapping to use based on filename
    mapping_key = None
    for key in gender_mappings.keys():
        if key in file_basename:
            mapping_key = key
            break

    if mapping_key:
        # Encode PreRgender
        if "PreRgender" in df_encoded.columns:
            df_encoded["PreRgender"] = df_encoded["PreRgender"].map(gender_mappings[mapping_key]).astype("Int64")

        # Encode PreRarea
        if "PreRarea" in df_encoded.columns:
            df_encoded["PreRarea"] = df_encoded["PreRarea"].map(area_mappings[mapping_key]).astype("Int64")

    return df_encoded


def process_csv_files_enriched(paths, output_dir: str = None, columns_to_keep: list = None,
                              suffix: str = "_selected_columns_cleaned", overwrite: bool = False,
                              enrich_with_features: bool = True, top_feature_percent: float = 0.5):
    """Process one or more CSV files: read -> clean -> save, with optional enrichment.

    Arguments:
        paths: str or list[str] - path or list of paths to input CSV files.
        output_dir: optional directory to write outputs; if None outputs are written
            next to each input file.
        columns_to_keep: optional list of column names to select before saving. If
            None, all columns from the cleaned DataFrame are saved.
        suffix: filename suffix inserted before the extension for output files.
        overwrite: if True will overwrite existing output files.
        enrich_with_features: if True, enrich target columns by imputing missing values
            using important features.
        top_feature_percent: percentage of top features to select for enrichment (default 50%).

    Returns:
        list of output file paths written.

    Usage (from Python):
        from data_prep import process_csv_files_enriched
        process_csv_files_enriched(r"D:\path\to\file.csv")
        process_csv_files_enriched(["a.csv","b.csv"], output_dir="D:\cleaned")
    """
    # normalize input to a list
    if isinstance(paths, (str,)):
        paths = [paths]

    if not isinstance(paths, (list, tuple)):
        raise TypeError("paths must be a path string or a list/tuple of paths")

    out_paths = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skipped '{p}': could not read CSV ({e})")
            continue

        # run cleaning
        df_clean = clean_pre_columns(df)

        # Feature selection and enrichment
        if enrich_with_features:
            # Select important features for the target columns
            target_columns = ["Diabetic_Duration(years)", "systolic", "diastolic", "current_smoking", "current_alcohol"]
            important_features = select_important_features(df_clean, target_columns, top_percent=top_feature_percent)

            # Enrich target columns by imputing missing values using important features
            df_clean = enrich_columns_with_features(df_clean, target_columns, important_features)

        # Encode categorical columns to numeric values based on file-specific mappings
        df_clean = encode_categorical_columns(df_clean, os.path.basename(p))

        # select columns if requested
        if columns_to_keep:
            existing = [c for c in columns_to_keep if c in df_clean.columns]
            if not existing:
                print(f"Warning: none of requested columns were found in '{p}'; saving full cleaned frame instead.")
            else:
                df_clean = df_clean[existing]

        # build output path
        base_name = os.path.splitext(os.path.basename(p))[0]
        out_dir = output_dir if output_dir else os.path.join(os.path.dirname(__file__), "prepared data")
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{base_name}{suffix}.csv"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            print(f"Skipped writing '{out_path}' (exists). Pass overwrite=True to replace.")
            out_paths.append(out_path)
            continue

        try:
            df_clean.to_csv(out_path, index=False)
            out_paths.append(out_path)
            print(f"Wrote cleaned CSV: {out_path}")
        except Exception as e:
            print(f"Failed to write '{out_path}': {e}")

    return out_paths


# Example (non-CLI) usage — call `process_csv_files_enriched` from your training or ETL script:
# from data_prep import process_csv_files_enriched
# process_csv_files_enriched([r'D:\poornima sukumar mam files\PrePostFinal (3).csv'], columns_to_keep=[...])