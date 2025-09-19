
### `SLEEP_QUALITY_ORDINAL`
- Purpose: mapping constant used to convert free-text sleep quality into an ordinal scale.
- Mapping used in `data_prep.py`:
	- `"very good"` -> 3
	- `"fairly good"` -> 2
	- `"fairly bad"` -> 1
	- `"very bad"` -> 0

This mapping is intentionally small and ordered: higher number = better sleep quality.

### `DIETARY_HABITS_ORDINAL`
- Purpose: mapping constant used to convert dietary habit frequency into an ordinal scale for bad habits.
- Mapping used in `data_prep.py`:
	- `"rarely/ never"` -> 2 (Good - rarely do bad things)
	- `"sometimes"` -> 1 (Bad - sometimes do bad things)
	- `"usually/often"` -> 0 (Very bad - usually do bad things)

### `MILK_CONSUMPTION_ORDINAL`
- Purpose: mapping constant used to convert milk consumption frequency into an ordinal scale for good habits.
- Mapping used in `data_prep.py`:
	- `"usually/often"` -> 2 (Good - usually consume milk)
	- `"sometimes"` -> 1 (Moderate - sometimes consume milk)
	- `"rarely/ never"` -> 0 (Bad - rarely consume milk)

### `sleepquality_to_ordinal(series, unknown_value=-1)`
- Purpose: deterministic encoder for `PreRsleepquality` that returns an integer ordinal.
- Input: pandas Series of strings (or NaN). Function uses the `SLEEP_QUALITY_ORDINAL` map after lower-casing and trimming.
- Output: pandas Series of ints where unknown/missing values are set to `unknown_value` (default -1).
- Notes: creates column `PreRsleepquality_ord` when used from `clean_pre_columns`.

### `yesno_to_binary(series)`
- Purpose: normalize many variants of Yes/No-like answers into a pandas nullable integer (Int64) with values 1, 0, or <NA>.
- Recognizes variants such as: `Yes`, `No`, `Y`, `N`, `1`, `0`, `Yes(1)`, `No(0)`, and tolerates punctuation and spacing.
- Behavior:
	- explicit affirmative -> `1`
	- explicit negative -> `0`
	- missing/unknown -> `<NA>` (Int64 dtype preserved)
- Used to convert family-diabetes flags and individual smoking/alcohol binary columns.

### `create_current_smoking(df)`
- Purpose: derived collapsed indicator `current_smoking` from multiple tobacco/smoking-related columns.
- Inputs: DataFrame; function checks these columns (if present):
	- `PreRtobuse`, `PreRtobdaily`, `PreRtobcurrent`, `PreRtobdailyuse`, `PreRtobaverage`,
		`PreRtobsmoking`, `PreRstouse`, `PreRstodaily`, `PreRstocurrent`, `PreRstodailyuse`, `PreRstoyear`
- Logic summary (vectorized):
	- For columns that are free-text or numeric counts (e.g., `PreRtobaverage`, `PreRtobsmoking`, `PreRstoyear`) any non-null value is treated as evidence of smoking (mapped to 1).
	- Other columns are normalized with `yesno_to_binary`.
	- `current_smoking` output: `1` if any checked column indicates smoking, `0` if at least one column is present but none indicate smoking, `<NA>` if none of the smoking columns exist or all are missing.
- Output: pandas Series dtype `Int64` named `current_smoking` when assigned into the DataFrame.

### `create_current_alcohol(df)`
- Purpose: derived collapsed indicator `current_alcohol` from multiple alcohol-related columns.
- Inputs: DataFrame; function checks these columns (if present):
	- `PreRalcoholuse`, `PreRalyear`, `PreRalthreemonths`, `PreRalthreemonthsuse`,
		`PreRallastmonth`, `PreRallastmonthuse`, `PreRlastmonthavg`, `PreRdrinktype`, `PreRalmlperday`
- Logic summary (vectorized):
	- For descriptive/non-binary columns (e.g., `PreRalyear`, `PreRlastmonthavg`, `PreRdrinktype`, `PreRalmlperday`) any non-null value is treated as evidence of alcohol use (mapped to 1).
	- Other columns are normalized with `yesno_to_binary`.
	- `current_alcohol` output: `1` if any checked column indicates alcohol use, `0` if at least one column is present but none indicate alcohol use, `<NA>` if none of the alcohol columns exist or all are missing.
- Output: pandas Series dtype `Int64` named `current_alcohol` when assigned into the DataFrame.

### `map_activity_duration_to_minutes(series, unknown_value=-1)`
- Purpose: deterministic encoder for activity duration columns that returns an integer representing the average minutes for a given time range.
- Input: pandas Series of strings (or NaN).
- Output: pandas Series of ints where unknown/missing values are set to `unknown_value` (default -1).
- Mapping:
	- `'at least 10 mins'` -> 10
	- `'10 - 30mins'` -> 20
	- `'30mins - 1hr'` -> 45
	- `'1hr - 1.5hrs'` -> 75
	- `'>1.5hrs'` -> 105

### `calculate_met_score(df)`
- Purpose: derived `Physical activity_total_METSCORE_value` and `PhysicalActivity_Optimal` from activity duration columns.
- Inputs: DataFrame; function checks these columns (if present):
	- `PreRmildactivityduration`, `PreRmoderateduration`, `PreRvigorousduration`
- Logic summary (vectorized):
	- Uses `map_activity_duration_to_minutes` to get average minutes for each activity type.
	- Calculates `Physical activity_total_METSCORE_value` based on the formula: `(mild_mins * 1.5 * 7) + (moderate_mins * 4.9 * 7) + (vigorous_mins * 6.8 * 7)`.
	- Creates `PhysicalActivity_Optimal`: `'Yes'` if `Physical activity_total_METSCORE_value` is >= 1100, otherwise `'No'`.
- Output: returns the DataFrame with the two new columns.

### `dietary_habits_to_ordinal(series, is_milk=False, unknown_value=-1)`
- Purpose: deterministic encoder for dietary habit columns that returns an integer ordinal representing health impact.
- Input: pandas Series of strings (or NaN), boolean flag for milk consumption.
- Output: pandas Series of ints where unknown/missing values are set to `unknown_value` (default -1).
- For bad habits (skip breakfast, less fiber, less fruit, less vegetables, meat, fried food, popcorn, sweet, drink):
	- `'Rarely/ Never'` -> 2 (Good - rarely do bad things)
	- `'Sometimes'` -> 1 (Bad - sometimes do bad things)
	- `'Usually/Often'` -> 0 (Very bad - usually do bad things)
- For good habits (milk consumption, when `is_milk=True`):
	- `'Usually/Often'` -> 2 (Good - usually consume milk)
	- `'Sometimes'` -> 1 (Moderate - sometimes consume milk)
	- `'Rarely/ Never'` -> 0 (Bad - rarely consume milk)
- Notes: creates columns with `_ord` suffix when used from `clean_pre_columns`.
### `calculate_bmi(height_cm, weight_kg)`
- Purpose: compute Body Mass Index (BMI) and surface it as `PreRBMI`.
- Input: two pandas Series: height in centimetres (`PreRheight`) and weight in kilograms (`PreRweight`).
- Output: pandas Series (float) with BMI values calculated as weight_kg / (height_m ** 2). Missing or invalid inputs produce `NaN`.
- Validation/notes:
	- Heights are expected in cm and are converted to metres inside the function.
	- The function guards against invalid measurements and sets BMI to `NaN` when height or weight is non-positive or when the computed BMI is outside a reasonable range (the implementation uses 10 < BMI < 100).
	- `clean_pre_columns` will create `PreRBMI` when both `PreRheight` and `PreRweight` exist in the DataFrame.

### `calculate_waist_hip_ratio(waist_cm, hip_cm)`
- Purpose: compute waist-to-hip ratio (WHR) and surface it as `PreWaisttoHipRatio`.
- Input: two pandas Series: waist and hip circumferences in centimetres (`PreRwaist`, `PreRhip`).
- Output: pandas Series (float) with WHR values (waist_cm / hip_cm). Missing or invalid inputs produce `NaN`.
- Validation/notes:
	- The function returns `NaN` for non-positive measurements and for extreme/unrealistic ratios. The implementation enforces 0.3 < WHR < 3.
	- `clean_pre_columns` will create `PreWaisttoHipRatio` when both `PreRwaist` and `PreRhip` exist in the DataFrame.

### `process_csv_files(paths, output_dir=None, columns_to_keep=None, suffix="_selected_columns_cleaned", overwrite=False)`
- Purpose: convenience helper to read CSV(s), run `clean_pre_columns`, optionally select columns and write cleaned CSV(s).
- Behavior summary:
	- Accepts a single file path string or a list of paths.
	- Reads each CSV with `pandas.read_csv`, runs `clean_pre_columns(df)`.
	- If `columns_to_keep` is provided only existing columns from that list are saved; otherwise the full cleaned frame is saved.
	- Writes output beside input file or to `output_dir` if specified. Default filename suffix is `"_selected_columns_cleaned"`.
	- Skips writing if output exists unless `overwrite=True`.


## Added cleaning functions

I added a small set of pandas-friendly cleaning functions in `data_prep.py` to handle
these columns: `PreRgender`, `PreRarea`, `PreRmaritalstatus`, `PreReducation`,
and `PreRpresentoccupation`.

What was added (functions)
- `normalize_gender(series)` -> returns categorical with values: 'Male','Female','Transgender','Other','Missing'
- `map_area(series, as_numeric=False)` -> categorical 'Urban'/'Rural'/'Missing' (or numeric 1/0/-1)
- `clean_marital_status(series, group_rare=True)` -> categorical with rare categories grouped into 'Other'
- `education_to_ordinal(series)` -> integer ordinal mapping (No formal schooling=0 ... University completed or higher=5), unknown -> -1
- `group_occupation(series)` -> grouped occupation categories (Professional, Clerical, Homemaker, Self-employed, Unskilled, Retired, Unemployed, Other, Missing)
- `clean_pre_columns(df)` -> convenience wrapper applying all cleaners to a DataFrame and adding `PreReducation_ord` and `PreRpresentoccupation_grp`.

Additional change added
- Family diabetes flags (`PreRdiafather`, `PreRdiamother`, `PreRdiabrother`, `PreRdiasister`) are converted from Yes/No variants to nullable integer 1/0 using a new helper `yesno_to_binary` in `data_prep.py`.
- Family diabetes flags (`PreRdiafather`, `PreRdiamother`, `PreRdiabrother`, `PreRdiasister`) and `PreRcurrentworking` are converted from Yes/No variants to nullable integer 1/0 using a new helper `yesno_to_binary` in `data_prep.py`.
- Individual tobacco/smoking binary columns (`PreRtobuse`, `PreRtobdaily`, `PreRtobcurrent`, `PreRtobdailyuse`, `PreRstouse`, `PreRstodaily`, `PreRstocurrent`, `PreRstodailyuse`) are converted to 1/0.
- New derived variable `current_smoking` (nullable Int64) created from 11 tobacco/smoking columns: returns 1 if any column has "Yes" or non-"No" value, 0 if all are "No", <NA> if all missing.
- New derived variable `current_alcohol` (nullable Int64) created from 9 alcohol consumption columns: returns 1 if any column has "Yes" or non-"No" value, 0 if all are "No", <NA> if all missing.
- Activity duration columns (`PreRmildactivityduration`, `PreRmoderateduration`, `PreRvigorousduration`) are converted to `_mins` columns with average minutes.
- Physical activity MET scores are calculated and summed into `Physical activity_total_METSCORE_value`.
- `PhysicalActivity_Optimal` column created: "Yes" if MET score >= 1100, otherwise "No".
- Dietary habit columns converted to ordinal scores with `_ord` suffix:
  - Bad habits (skip breakfast, less fiber, less fruit, less vegetables, meat, fried food, popcorn, sweet, drink): 2=rarely (good), 1=sometimes (bad), 0=usually (very bad)
  - Good habits (milk): 2=usually (good), 1=sometimes (moderate), 0=rarely (bad)


