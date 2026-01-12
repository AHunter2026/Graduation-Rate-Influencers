import pandas as pd
import numpy as np
import os
from pathlib import Path

######## CONFIGURATION ########

# Target countries and years
TARGET_COUNTRIES = ['DEU', 'FRA', 'GBR']
COUNTRY_NAMES = {
    'DEU': 'Germany',
    'FRA': 'France',
    'GBR': 'United Kingdom'
}
TARGET_YEARS = [2017, 2018, 2019]

# File paths
DATA_DIR = './Raw_Data'
OUTPUT_DIR = './Cleaned_Data'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

######## HELPER FUNCTION ########

def load_excel_safe(filepath, sheet_name=0):
    """Safely load Excel file with error handling"""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        print(f"{filepath} loaded Without Issue")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

######## CLEAN TEACHER WORKING HOURS DATA ########

def clean_teacher_hours():
    """
    Extract and clean teacher working hours data from OECD files.
    Extracts Upper Secondary (General) education level to align with secondary graduation rates.
    Returns a DataFrame with columns: country, year, teacher_hours
    """

    hours_data = []

    # Files, sheet names, and column numbers for Upper Secondary (General) Total Working Time
    # Note: Different years have different table structures
    # I used Upper Secondary to align with secondary education graduation rates
    files_config = {
        2017: {'file': 'OECD_Hours_2017.XLSX', 'sheet': 'Table D4.1.', 'col': 58},
        2018: {'file': 'OECD_Hours_2018.XLSX', 'sheet': 'Table D4.1b.', 'col': 22},
        2019: {'file': 'OECD_Hours_2019.XLSX', 'sheet': 'Table D4.2.', 'col': 22}
    }

    # Country name mapping
    country_name_map = {
        'DEU': 'Germany',
        'FRA': 'France',
        'GBR': 'England (UK)'  # Note: UK is represented as "England (UK)" in OECD data
    }

    for year, config in files_config.items():
        filepath = os.path.join(DATA_DIR, config['file'])
        print(f"\nProcessing {year} data from {config['sheet']}")

        df = pd.read_excel(filepath, sheet_name=config['sheet'])

        # Get the column number for this year's file structure
        primary_col = config['col']

        # Find rows for our target countries
        for country_code in TARGET_COUNTRIES:
            country_name = country_name_map[country_code]
            country_row = None

            # Find the row with this country
            for idx, row in df.iterrows():
                cell_value = str(row.iloc[0])
                if cell_value.startswith(country_name):
                    country_row = idx
                    break

            if country_row is not None:
                hours_value = df.iloc[country_row, primary_col]

                # Clean the value
                if pd.notna(hours_value) and hours_value not in ['m', 'a', '..']:
                    try:
                        hours_value = float(hours_value)
                        hours_data.append({
                            'country': country_code,
                            'year': year,
                            'teacher_hours': hours_value
                        })
                        print(f"{country_code} ({country_name}): {hours_value} hours")
                    except (ValueError, TypeError):
                        print(f"{country_code}: Invalid value '{hours_value}'")
                else:
                    print(f"{country_code}: Missing data (value: {hours_value})")
            else:
                print(f"{country_code} ({country_name}): Not found in file")

    # Create DataFrame
    df_hours = pd.DataFrame(hours_data)

    print(f"\nTeacher hours data extracted: {len(df_hours)} records")
    return df_hours

######## CLEAN CLASS SIZE DATA ########

def clean_class_size():
    """
    Extract and clean average class size data from OECD.
    Returns a DataFrame with columns: country, year, class_size
    """

    filepath = os.path.join(DATA_DIR, 'OECD_Avg_Class_Size.csv')
    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")

    # Filter for chosen countries, years, and secondary education
    df_filtered = df[
        (df['REF_AREA'].isin(TARGET_COUNTRIES)) &
        (df['TIME_PERIOD'].isin(TARGET_YEARS)) &
        (df['EDUCATION_LEV'] == 'ISCED11_2')  # secondary education
        ].copy()

    print(f"After filtering: {df_filtered.shape}")

    # Group by country and year, taking the mean of class sizes
    # (There are multiple records per country-year, likely different types of schools)
    df_class = df_filtered.groupby(['REF_AREA', 'TIME_PERIOD'])['OBS_VALUE'].mean().reset_index()
    df_class.columns = ['country', 'year', 'class_size']

    print(f"\nClass size data extracted: {len(df_class)} records")
    for _, row in df_class.iterrows():
        print(f"{row['country']} {row['year']}: {row['class_size']:.1f} students")

    return df_class

######## CLEAN EXPENDITURE DATA ########

def clean_expenditure():
    """
    Extract and clean government expenditure on education from World Bank.
    Returns a DataFrame with columns: country, year, expenditure_pct_gdp
    """

    filepath = os.path.join(DATA_DIR, 'WB_Expenditures.csv')
    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:20]}...")  # Show first 20 columns

    # Filter for chosen countries
    df_filtered = df[df['REF_AREA'].isin(TARGET_COUNTRIES)].copy()

    print(f"After filtering countries: {df_filtered.shape}")

    # Extract year columns (2017, 2018, 2019)
    expenditure_data = []

    for _, row in df_filtered.iterrows():
        country = row['REF_AREA']
        for year in TARGET_YEARS:
            year_col = str(year)
            if year_col in df.columns:
                value = row[year_col]
                if pd.notna(value):
                    expenditure_data.append({
                        'country': country,
                        'year': year,
                        'expenditure_pct_gdp': float(value)
                    })
                    print(f"{country} {year}: {value:.2f}% of GDP")
                else:
                    print(f"{country} {year}: Missing")

    df_expenditure = pd.DataFrame(expenditure_data)

    print(f"\nExpenditure data extracted: {len(df_expenditure)} records")
    return df_expenditure

######## CLEAN GRADUATION RATES DATA ########

def clean_graduation_rates():
    """
    Extract and clean graduation rates from UNESCO.
    Returns a DataFrame with columns: country, year, graduation_rate
    """

    filepath = os.path.join(DATA_DIR, 'UNESCO_Graduation_Rates.csv')
    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Filter for our countries and years
    df_filtered = df[
        (df['geoUnit'].isin(TARGET_COUNTRIES)) &
        (df['year'].isin(TARGET_YEARS))
        ].copy()

    print(f"After filtering: {df_filtered.shape}")

    # Extract the data we need
    df_filtered = df_filtered.rename(columns={
        'geoUnit': 'country',
        'value': 'graduation_rate'
    })

    df_grad = df_filtered[['country', 'year', 'graduation_rate']].copy()

    print(f"\nGraduation rates extracted: {len(df_grad)} records")
    for _, row in df_grad.iterrows():
        print(f"{row['country']} {row['year']}: {row['graduation_rate']:.1f}%")

    return df_grad

######## MERGE ALL DATASETS ########

def merge_datasets(df_hours, df_class, df_expenditure, df_grad):
    """
    Merge all four datasets on country and year.
    Returns final cleaned dataset ready for analysis.
    """

    # Start with teacher hours
    df_merged = df_hours.copy()
    print(f"Starting with teacher hours: {df_merged.shape}")

    # Merge class size
    df_merged = df_merged.merge(df_class, on=['country', 'year'], how='left')
    print(f"After adding class size: {df_merged.shape}")

    # Merge expenditure
    df_merged = df_merged.merge(df_expenditure, on=['country', 'year'], how='left')
    print(f"After adding expenditure: {df_merged.shape}")

    # Merge graduation rates
    df_merged = df_merged.merge(df_grad, on=['country', 'year'], how='left')
    print(f"After adding graduation rates: {df_merged.shape}")

    # Add country names
    df_merged['country_name'] = df_merged['country'].map(COUNTRY_NAMES)

    # Reorder columns for readability
    column_order = [
        'country', 'country_name', 'year',
        'graduation_rate', 'teacher_hours', 'class_size', 'expenditure_pct_gdp'
    ]
    df_merged = df_merged[column_order]

    # Sort by country and year
    df_merged = df_merged.sort_values(['country', 'year']).reset_index(drop=True)

    return df_merged

######## VALIDATE FINAL DATASET ########

def validate_dataset(df):
    """
    Validate the final merged dataset for completeness and quality.
    """

    # Checking for missing values
    print("\nMissing Values Check:")
    missing = df.isnull().sum()
    for col in missing.index:
        if missing[col] > 0:
            print(f"{col}: {missing[col]} missing values")
        else:
            print(f"{col}: Complete")

    # Checking for expected number of records
    expected_records = len(TARGET_COUNTRIES) * len(TARGET_YEARS)
    actual_records = len(df)
    print(f"\nRecord Count:")
    print(f"Expected: {expected_records} ({len(TARGET_COUNTRIES)} countries × {len(TARGET_YEARS)} years)")
    print(f"Actual: {actual_records}")

    if actual_records == expected_records:
        print("Complete dataset!")
    else:
        print("Missing records")

    # Show summary statistics
    print("\nSummary Statistics:")
    print(df.describe().round(2))

    # Show final dataset
    print("FINAL DATASET:")
    print("=" * 80)
    print(df.to_string(index=False))

    return len(missing[missing > 0]) == 0

######## MAIN EXECUTION ########

def main():
    """
    Main execution function that orchestrates the entire cleaning process.
    """
    print("Data Cleaning Script Summary")
    print("  Target: DEU, FRA, GBR | Years: 2017-2019")
    print("=" * 80)

    # Clean each dataset
    df_hours = clean_teacher_hours()
    df_class = clean_class_size()
    df_expenditure = clean_expenditure()
    df_grad = clean_graduation_rates()

    # Merge all datasets
    df_final = merge_datasets(df_hours, df_class, df_expenditure, df_grad)

    # Validate
    is_complete = validate_dataset(df_final)

    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, 'clean_graduation_influencers.csv')
    df_final.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")

    # Also save detailed versions for each dataset
    df_hours.to_csv(os.path.join(OUTPUT_DIR, 'clean_teacher_hours.csv'), index=False)
    df_class.to_csv(os.path.join(OUTPUT_DIR, 'clean_class_size.csv'), index=False)
    df_expenditure.to_csv(os.path.join(OUTPUT_DIR, 'clean_expenditure.csv'), index=False)
    df_grad.to_csv(os.path.join(OUTPUT_DIR, 'clean_graduation_rates.csv'), index=False)

    print("Individual cleaned datasets also saved")

    if is_complete:
        print("DATA CLEANING COMPLETE - READY FOR ANALYSIS!")
        print("=" * 80)
    else:
        print("DATA CLEANING COMPLETE - CHECK MISSING VALUES")
        print("=" * 80)

    return df_final


if __name__ == "__main__":
    df_final = main()