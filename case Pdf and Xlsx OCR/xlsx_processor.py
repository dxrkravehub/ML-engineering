
import pandas as pd
import os
import json
import numpy as np

def extract_data_from_xlsx(xlsx_path: str, extraction_rules: dict) -> dict | None:
    if not os.path.exists(xlsx_path):
        print(f"Error: File not found at '{xlsx_path}'")
        return None
    if not extraction_rules or "sheet_name" not in extraction_rules or "fields" not in extraction_rules:
        print("Error: Invalid or missing extraction rules.")
        return None
    sheet_name = extraction_rules["sheet_name"]
    fields_to_extract = extraction_rules["fields"]
    extracted_data = {}
    try:
        xls = pd.ExcelFile(xlsx_path)
        if sheet_name not in xls.sheet_names:
            print(f"Error: Sheet '{sheet_name}' not found in '{xlsx_path}'")
            return None
        df = xls.parse(sheet_name)
        for field, cell in fields_to_extract.items():
            try:
                col_letter = ''.join(filter(str.isalpha, cell)).upper()
                row_number = int(''.join(filter(str.isdigit, cell)))
                col_index = ord(col_letter) - ord('A')
                row_index = row_number - 1
                if row_index < 0 or col_index < 0 or row_index >= df.shape[0] or col_index >= df.shape[1]:
                     print(f"Warning: Cell '{cell}' for field '{field}' is outside the DataFrame bounds.")
                     extracted_data[field] = None
                     continue
                value = df.iloc[row_index, col_index]
                if pd.isna(value):
                    extracted_data[field] = None
                elif isinstance(value, (int, float)):
                     extracted_data[field] = float(value) if np.isfinite(value) else None
                else:
                    extracted_data[field] = str(value).strip()
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not extract data for field '{field}' from cell '{cell}': {e}")
                extracted_data[field] = None
            except Exception as e:
                 print(f"An unexpected error occurred while processing cell '{cell}' for field '{field}': {e}")
                 extracted_data[field] = None
        return extracted_data
    except FileNotFoundError:
        print(f"Error: XLSX file not found at '{xlsx_path}'")
        return None
    except Exception as e:
        print(f"An error occurred during XLSX processing for '{xlsx_path}': {e}")
        return None

def process_xlsx(xlsx_path: str, extraction_rules: dict) -> dict | None:
    print(f"Starting XLSX processing for: {xlsx_path}")
    extracted_data = extract_data_from_xlsx(xlsx_path, extraction_rules)
    if extracted_data is None:
        print("XLSX data extraction failed.")
    else:
        print("XLSX data extraction successful.")
    return extracted_data
