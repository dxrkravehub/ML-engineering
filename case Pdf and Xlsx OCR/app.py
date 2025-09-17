import streamlit as st
import os
import sys
import io
import json
import pandas as pd
import fitz
import requests
import numpy as np
from google.colab import userdata
import google.generativeai as genai
from PIL import Image

# Add the project directory to the Python path to import local modules
# In Colab, __file__ is not defined, so we assume the project directory is in the current path.
# If running as a standalone script, os.path.dirname(os.path.abspath(__file__)) would be used.
project_dir = "document_processor_app"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Now we can import the local modules
try:
    from pdf_processor import process_pdf
    from xlsx_processor import process_xlsx
except ImportError as e:
    st.error(f"Failed to import local modules: {e}. Make sure '{project_dir}' directory is in the correct path and contains 'pdf_processor.py' and 'xlsx_processor.py'.")
    st.stop() # Stop the Streamlit app if essential imports fail


# Initialize Gemini model - Attempt to get key from userdata first, then environment variables
GOOGLE_API_KEY = None
try:
    st.write("Attempting to retrieve GOOGLE_API_KEY from Colab secrets (userdata)...")
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        st.success("GOOGLE_API_KEY successfully retrieved from Colab secrets.")
        print("DEBUG: GOOGLE_API_KEY successfully retrieved using userdata.get().")
    else:
        st.warning("GOOGLE_API_KEY not found using userdata.get().")
        print("DEBUG: GOOGLE_API_KEY is None or empty after userdata.get().")

except Exception as e:
    st.warning(f"Error retrieving GOOGLE_API_KEY from Colab secrets (userdata): {e}")
    print(f"DEBUG: Exception during userdata.get(): {e}")

# If userdata.get failed or returned None, try environment variables
if not GOOGLE_API_KEY:
    try:
        st.write("Attempting to retrieve GOOGLE_API_KEY from environment variables...")
        GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        if GOOGLE_API_KEY:
            st.success("GOOGLE_API_KEY successfully retrieved from environment variables.")
            print("DEBUG: GOOGLE_API_KEY successfully retrieved using os.environ.get().")
        else:
            st.warning("GOOGLE_API_KEY not found in environment variables.")
            print("DEBUG: GOOGLE_API_KEY is None or empty after os.environ.get().")
    except Exception as e:
        st.warning(f"Error retrieving GOOGLE_API_KEY from environment variables: {e}")
        print(f"DEBUG: Exception during os.environ.get(): {e}")


# Initialize the Gemini model if the API key was found by either method
gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        st.success("Gemini model initialized.")
        print("DEBUG: Gemini model initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize Gemini model with provided API key: {e}. Please check your GOOGLE_API_KEY.")
        print(f"DEBUG: Exception during Gemini initialization: {e}")
else:
    st.error("GOOGLE_API_KEY not available. Gemini model cannot be initialized.")
    print("DEBUG: GOOGLE_API_KEY not available, skipping Gemini model initialization.")


st.title("Document Processor App")
st.write("Upload a PDF or XLSX file to extract key information.")

uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'xlsx'])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    st.write(f"Processing file: **{uploaded_file.name}**")

    # Create a temporary file to save the uploaded content
    temp_dir = "/tmp/streamlit_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extracted_data = None

    if file_extension == ".pdf":
        st.write("Detected PDF file. Processing...")
        pdf_fields_to_extract = ["document_number", "document_date", "seller", "buyer", "total_amount", "subject"]

        # Ensure gemini_model is available before attempting LLM post-processing
        if gemini_model:
            try:
                # Call the refactored process_pdf function
                extracted_data = process_pdf(temp_file_path, gemini_model=gemini_model, fields_to_extract=pdf_fields_to_extract)

                if isinstance(extracted_data, dict):
                    st.write("Extracted Data (Structured JSON):")
                    st.json(extracted_data)
                elif isinstance(extracted_data, str):
                    st.write("Raw text extracted from PDF (LLM structuring skipped or failed):")
                    st.text(extracted_data)
                elif extracted_data is None:
                    st.warning("PDF processing failed.")

            except Exception as e:
                st.error(f"An error occurred during PDF processing: {e}")
        else:
            st.warning("Gemini model not initialized. Skipping LLM post-processing for PDF.")
            # If Gemini model is not available, still try to get raw text via OCR
            try:
                 image = fitz.open(temp_file_path).load_page(0).get_pixmap().tobytes("png")
                 raw_text = get_ocr_text(Image.open(io.BytesIO(image)).convert("RGB"))
                 if raw_text:
                      st.write("Raw text extracted from PDF (Gemini not available):")
                      st.text(raw_text)
                 else:
                      st.warning("Failed to extract raw text from PDF via OCR.")
            except Exception as e:
                 st.error(f"Error during raw text extraction fallback: {e}")


    elif file_extension == ".xlsx":
        st.write("Detected XLSX file. Processing...")
        # Define placeholder extraction rules for XLSX
        # IMPORTANT: Replace these rules with the actual structure of your XLSX files
        xlsx_extraction_rules = {
            "sheet_name": "Sheet1", # Replace with the actual sheet name
            "fields": {
                "document_number": "B2",
                "document_date": "C2",
                "seller": "B3",
                "buyer": "B4",
                "total_amount": "D5",
                "subject": "B6"
            }
        }
        try:
            # Call the refactored process_xlsx function
            extracted_data = process_xlsx(temp_file_path, xlsx_extraction_rules)

            if extracted_data:
                st.write("Extracted Data (Structured JSON):")
                st.json(extracted_data)
            else:
                st.warning("XLSX processing failed.")

        except Exception as e:
            st.error(f"An error occurred during XLSX processing: {e}")

    else:
        st.warning(f"Unsupported file type: {file_extension}")

    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

elif uploaded_file is None:
    st.info("Please upload a PDF or XLSX file.")

# Add dummy get_ocr_text if it's not imported due to import error
if 'get_ocr_text' not in locals():
    def get_ocr_text(image, api_key=None, ocr_url=None):
        print("get_ocr_text dummy function called. Original import failed.")
        return None
