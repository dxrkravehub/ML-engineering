
import fitz
import os
from PIL import Image
import io
import requests
from google.colab import userdata
import google.generativeai as genai
import json

def pdf_to_image(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        return None
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            print(f"PDF file '{pdf_path}' contains no pages.")
            doc.close()
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        doc.close()
        return image
    except Exception as e:
        print(f"An error occurred during PDF to image conversion for '{pdf_path}': {e}")
        return None

def get_ocr_text(image: Image.Image, api_key: str = None, ocr_url: str = 'https://api.ocr.space/parse/image') -> str | None:
    if image is None:
        print("Error: No image provided for OCR.")
        return None
    if api_key is None:
        try:
            api_key = userdata.get('API_KEY')
            if not api_key:
                print("Warning: OCR.space API Key not found. Please add it to Colab secrets as 'API_KEY'.")
                return None
        except Exception as e:
            print(f"Warning: Error reading OCR.space API key from Colab secrets: {e}")
            return None
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        files = {'filename': ('image.png', img_byte_arr, 'image/png')}
        data = {'apikey': api_key, 'language': 'auto', 'ocrEngine': '2'}
        response = requests.post(ocr_url, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        if result and result.get('OCRExitCode') == 1:
            parsed_text = ""
            for region in result['ParsedResults']:
                parsed_text += region['ParsedText'] + "\n"
            return parsed_text
        elif result and result.get('OCRExitCode') in [2, 3]:
             print(f"OCR failed. Error message: {result.get('ErrorMessage', 'Unknown error')}")
             return None
        else:
            print(f"Unexpected response from OCR.space: {result}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling OCR.space API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OCR.space processing: {e}")
        return None

def postprocess_with_llm(raw_text: str, gemini_model: genai.GenerativeModel, fields_to_extract: list) -> dict | None:
    if not raw_text:
        print("Warning: No raw text provided for LLM post-processing.")
        return None
    if gemini_model is None:
        print("Error: Gemini model not provided.")
        return None
    if not fields_to_extract:
        print("Warning: No fields specified for extraction by LLM.")
        return {}
    prompt = "You are an expert document processor. Here is the raw text extracted from a document:\n\n" + raw_text + "\n\n" + \
             "Your task:\n" + \
             "1. Analyze the text and attempt to extract key information for the following fields: " + ', '.join(fields_to_extract) + ".\n" + \
             "2. Structure the extracted data in JSON format. Use the following English keys: " + ', '.join(fields_to_extract) + ".\n" + \
             "3. Attempt to format dates as YYYY-MM-DD, if possible.\n" + \
             "4. Ensure numerical fields contain only numbers.\n" + \
             "5. If a field is not found, include it in the JSON with a value of null.\n" + \
             "6. Return only the JSON object, with no extra words or formatting outside the JSON block.\n\n" + \
             "Example expected JSON format:\n" + \
             "```json\n" + \
             "{\n"
    for i, field in enumerate(fields_to_extract):
        prompt += f'  "{field}": "..."'
        if i < len(fields_to_extract) - 1:
            prompt += ","
        prompt += "\n"
    prompt += "}\n```"
    try:
        response_llm = gemini_model.generate_content(prompt)
        gemini_text = response_llm.text.strip()
        try:
            if gemini_text.startswith("```json"):
                gemini_text = gemini_text[len("```json"):].strip()
            if gemini_text.endswith("```"):
                gemini_text = gemini_text[:-len("```")].strip()
            extracted_data = json.loads(gemini_text)
            return extracted_data
        except json.JSONDecodeError:
            print("Failed to decode Gemini's response as JSON.")
            print("Gemini's raw response:")
            print(gemini_text)
            return None
        except Exception as e:
            print(f"An error occurred while processing Gemini's JSON response: {e}")
            return None
    except Exception as e:
        print(f"An error occurred while calling Gemini API: {e}")
        return None

def process_pdf(pdf_path: str, gemini_model: genai.GenerativeModel = None, fields_to_extract: list = None) -> str | dict | None:
    print(f"Starting PDF processing for: {pdf_path}")
    image = pdf_to_image(pdf_path)
    if image is None:
        print("PDF to image conversion failed.")
        return None
    raw_text = get_ocr_text(image)
    if raw_text is None:
        print("OCR text extraction failed.")
        return None
    if gemini_model is not None and fields_to_extract is not None:
        structured_data = postprocess_with_llm(raw_text, gemini_model, fields_to_extract)
        if structured_data is None:
            print("LLM post-processing failed or returned no data.")
            return None
        return structured_data
    else:
        return raw_text
