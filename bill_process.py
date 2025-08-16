from dotenv import load_dotenv
import os
from PIL import Image
import base64
import pandas as pd
import json
import pytesseract
import cv2
import numpy as np
import re
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use gemini-2.0-flash model for multi-modal input
model = genai.GenerativeModel("gemini-2.0-flash")

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

def extract_json_from_codeblock(text):
    """
    Extracts JSON string from markdown-style code block like ```json ... ```
    """
    if "```json" in text:
        # Remove triple backticks and 'json' identifier
        json_str = text.split("```json")[1].split("```")[0].strip()
        return json_str
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0].strip()
        return json_str
    else:
        # No code block detected, maybe raw JSON
        print("Warning: No markdown code block detected; returning original trimmed text.")
        return text.strip()

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            image_binary = img_file.read()
            base64_data = base64.b64encode(image_binary).decode("utf-8")
            return base64_data
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def performOCR():
    img_path = './download.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    print("OCR Output:\n", text)
    return text

def get_bill_details(model):
    invoice_text = performOCR()

    lines = invoice_text.splitlines()
    item_lines = []
    for line in lines:
        if re.match(r"^[A-Z].*\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+$", line):
            if not re.search(r"\b(BAG|DISCOUNT|SCHEME|TOTAL|GST|TAX|ROUND|THANK|SUBTOTAL)\b", line, re.IGNORECASE):
                item_lines.append(line.strip())
    filtered_text = "\n".join(item_lines)
    print("Filtered for items:\n", filtered_text)

    prompt = f"""
You are given a messy OCR-extracted text from a retail invoice. Extract a JSON array of the purchased items.
For each item, extract only:
- name (the product, such as 'BAKE FRESH TO', 'SPRITE PET 1.')
- quantity (matching the 'Qty' or second number on each line, may be 1.00, 2.00 etc.)
- unit_price (the price per each item, not the amount; use the first price after the product name.)
- total (the total for the item row, which equals unit_price * quantity; it's the last price in that line)
Ignore all GST, discount, subtotal, roundoff, tax, bag, and payment lines. Do not include lines that are only codes in parentheses or curly brackets. Disregard sections that list GST, taxes, totals, or approval codes.
Receipt lines:
{filtered_text}
"""

    image_path = "download.jpg"
    if not os.path.exists(image_path):
        print("Image file does not exist.")
        return None

    image = Image.open(image_path)

    try:
        response = model.generate_content([prompt, image], generation_config=generation_config)
    except Exception as e:
        print(f"Model call failed: {e}")
        return None

    candidates = getattr(response, "candidates", [])
    if candidates:
        for candidate in candidates:
            if getattr(candidate, "finish_reason", None) == 2:
                print("Model returned no result. Finish reason 2.")
                continue
            content = getattr(candidate, "content", None)
            if content:
                parts = getattr(content, "parts", None)
                if parts:
                    combined_text = "".join([part.text for part in parts if hasattr(part, "text")])
                    print("Gemini extracted JSON string:\n", combined_text)
                    return combined_text
                else:
                    print("No parts present in content.")
    print("No valid response from Gemini model. Full response:", response)
    return None

def add_item_to_dataframe(item, quantity, price, total):
    global df
    new_row = pd.DataFrame([[item, quantity, price, total]], columns=['Item', 'Quantity', 'Price', 'Total'])
    df = pd.concat([df, new_row], axis=0, ignore_index=True)
    return json.dumps(df.to_json(orient='records'))

df = pd.DataFrame(columns=['Item', 'Quantity', 'Price', 'Total'])

def get_dataframes_using_convo(model, bill_details):
    try:
        json_str = extract_json_from_codeblock(bill_details)
        items = json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse Gemini JSON output: {e}")
        print("Raw Gemini output:", bill_details)
        return pd.DataFrame(columns=['Item Name', 'Quantity', 'Unit Price', 'Total'])

    data = []
    for item in items:
        name = item.get('name', 'Unknown')
        quantity = float(item.get('quantity', 1))
        unit_price = float(item.get('unit_price', 0))
        total = float(item.get('total', quantity * unit_price))
        data.append([name, quantity, unit_price, total])

    df = pd.DataFrame(data, columns=['Item Name', 'Quantity', 'Unit Price', 'Total'])
    return df


if __name__ == "__main__":
    bill_details = get_bill_details(model)
    if bill_details:
        df = get_dataframes_using_convo(model, bill_details)
        print(df)
        df.to_csv('bill.csv', index=False)
    else:
        print("No bill details extracted.")
