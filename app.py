from fastapi import FastAPI, File, UploadFile
import pytesseract
from PIL import Image
import re
import pickle
import pandas as pd
from pdf2image import convert_from_bytes
from io import BytesIO
import numpy as np
import math

app = FastAPI()

# Load the trained Random Forest model
with open("DT_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

disease_mapping = {
    0: "All values are within normal range",
    1: "Immunodeficiency",
    2: "Macrocytic Hyperchromic (megaloblastic anemia)",
    3: "Microcytic Hypochromic (Iron deficiency anemia)",
    4: "Normocytic Normochromic (Acute bleeding)",
    5: "Polycythemia",
    6: "Thalassemia Trait or Early Iron Deficiency",
    7: "You have an infection",
    8: "You have thrombocytopenia",
    9: "You have thrombocytosis",
}


# Function to extract text from an image or PDF
def extract_text_from_file(file: bytes, filename: str):
    if filename.lower().endswith(".pdf"):
        images = convert_from_bytes(file)
        text = "".join(pytesseract.image_to_string(img) for img in images)
    else:
        image = Image.open(BytesIO(file))
        text = pytesseract.image_to_string(image)
    return text


# Function to clean and preprocess the extracted text
def preprocess_text(text):
    corrections = {
        "—": "-",
        "=": "-",
        "&": "%",
        "rt": "13",
        "di.i": "11.1",
        "On7": "0.7",
        "1341": "13.1",
        "353": "3.53",
        "253": "2.53",
        "ag - OY": "79 - 97",
        "X10E3/uL 3.4": "3.4 - 10.8",
    }
    for old, new in corrections.items():
        text = text.replace(old, new)
    return text


# Function to extract CBC values from text
def parse_cbc_results(text):
    target_parameters = {
        "WBC": ["White Blood Cells", "WBC", "Leukocytes"],
        "HB": ["Hemoglobin", "HB", "HGB"],
        "MCV": ["Mean Corpuscular Volume", "MCV"],
        "PC": ["Platelet Count", "Platelets", "PC"],
        "RBC": ["Red Blood Cells", "RBC"],
        "MCH": ["Mean Corpuscular Hemoglobin", "MCH"],
        "MCHC": ["Mean Corpuscular Hemoglobin Concentration", "MCHC"],
        "RDW": ["Red Cell Distribution Width", "RDW"],
        "Age": ["Age", "DOB", "Birth Year"],
        "Gender": ["Gender", "Sex"],
    }

    results = {}
    lines = text.splitlines()

    for line in lines:
        if re.search(r"\d", line) or any(
            kw.lower() in line.lower() for kw in ["male", "female"]
        ):
            for param, keywords in target_parameters.items():
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    if param == "Gender":
                        if "female" in line.lower():
                            results[param] = "Female"
                        elif "male" in line.lower():
                            results[param] = "Male"
                    else:
                        match = re.search(r"([\d.]+)", line)
                        if match:
                            results[param] = float(match.group(1))
                    break

    # Default values for missing ones
    results.setdefault("Age", np.nan)
    results.setdefault("Gender", "Male")
    results.setdefault("RBC", np.nan)
    results.setdefault("MCH", np.nan)
    results.setdefault("MCHC", np.nan)
    results.setdefault("RDW", np.nan)

    return results


def replace_nan(obj):
    if isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


# API endpoint for processing CBC reports
@app.post("/analyze_cbc")
async def upload_file(file: UploadFile = File(...)):
    file_content = await file.read()
    extracted_text = extract_text_from_file(file_content, file.filename)
    cleaned_text = preprocess_text(extracted_text)
    cbc_results = parse_cbc_results(cleaned_text)

    # Convert gender to numerical (Male: 0, Female: 1)
    gender_val = cbc_results.get("Gender", "Male")
    cbc_results["Gender"] = 1 if gender_val.lower() == "female" else 0

    # Convert to DataFrame with correct order
    feature_order = [
        "Gender",
        "Age",
        "Haemoglobin",
        "Red Cell Count",
        "MCV",
        "MCH",
        "MCHC",
        "RDW",
        "TLC",
        "Platelet Count",
    ]
    input_data = pd.DataFrame(
        [
            {
                "Gender": cbc_results.get("Gender", 0),
                "Age": cbc_results.get("Age", np.nan),
                "Haemoglobin": cbc_results.get("HB", np.nan),
                "Red Cell Count": cbc_results.get("RBC", np.nan),
                "MCV": cbc_results.get("MCV", np.nan),
                "MCH": cbc_results.get("MCH", np.nan),
                "MCHC": cbc_results.get("MCHC", np.nan),
                "RDW": cbc_results.get("RDW", np.nan),
                "TLC": cbc_results.get("WBC", np.nan),
                "Platelet Count": cbc_results.get("PC", np.nan),
            }
        ]
    )
    input_data.fillna(input_data.mean(), inplace=True)

    prediction = int(model.predict(input_data)[0])

    prediction_disease = disease_mapping.get(prediction, "unKnown")
    result = prediction_disease
    return {result}

    #return result


@app.get("/")
def home():
    return {"message": "Welcome to the CBC Analysis API!"}
