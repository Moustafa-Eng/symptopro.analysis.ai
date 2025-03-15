from fastapi import FastAPI, File, UploadFile
import pytesseract
from PIL import Image
import re
from pdf2image import convert_from_bytes
from io import BytesIO

app = FastAPI()


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


# Function to parse CBC results
def parse_cbc_results(text):
    target_parameters = {
        "WBC": ["White Blood Cells", "WBC", "Leukocytes"],
        "HB": ["Hemoglobin", "HB", "HGB"],
        "MCV": ["Mean Corpuscular Volume", "MCV"],
        "PC": ["Platelet Count", "Platelets", "PC"],
    }

    lines = text.splitlines()
    results = {}

    for line in lines:
        if re.search(r"\d", line):
            for param, keywords in target_parameters.items():
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    match = re.search(r"([\d.]+)\s+[^\d]*([\d.]+)\s*-\s*([\d.]+)", line)
                    if match:
                        results[param] = {
                            "Result": float(match.group(1)),
                            "Reference Range": (
                                float(match.group(2)),
                                float(match.group(3)),
                            ),
                        }
                    break

    return results


# Function to compare values
def compare_value(value, low, high):
    if value < low:
        return "LOW"
    elif value > high:
        return "HIGH"
    return "NORM"


# Function to analyze CBC
def analyze_cbc(results):
    if not results:
        return {"error": "No valid CBC results found"}

    HB = results.get("HB", {}).get("Result")
    WBC = results.get("WBC", {}).get("Result")
    PC = results.get("PC", {}).get("Result")
    MCV = results.get("MCV", {}).get("Result")

    HB_range = results.get("HB", {}).get("Reference Range", (None, None))
    WBC_range = results.get("WBC", {}).get("Reference Range", (None, None))
    PC_range = results.get("PC", {}).get("Reference Range", (None, None))
    MCV_range = results.get("MCV", {}).get("Reference Range", (None, None))

    def get_status(value, ref_range):
        if value is None or ref_range == (None, None):
            return "UNKNOWN"
        return compare_value(value, *ref_range)

    hb_status = get_status(HB, HB_range)
    wbc_status = get_status(WBC, WBC_range)
    pc_status = get_status(PC, PC_range)
    mcv_status = get_status(MCV, MCV_range)

    analysis = []

    if hb_status == "LOW":
        if mcv_status == "LOW":
            analysis.append("Microcytic Hypochromic (Iron deficiency anemia)")
        elif mcv_status == "HIGH":
            analysis.append("Macrocytic Hyperchromic (Megaloblastic anemia)")
        else:
            analysis.append(
                "Normocytic Normochromic (Aplastic anemia or Acute bleeding)"
            )

    if hb_status == "NORM":
        if mcv_status == "LOW":
            analysis.append("Thalassemia Trait or Early Iron Deficiency")
        elif mcv_status == "HIGH":
            analysis.append("Vitamin B12 or Folate Deficiency")

    if hb_status == "HIGH":
        analysis.append("Polycythemia")

    if wbc_status == "LOW":
        analysis.append("Immunodeficiency")
    elif wbc_status == "HIGH":
        analysis.append("Possible infection")

    if pc_status == "LOW":
        analysis.append("Thrombocytopenia (low platelets)")
    elif pc_status == "HIGH":
        analysis.append("Thrombocytosis (high platelets)")

    return {
        "Results": results,
        "Analysis": analysis if analysis else ["All values are within normal range"],
    }


# API endpoint to handle file uploads
@app.post("/analyze_cbc")
async def upload_file(file: UploadFile = File(...)):
    file_content = await file.read()
    extracted_text = extract_text_from_file(file_content, file.filename)
    cleaned_text = preprocess_text(extracted_text)
    cbc_results = parse_cbc_results(cleaned_text)
    analysis = analyze_cbc(cbc_results)

    return {"CBC Results": cbc_results, "Analysis": analysis}