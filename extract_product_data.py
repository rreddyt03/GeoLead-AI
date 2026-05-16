# PDF Product Data Extractor
# This script extracts product information (sizes, types, materials, etc.) from all PDFs in the PDF folder.
# It saves the extracted data to a JSON file for use by the chat agent.

import os
import json
import re
from pathlib import Path
import pdfplumber

PDF_DIR = Path(__file__).parent / "PDF"
OUTPUT_JSON = Path(__file__).parent / "product_data.json"

# Patterns to extract sizes (e.g., 5x4, 36" x 48", etc.)
SIZE_PATTERNS = [
    r"(\d{1,3})\s*[xX*]\s*(\d{1,3})",           # 5x4, 5 x 4, 5*4
    r"(\d{1,3})\s*by\s*(\d{1,3})",              # 5 by 4
    r"(\d{1,3})\s*feet?\s*by\s*(\d{1,3})\s*feet?", # 5 feet by 4 feet
    r"(\d{1,3})\"?\s*[xX]\s*(\d{1,3})\"?"  # 36" x 48"
]

# Extract text from all PDFs
def extract_text_from_pdfs(pdf_dir):
    all_text = ""
    for pdf_file in pdf_dir.glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() + "\n"
    return all_text

# Extract sizes from text
def extract_sizes(text):
    sizes = set()
    for pat in SIZE_PATTERNS:
        for match in re.findall(pat, text, re.IGNORECASE):
            sizes.add(f"{match[0]}x{match[1]}")
    return sorted(sizes)

# Extract product types (simple heuristic)
def extract_types(text):
    types = set()
    for line in text.splitlines():
        if any(word in line.lower() for word in ["window", "door"]):
            types.add(line.strip())
    return sorted(types)

# Main extraction
def main():
    text = extract_text_from_pdfs(PDF_DIR)
    sizes = extract_sizes(text)
    types = extract_types(text)
    data = {
        "sizes": sizes,
        "types": types
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Extracted {len(sizes)} sizes and {len(types)} types. Data saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
