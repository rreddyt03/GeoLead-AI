
# PDF Analyzer and Knowledge Extractor
# This script will extract and summarize product knowledge from all PDFs in the PDF folder.
# It will output a summary and a structured JSON knowledge base for RAG.

import os
import json
from pathlib import Path
import pdfplumber

PDF_DIR = Path(__file__).parent / "PDF"
OUTPUT_JSON = Path(__file__).parent / "product_knowledge.json"

# Extract all text from all PDFs
def extract_text_from_pdfs(pdf_dir):
    all_text = ""
    for pdf_file in pdf_dir.glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += f"\n--- {pdf_file.name} PAGE {page.page_number} ---\n" + page_text + "\n"
    return all_text

# Simple heuristic to extract product series, types, sizes, and features
def analyze_text(text):
    import re
    series = set()
    types = set()
    sizes = set()
    materials = set()
    features = set()
    for line in text.splitlines():
        l = line.lower()
        # Series
        for s in ["100 series", "200 series", "400 series", "a-series", "e-series", "coastal", "big doors"]:
            if s in l:
                series.add(s)
        # Types
        for t in ["window", "door", "awning", "casement", "sliding", "bay", "bow", "picture", "double-hung", "single-hung"]:
            if t in l:
                types.add(t)
        # Sizes
        for m in re.findall(r'(\d{2,3})["′]?\s*[xX]\s*(\d{2,3})["′]?', l):
            sizes.add(f"{m[0]}x{m[1]}")
        # Materials
        for mat in ["vinyl", "wood", "aluminum", "fiberglass", "composite"]:
            if mat in l:
                materials.add(mat)
        # Features
        for f in ["energy efficient", "warranty", "coastal", "impact", "custom", "modern", "classic", "grilles", "hardware", "glass"]:
            if f in l:
                features.add(f)
    return {
        "series": sorted(series),
        "types": sorted(types),
        "sizes": sorted(sizes),
        "materials": sorted(materials),
        "features": sorted(features)
    }

def main():
    print("Extracting and analyzing PDF product guides...")
    text = extract_text_from_pdfs(PDF_DIR)
    knowledge = analyze_text(text)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(knowledge, f, indent=2)
    print("\nSUMMARY OF PRODUCT KNOWLEDGE:")
    for k, v in knowledge.items():
        print(f"{k.title()}: {', '.join(v) if v else 'None found'}")
    print(f"\nStructured knowledge base saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
