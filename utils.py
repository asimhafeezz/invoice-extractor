import json
import pdfplumber
import pytesseract
from PIL import Image
from langchain.schema import Document

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Function to extract text from image
def extract_text_from_image(file):
    img = Image.open(file)
    return pytesseract.image_to_string(img)

# Function to load examples from .jsonl file
def load_examples_from_file(filepath):
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(Document(page_content=json.dumps(data)))
    return examples