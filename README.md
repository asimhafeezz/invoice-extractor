# Invoice Extractor (RAG + OpenAI + Streamlit)

This is a simple tool to extract structured invoice data (like invoice number, date, vendor, total) from PDF and image invoices using OCR, LangChain, and OpenAI GPT-4.

It also stores extracted data in a vector database (FAISS) to enable future similarity-based retrieval.

## Demo

Watch the demo video here:  
[invoice_extractor_demo](https://github.com/user-attachments/assets/e0aeb6c8-4b24-40bf-8255-5857e5e05188)


## Features

- Upload PDF or image invoices
- Extracts fields like invoice number, date, vendor, total
- Uses LangChain + OpenAI for field extraction
- Stores extracted results in FAISS vector store
- Clean UI built with Streamlit

## How to Run

1. Clone this repo

```bash
git clone https://github.com/your-username/invoice-extractor.git
cd invoice-extractor
```

2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies

```bash
pip install -r requirements.txt
```

4. Add your OpenAI API key
Create a .env file in the root folder and add your key:
```bash
OPENAI_API_KEY=your-openai-api-key
```

5. Start the Streamlit app

```bash
.venv/bin/streamlit run main.py
```
or
```bash
streamlit run main.py
```
