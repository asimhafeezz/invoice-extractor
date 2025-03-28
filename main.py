import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
import json

from utils import extract_text_from_pdf, extract_text_from_image, load_examples_from_file

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load or initialize the vector store
VECTOR_DB_PATH = "faiss_index"
embedding = OpenAIEmbeddings()

if os.path.exists(VECTOR_DB_PATH):
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)
else:
    example_docs = load_examples_from_file("examples.jsonl")
    vectorstore = FAISS.from_documents(example_docs, embedding)
    vectorstore.save_local(VECTOR_DB_PATH)

retriever = vectorstore.as_retriever()

# Prompt for extraction
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI trained to extract structured invoice data.
Use the examples below as guidance.

Examples:
{context}

---

Now extract the following fields from the invoice:
- invoice_number
- date
- vendor
- total

Invoice text:
{question}

Return the data in JSON format.
"""
)

# Setup LLM QA Chain
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# Streamlit UI
st.set_page_config(page_title="Invoice Data Extractor")
st.title("Invoice Data Extractor")

uploaded_file = st.file_uploader("Upload an Invoice (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.success("File uploaded successfully.")

    if uploaded_file.name.endswith(".pdf"):
        invoice_text = extract_text_from_pdf(uploaded_file)
    else:
        invoice_text = extract_text_from_image(uploaded_file)

    st.subheader("Extracted Text using OCR")
    st.text_area("Invoice Text", invoice_text, height=300)

    if st.button("Extract Invoice Data"):
        result = qa_chain.invoke({"query": invoice_text})

        st.subheader("Extracted Fields")
        st.json(result["result"])

        with st.expander("Retrieved Example Prompts Used"):
            for doc in result["source_documents"]:
                st.markdown(f"- {doc.page_content}")

       # Save the extracted JSON result to the vector store
        extracted_json_str = json.dumps(result["result"])
        new_doc = Document(page_content=extracted_json_str)
        vectorstore.add_documents([new_doc])
        vectorstore.save_local(VECTOR_DB_PATH)

        st.success("Extracted text has been added to the vector database for future similarity searches.")