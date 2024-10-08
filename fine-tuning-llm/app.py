import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def main():
    st.title("Document Question Answering with RAG and LLMs")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        # Read and process the PDF
        text = read_pdf(uploaded_file)
        st.write("Document Loaded Successfully!")

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        st.write(f"Document split into {len(chunks)} chunks.")

        # Create embeddings
        st.write("Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        # Build vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.write("Embeddings and vector store created.")

        # Load LLM model
        st.write("Loading language model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = 'gpt2'  # You can choose a different model compatible with your hardware
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        generation_pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=0 if device=='cuda' else -1,
            max_length=1024  # Be cautious with this value as it can consume more memory
        )

        llm = HuggingFacePipeline(pipeline=generation_pipeline)
        st.write("Language model loaded.")

        # Build RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # User input for querying
        query = st.text_input("Enter your question about the document:")
        if query:
            st.write("Generating answer...")
            answer = qa.run(query)
            st.write("**Answer:**")
            st.write(answer)

if __name__ == '__main__':
    main()
