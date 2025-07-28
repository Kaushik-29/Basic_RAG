import os
import streamlit as st
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
import faiss

def process_input(input_type, input_data):
    """Process input data based on type and return a FAISS vectorstore."""
    try:
        if input_type == "Link":
            documents = []
            for link in input_data:
                loader = WebBaseLoader(link)
                documents.extend(loader.load())
        elif input_type == "PDF":
            if input_data is None:
                st.error("No PDF file uploaded.")
                return None
            pdf_reader = PdfReader(BytesIO(input_data.read()))
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            documents = text
        elif input_type == "DOCX":
            if input_data is None:
                st.error("No DOCX file uploaded.")
                return None
            doc = Document(BytesIO(input_data.read()))
            documents = "\n".join([para.text for para in doc.paragraphs])
        elif input_type == "TXT":
            if input_data is None:
                st.error("No TXT file uploaded.")
                return None
            documents = input_data.read().decode("utf-8")
        elif input_type == "Text":
            if not input_data:
                st.error("No text provided.")
                return None
            documents = input_data
        else:
            st.error(f"Unsupported input type: {input_type}")
            return None
    except Exception as e:
        st.error(f"Error processing input: {e}")
        return None

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    if input_type == "Link":
        texts = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in texts]
    else:
        texts = text_splitter.split_text(documents)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)
    vectorstore = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vectorstore.add_texts(texts)
    return vectorstore

def answer_question(vectorstore, query, api_key):
    """Answer a question using the provided vectorstore and Gemini LLM."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=api_key
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )
        result = qa({"query": query})
        return result
    except Exception as e:
        st.error(f"Error answering question: {e}")
        return None

def main():
    st.set_page_config(page_title="RAG Q&A App with Gemini", layout="wide")
    st.title("RAG Q&A App with Gemini")
    st.markdown("""
    This app lets you upload documents or provide links/text, then ask questions using Google's Gemini model with Retrieval-Augmented Generation (RAG).\
    **Instructions:**\
    1. Enter your Gemini API key in the sidebar.\
    2. Choose your input type and provide/upload your data.\
    3. Click 'Process Document' to build the knowledge base.\
    4. Ask questions about your data!
    """)
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    if not api_key:
        st.sidebar.warning("Please enter your Gemini API key.")
    if st.sidebar.button("Clear Vectorstore"):
        st.session_state.pop("vectorstore", None)
        st.success("Vectorstore cleared.")
    input_type = st.selectbox("Choose Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
    input_data = None
    if input_type == "Link":
        link_count = st.number_input("Number of Links", min_value=1, max_value=5, step=1)
        input_data = []
        for i in range(int(link_count)):
            url = st.text_input(f"URL {i+1}", key=f"url_{i}")
            if url:
                input_data.append(url)
        if not input_data:
            input_data = None
    elif input_type == "Text":
        input_data = st.text_area("Enter your text")
    else:
        input_data = st.file_uploader(f"Upload a {input_type} file", type=input_type.lower())
    if st.button(" Process Document"):
        if not api_key:
            st.error("Please provide a valid Gemini API key in the sidebar.")
        else:
            with st.spinner("Processing input and building vectorstore..."):
                vectorstore = process_input(input_type, input_data)
                if vectorstore:
                    st.session_state["vectorstore"] = vectorstore
                    st.success(" Vectorstore created!")
    if "vectorstore" in st.session_state:
        query = st.text_input(" Ask a Question")
        if st.button(" Submit"):
            if not query:
                st.warning("Please enter a question.")
            elif not api_key:
                st.error("Please provide a valid Gemini API key in the sidebar.")
            else:
                with st.spinner("Thinking..."):
                    result = answer_question(st.session_state["vectorstore"], query, api_key)
                    if result:
                        st.markdown("**Answer:**")
                        st.write(result["result"])
                        if result.get("source_documents"):
                            st.markdown("**Sources:**")
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.markdown(f"**Source {i}:**\n{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")

if __name__ == "__main__":
    main()
