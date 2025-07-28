# Basic_RAG
# 🔍 RAG Q&A App with Gemini

This is a **Retrieval-Augmented Generation (RAG)** based Streamlit app that allows users to upload or link documents (PDF, DOCX, TXT, or plain text) and ask questions about the content using **Google's Gemini (1.5 Flash)** via the LangChain framework.

## 🚀 Features

- 📄 Accepts inputs from: 
  - URLs (web scraping)
  - PDF documents
  - DOCX files
  - TXT files
  - Raw text
- 🧠 Uses Sentence Transformers for embedding text.
- 🔍 Powered by FAISS for fast vector search.
- 💬 Queries answered by Google's Gemini using LangChain's `RetrievalQA` chain.
- 🖥️ Interactive, clean Streamlit UI.

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini 1.5 Flash
- **Embedding Model:** all-mpnet-base-v2 (Sentence Transformers)
- **Vector Store:** FAISS
- **Frameworks/Libraries:** LangChain, HuggingFace, PyPDF2, python-docx

---

