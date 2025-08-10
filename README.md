# Abdominal Pain in Adults – Medical Chatbot

This project is an **AI-powered medical chatbot** that answers questions **strictly related to abdominal pain in adults**.  
It uses a dataset of medical articles, processes them into vector embeddings, and performs **retrieval-augmented generation (RAG)** to provide accurate, dataset-grounded answers.

---

## Features
- **Abdominal Pain–Focused Knowledge**  
  Only answers medical queries directly related to abdominal pain in adults.
- **Polite Small Talk**  
  Responds naturally to greetings or general conversation.
- **Professional Disclaimers**  
  If asked about other medical issues, politely states that knowledge is limited.
- **Vector Search with FAISS**  
  Uses a FAISS vector store for fast Similarity search.
- **Interactive Chatbot UI**  
  Built with Streamlit for a conversational experience.
- **FastAPI Backend**  
  Handles LLM requests and connects to the vector store.

---

## Tech Stack
- **LangChain** – For RAG pipeline and tool integration
- **HuggingFace Embeddings** – To generate vector embeddings
- **FAISS** – Vector search storage
- **FastAPI** – Backend API
- **Streamlit** – Frontend chatbot UI
- **Google Generative AI (Gemini)** – LLM for answering questions

---

## Project Structure
    .
    ├── medical_articles.json      # Dataset of abdominal pain articles
    ├── vectorstore/               # FAISS index folder
    ├── app.py                     # FastAPI backend
    ├── chatbot.py                 # Streamlit chatbot interface
    ├── requirements.txt           # Python dependencies
    └── README.md                  # Project documentation

---
## Prepare Dataset & Vector Store

- Make sure your medical_articles.json file is in the project root.
- Run your FAISS creation script to index the dataset.

---
## Start Backend
    uvicorn app:app --reload

---
## Start Chatbot UI
    streamlit run client.py


