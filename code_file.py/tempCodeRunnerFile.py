splitter = RecursiveCharacterTextSplitter(chunks_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)


embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

)

vectordb = FAISS.from_documents(chunks,embedding)
vectordb.save_local("vectorstore")


vectordb = FAISS.load_local("vectorstore",embedding)
qa_chain = RetrievalQA.from_chain_type(
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=1
    ),
    retriever = vectordb.as_retriever()
)

response = qa_chain.run("What are the common causes of abdominal pain?")
print(response)
