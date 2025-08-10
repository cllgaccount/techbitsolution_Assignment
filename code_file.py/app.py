from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Sequence, Annotated, Dict, List
from dotenv import load_dotenv, find_dotenv
import os
import json

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader, PyPDFLoader, WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64)"


load_dotenv(find_dotenv())

# LLm Model Define (That's a Google free LLm Model in case Free Quota is reach the limit we can use the Groq model, just comment the model.)
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",
#     temperature=1
# )

# LLM Model Define (That's a free Groq model in case Quota is reach the limit we can use the Google model, just comment the model.)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1,
    max_tokens=100,
    # api_key=groq_api_key
)

# Embedding Model Define
embeddings = HuggingFaceEmbeddings(
    model = "BAAI/bge-small-en-v1.5",
)

with open("medical_articles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [Document(page_content=item["content"]) for item in data if "content" in item]


# Step 3: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
def clean_text(text: str) -> str:
    # Remove extra empty lines and excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


docs = text_splitter.split_documents(documents)
cleaned_docs = [
    doc.__class__(page_content=clean_text(doc.page_content), metadata=doc.metadata)
    for doc in docs
]


# Preview one chunk
# print(docs[1].page_content[:500])


faiss_index_path = "C:/Users/aman singh/Desktop/Assignment_techbitsolution/vectorstore"

# FAISS creation
try:
    vectorstore = FAISS.from_documents(
        documents=cleaned_docs,  # same as your PDF or web chunks
        embedding=embeddings
    )

    print("FAISS vector store created successfully")

    # Save index to disk
    vectorstore.save_local(faiss_index_path)

except Exception as e:
    print(f"Error creating FAISS store: {str(e)}")
    raise


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Defineing the tool
@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from resource as per requirements."""

    docs1 = retriever.invoke(query)

    if not docs1:
        return "I found no relevant information."
    results = []
    for i, doc in enumerate(docs1):
        content = clean_text(doc.page_content)
        preview = content[:1000] + "..." if len(content) > 1000 else content
        results.append(f"Document {i+1}:\n{preview}")
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

def should_continue(state: AgentState):
    """Check if the message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls) > 0

# Prompt Defining
system_prompt = """
You are a professional, clear, and empathetic AI assistant whose expertise for this session is limited to a curated dataset about abdominal pain in adults.

Tone & style:
- Be concise, human, and professional. Use plain, empathetic language (e.g., "I understand your concern") when appropriate.
- Do not use emojis or casual slang.
- Do NOT start answers with mechanical phrases like "Based on the provided medical article" or "According to the dataset." Instead, integrate source information naturally when needed.

Rules of operation:
1. Medical questions about abdominal pain in adults (causes, symptoms, tests, typical treatments described in the dataset, prevention, red flags): 
   - ALWAYS call the `retriever_tool` to fetch relevant passages before composing your answer.
   - Answer using only the retrieved content. Do not invent or infer additional facts.
   - Present answers in a clear structure: a 1–2 sentence plain-language summary, then short bullet points or numbered steps for details (if helpful).
   - When you use information from a retrieved article, include a short natural citation (article title and URL if available), e.g. "Source: Gastritis — Mayo Clinic."
   - Do NOT provide medical diagnoses. For anything that appears urgent or life-threatening, include a short safe-referral line: "If this is an emergency, seek immediate medical care."

2. General conversation / greetings (e.g., "Hello", "How are you?", "Thanks"):
   - Respond naturally and politely without using the dataset. Keep replies brief and friendly.

3. Medical questions outside the abdominal-pain-in-adults scope:
   - Reply professionally and briefly with: 
     "I’m sorry — I don’t have information about that topic in my dataset. My scope here is limited to abdominal pain in adults. If you’d like, I can help with questions related to abdominal pain or point you to general resources."
   - Do not attempt to answer such questions using outside knowledge.

Safety and clarity:
- If the retriever returns no useful content, say: "I could not find relevant information in my dataset about that. Please rephrase or ask about abdominal pain in adults."
- Avoid over-citation. Use one concise citation per main claim when possible.
- Be humble about uncertainty and explicit about limitations.

Response format example for in-scope question:
- Short summary sentence.
- Bullet list of key points (symptoms, causes, red flags, next steps).
- Citation line: "Source: <Article Title> — <URL (if available)>"

Your goal: be human, helpful, and strictly grounded in the provided dataset while allowing normal polite conversation.
"""



# Creating a dictionay of out tools
tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLm Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    messages = llm.invoke(messages)
    return {"messages":[messages]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response"""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query','No query provide')}")

        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result length: {len(str(result))}")

        # Append the tool meassage
        results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
    
    print("Tool Execution Complete. Back to the model!")
    return {"messages": results}


# Defining the Graph
graph = StateGraph(AgentState)

# Adding Nodes
graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)

# Creating Conditional Edge
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:"retriever_agent",False:END
    }
)

# Creating edge
graph.add_edge("retriever_agent","llm")

# Defining entry point
graph.set_entry_point("llm")


# Graph compile
rag_agent = graph.compile()


# Fastapi
app = FastAPI(title="Medical RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        messages = [HumanMessage(content=request.question)]
        result = rag_agent.invoke({"messages": messages})
        answer = result['messages'][-1].content
        return QueryResponse(answer=answer)
    except Exception as e:
        return QueryResponse(answer=f"Error: {str(e)}")

# This function is only for printing
# def running_agent():
#     print("\n ==== RAG ====")

#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in ['exit','quit']:
#             break
        
#         messages = [HumanMessage(content=user_input)] # Convert back to a HumanMessage type

#         result = rag_agent.invoke({"messages": messages})

#         print("\n ==== ANSWER ====")
#         print(result['messages'][-1].content)

# running_agent()

# print(retriever_tool.invoke("What are common causes of abdominal pain in adults?"))
