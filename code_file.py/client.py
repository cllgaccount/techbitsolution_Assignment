import requests
import streamlit as st

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/ask"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # Each item will be {"role": "user"/"assistant", "content": "..."}

# Function to query FastAPI backend
def get_google_response(input_text):
    try:
        response = requests.post(API_URL, json={"question": input_text})
        response.raise_for_status()
        return response.json().get('answer', "No answer field in response.")
    except Exception as e:
        return f"Error: {str(e)}"

# App title
st.title("Medical Chatbot")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your medical question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot's answer from FastAPI
    answer = get_google_response(prompt)

    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(answer)
