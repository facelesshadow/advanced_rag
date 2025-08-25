# app.py
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from rag.rag import load_text_to_docs, create_retriever, build_rag_chain

# --- Session Memory Store ---
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("📚 RAG Chatbot with Memory")

api_key = st.text_input("Enter Google API Key:")

uploaded_text = st.text_area("Paste your document text here:")

if st.button("Process Document") and api_key and uploaded_text:
    with st.spinner("Processing..."):
        docs = load_text_to_docs(uploaded_text, api_key)
        retriever = create_retriever(docs, api_key)
        rag_chain = build_rag_chain(api_key, retriever)

        # Wrap with memory
        st.session_state.conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        st.success("Document processed! Start chatting below 👇")

# --- Chat Interface ---
# If you’re using RunnableWithMessageHistory, 
# it doesn’t automatically inject {chat_history} into your prompt. 
# You still need to include a placeholder in your prompt (like {chat_history}) if
#  you want the history to appear in the model’s input.

if "conversational_rag" in st.session_state:
    session_id = "user_session"

    # Display all past messages
    for msg in get_session_history(session_id).messages:
        role = "assistant" if msg.type == "ai" else "user"
        st.chat_message(role).write(msg.content)

    # Input box for next question
    user_input = st.chat_input("Ask me something...")
    if user_input:
        response = st.session_state.conversational_rag.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )["answer"]

        # Add new messages to history automatically via RunnableWithMessageHistory
        # Then replay will catch them on next rerun
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response)
