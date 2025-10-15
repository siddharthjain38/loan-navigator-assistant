"""
Streamlit chat interface for loan policy assistant.
"""
import streamlit as st
import requests
import json

# Page config
st.set_page_config(page_title="Loan Policy Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
st.title("Loan Policy Assistant")

# Chat message container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about loan policies"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        # Send request to backend
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": prompt}
        )
        response_data = response.json()

        # Add assistant response to chat
        with st.chat_message("assistant"):
            st.write(response_data["answer"])

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["answer"]
        })

    except Exception as e:
        st.error(f"Error: {str(e)}")