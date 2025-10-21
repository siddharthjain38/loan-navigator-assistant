"""
Streamlit chat interface for loan policy assistant with conversation memory.
"""
import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Loan Policy Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Sidebar with controls
with st.sidebar:
    st.header("💬 Conversation")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        # Clear backend conversation
        if st.session_state.session_id:
            try:
                requests.delete(f"http://localhost:8000/chat/history/{st.session_state.session_id}")
            except:
                pass
        
        # Clear local state
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### 📋 What I can help with:
    - **Current loan details** - Check EMI, balance, status
    - **EMI calculations** - Calculate new loan EMIs
    - **Loan policies** - Eligibility, documentation, rules
    - **What-if scenarios** - Compare different tenures
    
    ### 💡 Example queries:
    - "Current EMI for customer 1900"
    - "Calculate EMI for 5 lakh loan"
    - "What is the eligibility criteria?"
    """)
    
    if st.session_state.session_id:
        st.info(f"Session: {st.session_state.session_id[:8]}...")

# Main chat UI
st.title("💰 Loan Navigator Assistant")
st.caption("Your intelligent loan assistant powered by AI")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about loans..."):
    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show typing indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send request to backend with session ID
                response = requests.post(
                    "http://localhost:8000/chat/",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Update session ID from backend
                    st.session_state.session_id = response_data["session_id"]
                    
                    # Display assistant response
                    st.markdown(response_data["answer"])
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data["answer"]
                    })
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                
            except requests.exceptions.ConnectionError:
                error_msg = "❌ Cannot connect to backend. Make sure the API is running on http://localhost:8000"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })