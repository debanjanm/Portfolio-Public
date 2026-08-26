import streamlit as st
from pipeline.flow import MentalChatbot

st.title("Therapist")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = MentalChatbot()

with st.chat_message("user"):
    user_prompt = st.text_input("Say Hello to start your conversation")

if st.button("Response") and user_prompt:
    with st.spinner("I'm thinking..."):
        output = st.session_state.pipeline.executing(user_prompt, "streamlit-user", "streamlit-session")
        with st.chat_message("assistant"):
            st.write(output)
