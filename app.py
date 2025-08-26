import os
from openai import OpenAI
import streamlit as st

# Fetch key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)



# Streamlit UI
st.title("Healthcare Support Agent")
st.write("How can I help you today?")

# User input
user_query = st.text_input("Ask me anything about healthcare services:")

if user_query:
    # Query the AI model
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # lightweight model for fast responses
        messages=[
            {"role": "system", "content": "You are a helpful customer support agent for a healthcare organization. Answer politely and clearly."},
            {"role": "user", "content": user_query}
        ]
    )

    # Show response
    st.write("Agent:", response.choices[0].message["content"])
