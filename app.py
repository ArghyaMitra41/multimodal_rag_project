import streamlit as st
from rag import process_files, query_rag

st.set_page_config(page_title="Multimodal RAG Assistant")

st.title("📄🖼️ Smart PDF + Image Research Assistant")

uploaded_files = st.file_uploader("Upload PDFs / Images / Text", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing files..."):
        process_files(uploaded_files)
    st.success("Files processed successfully!")

query = st.text_input("Ask a question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        answer = query_rag(query)
    st.write("### Answer")
    st.write(answer)