import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

st.title("😺Ask About Website")
groq_api_key = "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"
website_url =st.text_input("Ask a question about the website")
if website_url and groq_api_key:
    # 3. Load the data from the website
    if "vector_store" not in st.session_state:
        with st.spinner("Processing....."):
            try:
                # Load the raw text
                loader = WebBaseLoader(website_url)
                docs = loader.load()
                
                # Split the text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Create embeddings (Using HuggingFace so it runs locally)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
                
                st.session_state.vector_store = vector_store
                st.success("Process done!....")
            except Exception as e:
                st.error(f"Error loading e: {e}")

    
    question = st.text_input("Ask a question about the website")

    if question and "vector_store" in st.session_state:
        # 5. Setup the AI Chain
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile"
        )

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}
        """)

        # Using the Classic functions imported above
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # 6. Get the Answer
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": question})
            st.write(response["answer"])