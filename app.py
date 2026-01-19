import streamlit as st
import uuid
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

# --- FUNCTION TO CLEAR DATA ---
def reset_context():
    # This deletes the previous vector store from memory
    if "vector_store" in st.session_state:
        del st.session_state.vector_store
    if "user_question" in st.session_state:
        st.session_state.user_question = ""

# --- INPUT WITH CALLBACK ---
# When you paste a new URL, 'reset_context' runs immediately to clear old data
website_url = st.text_input(
    "Enter Website URL", 
    key="url_input", 
    on_change=reset_context
)

if website_url and groq_api_key:
    if "vector_store" not in st.session_state:
        with st.spinner("Processing new website..."):
            try:
                # 1. Load Data
                loader = WebBaseLoader(website_url)
                docs = loader.load()
                
                # 2. Split Data
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # 3. Create Embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # 4. Create Vector Store with a UNIQUE COLLECTION NAME
                # This ensures the new data never mixes with the old data
                unique_id = f"collection_{uuid.uuid4()}" 
                
                vector_store = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings, 
                    collection_name=unique_id  # <--- CRITICAL FIX
                )
                
                st.session_state.vector_store = vector_store
                st.success("Website processed successfully!")
            except Exception as e:
                st.error(f"Error loading website: {e}")

    # Ask Question
    question = st.text_input("Ask a question about the website", key="user_question")

    if question and "vector_store" in st.session_state:
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile"
        )

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context. 
        If the answer is not in the context, say "I cannot find the answer on this specific website."

        <context>
        {context}
        </context>

        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Thinking..."):
            try:
                response = retrieval_chain.invoke({"input": question})
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Error generating answer: {e}")

