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

# ------------------------------------------------------------------
# SECURITY WARNING: Never commit API keys to GitHub or share them publicly.
# ------------------------------------------------------------------
groq_api_key = "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"

# --- STEP 1: DEFINE THE RESET FUNCTION ---
# This runs only when the user changes the URL
def reset_context():
    if "vector_store" in st.session_state:
        del st.session_state.vector_store
    if "user_question" in st.session_state:
        st.session_state.user_question = ""

# --- STEP 2: INPUT WITH CALLBACK ---
# The on_change parameter connects the input to the reset function
website_url = st.text_input(
    "Enter Website URL", 
    key="url_input", 
    on_change=reset_context
)

if website_url and groq_api_key:
    # 3. Load the data from the website
    # This check ensures we don't re-process if the data is already there
    if "vector_store" not in st.session_state:
        with st.spinner("Processing new website..."):
            try:
                # Load the raw text
                loader = WebBaseLoader(website_url)
                docs = loader.load()
                
                # Split the text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Create embeddings (Using HuggingFace so it runs locally)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Store in vector db
                vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
                
                st.session_state.vector_store = vector_store
                st.success("Website processed successfully!")
            except Exception as e:
                st.error(f"Error loading website: {e}")

    # 4. Ask the Question
    question = st.text_input("Ask a question about the website", key="user_question")

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

        # Using standard langchain chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # 6. Get the Answer
        with st.spinner("Thinking..."):
            try:
                response = retrieval_chain.invoke({"input": question})
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Error generating answer: {e}")
