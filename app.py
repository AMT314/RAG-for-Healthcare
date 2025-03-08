import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings

# Suppress LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up the Groq client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Set up the language model
llm = ChatGroq(model_name="llama3-70b-8192")

# Set up the PDF loader
loader = PyPDFLoader('health.pdf') 
data = loader.load()

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text = text_splitter.split_documents(data)

# Set up the embeddings
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Set up the vector store
db = FAISS.from_documents(text, embeddings)

# Set up the retriever
retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 4})

# Set up the prompt template
prompt_template = """
You are a helpful assistant who has the ability to generate the answers only from the provided context.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer to the question only in single line.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Set up the retrieval QA chain
qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type='stuff',
                                retriever=retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": prompt})

# Set up the Streamlit app
st.title("RAG Healthcare Chatbot")

# Get the user's question
question = st.text_input("What do you want to know about healthcare?")

# If the user has entered a question, generate and display the response
if question:
    result = qa(question)
    st.write(result['result'])