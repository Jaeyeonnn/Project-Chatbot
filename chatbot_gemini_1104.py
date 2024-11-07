import os
import pandas as pd
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as gemini
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import google.generativeai as gemini

gemini.configure(api_key='your_api_key')
model = gemini.GenerativeModel(
    model_name='gemini-pro')

# Streamlit app title
st.title('BLCK UNICRN Chatbot')
input_text = st.text_input("Please enter your question:")

# Load CSV data
csv_path = "/users/jl/desktop/chatbot/linkedindatatest.csv"
df = pd.read_csv(csv_path)
df.fillna('NA', inplace=True)

GOOGLE_API_KEY = 'your_api_key'
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Convert data into embeddings and store it in a vector store
data_texts = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
print(f'len text={len(data_texts)}')
vector_store = FAISS.from_texts(data_texts, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Retrieval setup
retriever = vector_store.as_retriever()

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the data provided to answer questions accurately."),
        ("user", "Question: {question}")
    ]
)

# Conversational Chain with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create conversational chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=GoogleGenerativeAI(model="gemini-1.5-flash"),  # Use the same or another model for conversation
    retriever=retriever,
    memory=memory  # Pass the memory to the chain
)

# Execute the chain if there is input text
if input_text:
    print(f'input_text={input_text}')
    response = conversation_chain({"question": input_text})
    st.write(response)
