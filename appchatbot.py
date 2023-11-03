
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

#function to load documents

def load_documents():
    loader = DirectoryLoader('data/', glob = "*.pdf", loader_cls= PyPDFLoader)
    documents = loader.load()
    return documents

#Function to split text into chunks

def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 ,chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

#functions to create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device' :"cpu"})
    return embeddings

#Function to create vector store
def create_vector_store(text_chunks, embeddings):
    vectore_store = FAISS.from_documents(text_chunks, embeddings)
    return vectore_store

custom_template = """
[INST] You are a helpful chat assistant. 
1. Your name is D and your owner is ABC company. 
2. You have Toasmasters documents. If the question {query} is unrealted to the documents, give the following answer. 
3. Dont genereate answers for following areas{areas}. Return answer 'I don't know'.
query related with : Mathematics/ Science/ any question unrealted with Toasmasters.
Answer: I don't the answer.
[/INST]
"""

#Function to create LLMs model

def create_llm_model():
    llm = CTransformers(model = "mistral-7b-instruct-v0.1.Q4_K_M.gguf" ,
                        config = {'max_new_tokens' :512,
                                  'temperature' : 0.01} ,
                                  streaming = True, 
                                  PromptTemplate = custom_template)
    return llm

#loading of documents
documents = load_documents()

#Split text into chunks
text_chunks = split_text_into_chunks(documents)

#Create embeddings
embeddings = create_embeddings()

#create vectore store
vector_store = create_vector_store(text_chunks,embeddings)

#create LLMs model
llm = create_llm_model()

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Create chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory,)
#print(chain)
#query = "Who is the president of the toasmasters? "
#query = "What is the club mission? "
#query = "Who is the TM Sajan Pushpanathan ?"
#query = "What is the meeting date ? "
query = "What is the calcullus definition ? "
result = chain({"question": query})

print(result["answer"])