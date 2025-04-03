import streamlit as st
import os
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import gdown

# Google Drive Folder ID
folder_id = "1ykKlRQH7wXBl9P1YHAOVUfcfVs0PpNRs"  # Your folder ID

# Define the local folder path where the vector index will be stored
local_folder = "./vector_index_2"
# Download the folder from Google Drive if it doesn't exist locally
if not os.path.exists(local_folder):
    os.makedirs(local_folder, exist_ok=True)
    gdown.download_folder(id=folder_id, output=local_folder, quiet=False
# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(model_name=hf_model, task="text-generation", token=token)

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "./embeddings_cache"

embeddings = HuggingFaceEmbedding(model_name=embedding_model, cache_folder=embeddings_folder)

# Load Vector Database
storage_context = StorageContext.from_defaults(persist_dir="./vector_index_2")
vector_index_2 = load_index_from_storage(storage_context, embed_model=embeddings)

# Retriever
retriever = vector_index_2.as_retriever(similarity_top_k=2)

# Prompts
prompts = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful chatbot having a conversation with a human."),
    ChatMessage(role=MessageRole.SYSTEM, content="Answer the question based only on the provided context and previous conversation."),
    ChatMessage(role=MessageRole.SYSTEM, content="Keep your answers short and to the point."),
]

# Memory
memory = ChatMemoryBuffer.from_defaults()

# Initialize bot with memory
@st.cache_resource
def init_bot():
    return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)

rag_bot = init_bot()

##### Streamlit UI #####

st.title("Chatier & Chatier")

# Display chat history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

# Handle user input
if prompt := st.chat_input("Curious minds wanted!"):
    st.chat_message("human").markdown(prompt)

    with st.spinner("Going down the rabbit hole for answers..."):
        answer = rag_bot.chat(prompt)
        response = answer.response

        with st.chat_message("assistant"):
            st.markdown(response)
