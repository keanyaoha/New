
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
from google.colab import userdata

# download saved vector database for Alice's Adventures in Wonderland
gdown --folder 1ykKlRQH7wXBl9P1YHAOVUfcfVs0PpNRs
token = userdata.get('Otu_ocha')

hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(model_name = hf_model, task = "text-generation", token = token)

embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/"

embeddings = HuggingFaceEmbedding(model_name=embedding_model,
                                  cache_folder=embeddings_folder)

storage_context = StorageContext.from_defaults(persist_dir="/content/vector_index_2")
vector_index_2 = load_index_from_storage(storage_context, embed_model=embeddings)
retriever = vector_index_2.as_retriever(similarity_top_k=2)

prompts = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a nice chatbot having a conversation with a human."),
    ChatMessage(role=MessageRole.SYSTEM, content="Answer the question based only on the following context and previous conversation."),
    ChatMessage(role=MessageRole.SYSTEM, content="Keep your answers short and succinct."),
    ChatMessage(role=MessageRole.SYSTEM, content="Dont respond to user and assitant.")
]

memory = ChatMemoryBuffer.from_defaults()

rag_bot = ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)


ans = rag_bot.chat("Do you know the context?")

print(ans.response)
ans = rag_bot.chat("What are the sections of this article?")
print(ans.response)

bot_2 = ContextChatEngine(llm=llm,
                          retriever=retriever,
                          memory=ChatMemoryBuffer.from_defaults(),
                          prefix_messages=prompts
)

# Start the conversation loop
while True:
  user_input = input("You: ")

  # Check for exit condition
  if user_input.lower() == 'end':
      print("Ending the conversation. Goodbye!")
      break

  # Get the response from the conversation chain
  response = bot_2.chat(user_input)
  # Print the chatbot's response
  print(response.response)

