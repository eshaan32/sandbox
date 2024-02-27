# import os

# def read_directory(directory_path):
#     for filename in os.listdir(directory_path):
#         if os.path.isfile(os.path.join(directory_path, filename)):
#             with open(os.path.join(directory_path, filename), 'r') as file:
#                 content = file.read()
#                 print(f"File: {filename}, Content: {content}")

# # Replace 'path/to/your/directory' with the actual path to your directory
# directory_path = "data"
# read_directory(directory_path)

# Imports for simple llama index chat communicator. Non CLI

# core llama index class will call the Vector Store, the directory reader to load data from n amount of files into the index. The Settings import "is a bundle of commonly used resources used during the indexing and querying stage in a LlamaIndex pipeline/application" Telling llama index which forms of what models to use within querying stages and the indexing stages (what model to embed with) (what model to query with)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# sett the embedding model. If using local models (like I am with ollama) this will resolve to a local hugging face embedding model
from llama_index.core.embeddings import resolve_embed_model

# Ollama client to utilize a pulled and served dolphin-mixtral model (uncensored and large capabilies at 7b)
from llama_index.llms.ollama import Ollama

# Import Sys for CLI enablement
import sys
import logging

def queryDocument(): 
  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
  # Capture command line arguments into a List[str] args
  args = sys.argv

  #Check if there was no query (ie: list of args length is 0)
  if len(sys.argv) < 1:
    print("Define a query when running test.py")
    return

  # The command line argument is stored in args[1] as args[0] is the file name, test.py

  # Read the documents from the data folder.
  print("Loading documents")
  documents = SimpleDirectoryReader("data").load_data(show_progress="true")

  print("Setting embed model")
  # Set the embed model to a local hugging face (starts with "local" prefix)
  Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

  # Set the conversive LLM to a local Ollama server running a local dolphin-mixtral (or any other) model
  print("Setting language model for conversation")
  Settings.llm = Ollama(model="dolphin-mixtral:latest")

  # What is our index. The VectorStoreIndex. Im sure can use other vector store indexes here, but for simplicity and plug-and-play using llamaIndex core vectorStoreIndex
  print("Indexing Vector Store from loaded documents (in the data folder)")
  index = VectorStoreIndex.from_documents(documents)

  # as_query_engine resolves the LLM being used for queries via 
  # llm = (
  #         resolve_llm(llm, callback_manager=self._callback_manager)
  #         if llm
  #         else llm_from_settings_or_context(Settings, self.service_context)
  #       )
  print("Setting the query engine")
  query_engine = index.as_query_engine()

  # Does this commit work?

  print("Fetching response")
  response = query_engine.query(args[1])
  print(response)

if __name__ ==  '__main__':
  print("running")
  queryDocument()