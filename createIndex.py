import os

from langchain.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
from llama_index import VectorStoreIndex, SimpleDirectoryReader

from llama_index import ServiceContext
from langchain.llms import CTransformers
DATA_PATH = 'data/'



##def load_llm():

   ###   model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
     ###   model_type="llama",
    ###    max_new_tokens = 512,
     ##   temperature = 0.5
  ##  )
  ##  return llm
llm=OpenAI(temperature=0.6)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20)
def createIndex():
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    index = VectorStoreIndex.from_documents(documents,service_context=service_context)
    
    index.storage_context.persist("EMBEDDINGS_STORE_USING_LLAMA_INDEX")
    

    
    
if __name__ == "__main__":
    createIndex()