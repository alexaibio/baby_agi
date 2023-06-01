'''
PERSONAL ASSISTANT with LangChain and OpenAI
https://python.langchain.com/en/latest/use_cases/agents/baby_agi.html
https://python.langchain.com/en/latest/use_cases/personal_assistants.html
'''
import os
from typing import Dict, List, Optional, Any
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.base import Chain
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss
from my_controller import BabyAGI
import openai
#from settings import config
openai.api_key = os.getenv('OPENAI_API_KEY')     #config["OPENAI_API_KEY"]


# Define your embedding model
embeddings_model = OpenAIEmbeddings()

######## Faiss: A library for efficient similarity search
# Initialize the vector store as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# initialize the LLM model
llm = OpenAI(temperature=0)  # define a model

# initialize BabyAGI
verbose = False   # Logging of LLMChains
max_iterations: Optional[int] = 5   # If None, will keep on and going forever
baby_agi = BabyAGI.from_llm(llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations)

# run it
OBJECTIVE = "Help me to find a role of PDE10A in treatment of psychomotor agitation "
print()
print(f'OBJECTIVE: {OBJECTIVE}')
baby_agi({"objective": OBJECTIVE})

