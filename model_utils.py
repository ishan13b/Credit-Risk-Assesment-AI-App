import os
from llama_index.llms.groq import Groq

os.environ["GROQ_API_KEY"] = "<TOKEN>"



llm = Groq(model="llama-3.1-8b-instant")
llm_70b = Groq(model="llama-3.1-70b-versatile")
llm_70b_tool = Groq(model="llama3-groq-70b-8192-tool-use-preview")