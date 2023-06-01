# https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81
import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import PALChain
openai.api_key = os.getenv('OPENAI_API_KEY')


llm = OpenAI(
          model_name="text-davinci-003", # default model
          temperature=0.9
)


#PAL - Programme Aided Language Model
palchain = PALChain.from_math_prompt(llm=llm, verbose=True)
print(palchain.prompt.template)

palchain.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")
