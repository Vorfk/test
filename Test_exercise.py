from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import tiktoken
import requests
import html2text
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_PROXY = os.getenv('OPENAI_PROXY')
prompt_template = os.getenv('prompt_template')
url = os.getenv('url')

llm = OpenAI(model_name="gpt-3.5-turbo-instruct",
             openai_api_key=OPENAI_API_KEY,
             temperature=0,
             openai_proxy=OPENAI_PROXY)
prompt = PromptTemplate.from_template(prompt_template)

response = requests.get(url)
html_code = response.text
h = html2text.HTML2Text()
h.ignore_links = True
text_from_html = h.handle(html_code)

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")

if len(encoding.encode(text_from_html)) >= 4000:
    raise ValueError("Number of prompt's tokens is more than 4000")

output = llm.invoke(prompt.format(html=text_from_html))
print(output)
jytkt