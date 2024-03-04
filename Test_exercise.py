import os
import tiktoken
import requests
import html2text
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

if os.path.exists(os.getcwd()+r'/.env'):
    load_dotenv()
else:
    raise FileNotFoundError(".env not found")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_PROXY = os.getenv('OPENAI_PROXY')
prompt_template = "From {html} give me description of item 'Топор Optimal 800г, Matrix/21658', no more than 500 characters, output in russian"
URL_TO_PARSE = os.getenv('URL_TO_PARSE')

llm = OpenAI(model_name="gpt-3.5-turbo-instruct",
             openai_api_key=OPENAI_API_KEY,
             temperature=0,
             openai_proxy=OPENAI_PROXY)
prompt = PromptTemplate.from_template(prompt_template)

response = requests.get(URL_TO_PARSE)
html_code = response.text
h = html2text.HTML2Text()
h.ignore_links = True
text_from_html = h.handle(html_code)

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")

if len(encoding.encode(text_from_html)) >= 4000:
    raise ValueError("Number of prompt's tokens is more than 4000")

output = llm.invoke(prompt.format(html=text_from_html))
print(output)