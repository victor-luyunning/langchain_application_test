import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

chat = ChatOpenAI(
    model="qwen-max-2025-01-25",
    temperature=0.0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


#template_string = """Translate the text \
#that is delimited by triple backticks \
#into a style that is {style}.
#text: ```{text}```
#"""
#
#customer_email = """
#Arrr, I be fuming that me blender lid \
#flew off and splattered me kitchen walls \
#with smoothie! And to make matters worse, \
#the warranty don't cover the cost of \
#cleaning up me kitchen. I need yer help \
#right now, matey!
#"""
#customer_style = """American English \
#in a calm and respectful tone
#"""
#prompt_template = ChatPromptTemplate.from_template(template_string)
#customer_messages = prompt_template.format_messages(
#    text=customer_email,
#    style=customer_style
#)
#customer_response = chat.invoke(customer_messages)
#print(customer_response.content)
#
#
#service_reply = """
#Hey there customer, \
#the warranty does not cover \
#cleaning expenses for your kitchen \
#because it's your fault that \
#you misused your blender \
#by forgetting to put the lid on before \
#starting the blender. \
#Tough luck! See ya!
#"""
#service_style_pirate = """
#a polite tone \
#that speaks in English Pirate
#"""
#prompt_template = ChatPromptTemplate.from_template(template_string)
#service_messages = prompt_template.format_messages(
#    text=service_reply,
#    style=service_style_pirate
#)
#service_response = chat.invoke(service_messages)
#print(service_response.content)


#review_template = """
#For the following text, extract the following information:
#
#gift: Was the item purchased as a gift for someone else? Answer True or False.
#delivery_days: How many days did it take for the product to arrive?
#price_value: Extract any sentences about the value or price.
#
#Format the output as JSON with the following keys:
#gift
#delivery_days
#price_value
#
#text: {text}
#"""
#customer_review = """
#This leaf blower is pretty amazing. It has four settings:
#candle blower, gentle breeze, windy city, and tornado.
#It arrived in two days, just in time for my wife's
#anniversary present.
#I think my wife liked it so much she was speechless.
#So far I've been the only one using it, and I've been
#using it every other morning to clear the leaves on our lawn.
#It's slightly more expensive than the other leaf blowers
#out there, but I think it's worth it for the extra features.
#"""
#prompt_template = ChatPromptTemplate.from_template(review_template)
#customer_messages=prompt_template.format_messages(text=customer_review)
#customer_response = chat.invoke(customer_messages)
#print(customer_response.content)


gift_schema = ResponseSchema(
    name="gift",
    description="Was the item purchased as a gift? Answer True/False."
)
delivery_days_schema = ResponseSchema(
    name="delivery_days",
    description="How many days did delivery take? Output -1 if unknown."
)
price_value_schema = ResponseSchema(
    name="price_value",
    description="Extract price/value comments as a list."
)
response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
review_template_2 = """
For the following text, extract:
gift: {gift_description}
delivery_days: {delivery_days_description}
price_value: {price_value_description}

text: {text}

{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(review_template_2)
customer_review_2 = """
This leaf blower is amazing. It has four settings and arrived in two days. 
It's slightly more expensive than others but worth it.
"""
messages = prompt.format_messages(
    text=customer_review_2,
    format_instructions=format_instructions,
    gift_description=gift_schema.description,
    delivery_days_description=delivery_days_schema.description,
    price_value_description=price_value_schema.description
)
response = chat.invoke(messages)
print(response.content)
output_dict = output_parser.parse(response.content)
print(f"Parser result: {output_dict}")

