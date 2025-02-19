import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)
from langchain.chains import LLMChain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

llm = ChatOpenAI(
    model="llama3.3-70b-instruct",
    temperature=0.0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
llm2 = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url="https://chat.aidirectlink.icu/v1"
)
prompt = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

# ----------------- ConversationBufferMemory 示例 -----------------

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#legacy_chain = LLMChain(
#    llm=llm,
#    memory=memory,
#    prompt=prompt,
#    verbose=True
#)
#legacy_result = legacy_chain.invoke({"text": "my name is bob"})
#print(memory.load_memory_variables({})) # ==verbose=True
#legacy_result = legacy_chain.invoke({"text": "what was my name"})
#legacy_result = legacy_chain.invoke({"text": "what is 1+1?"})
#print(legacy_result)

# ----------------- ConversationBufferWindowMemory 示例 -----------------
#memory=ConversationBufferWindowMemory(k=1) # k取决于窗口大小，记住几轮对话
#memory.save_context({"input": "Hi"}, {"output": "my name is bob"})
#print(memory.load_memory_variables({}))
#memory.save_context({"input": "what is 1+1?"}, {"output": "It is 2"})
#print(memory.load_memory_variables({}))

# ----------------- ConversationTokenBufferMemory 示例 -----------------
#memory=ConversationTokenBufferMemory(llm=llm2, max_token_limit=20) # max_token_limit取决于token大小
#memory.save_context({"input": "Hi"}, {"output": "my name is bob"})
#memory.save_context({"input": "what is 1+1?"}, {"output": "It is 2"})
#memory.save_context({"input": "what is my name?"}, {"output": "bob"})
#print(memory.load_memory_variables({}))

# ----------------- ConversationSummaryBufferMemory 示例 -----------------
schedule = (
    "There is a meeting at 8am with your product team. "
    "You will need your powerpoint presentation prepared. "
    "9am-12pm have time to work on your LangChain project. "
    "At Noon, lunch with a customer interested in AI. "
    "Bring your laptop to show the latest LLM demo."
)
memory = ConversationSummaryBufferMemory(llm=llm2, max_token_limit=50)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, {"output": schedule})
print(memory.load_memory_variables({}))
