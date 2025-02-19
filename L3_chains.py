import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiPromptChain, MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser



df=pd.read_csv("Data.csv")
llm = ChatOpenAI(
    model="qwen-max-2025-01-25",
    temperature=0.9,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
product = "Queen Size Sheet Set"

#chain = LLMChain(
#    llm=llm,
#    prompt=prompt,
#    verbose=True,
#)
#chain.invoke(product)

### SimpleSequentialChain
## prompt template 1
#first_prompt = ChatPromptTemplate.from_template(
#    "What is the best name to describe a company that makes {product}?"
#)
## Chain 1
#chain_one = LLMChain(llm=llm, prompt=first_prompt)
## prompt template 2
#second_prompt = ChatPromptTemplate.from_template(
#    "Write a 20 words description for the following company:{company_name}"
#)
## chain 2
#chain_two = LLMChain(llm=llm, prompt=second_prompt)
#overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
#overall_simple_chain.invoke(product)

### SequentialChain
## prompt template 1: translate to english
#first_prompt = ChatPromptTemplate.from_template(
#    "Translate the following review to english:\n\n{Review}"
#)
## chain 1: input= Review and output= English_Review
#chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")
## prompt template 2: summarize
#second_prompt = ChatPromptTemplate.from_template(
#    "Can you summarize the following review in 1 sentence:\n\n{English_Review}"
#)
## chain 2: input= English_Review and output= summary
#chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")
## prompt template 3: translate to english
#third_prompt = ChatPromptTemplate.from_template(
#    "What language is the following review:\n"
#    "\n{Review}"
#)
## chain 3: input= Review and output= language
#chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")
## prompt template 4: follow-up message
#fourth_prompt = ChatPromptTemplate.from_template(
#    "Write a follow up response to the following summary in the specified language:\n"
#    "\nSummary: {summary}\n"
#    "\nLanguage: {language}"
#)
## chain 4: input= summary, language and output= followup_message
#chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")
## overall_chain: input= Review and output= English_Review,summary, followup_message
#overall_chain = SequentialChain(
#    chains=[chain_one, chain_two, chain_three, chain_four],
#    input_variables=["Review"],
#    output_variables=["English_Review", "summary", "followup_message"],
#    verbose=True
#)
#review = df.Review[5]
#overall_chain.invoke(review)
#print(overall_chain.invoke(review))

## Router Chain

from typing import Literal
import os
import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Define the prompts we will route to
prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very smart physics professor. \
        You are great at answering questions about physics in a concise\
        and easy to understand manner. \
        When you don't know the answer to a question you admit\
        that you don't know."),
        ("human", "{input}")
    ]
)
prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a very good mathematician. \
        You are great at answering math questions. \
        You are so good because you are able to break down \
        hard problems into their component parts, 
        answer the component parts, and then put them together\
        to answer the broader question."""),
        ("human", "{input}")
    ]
)
prompt_3 = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a very good historian. \
        You have an excellent knowledge of and understanding of people,\
        events and contexts from a range of historical periods. \
        You have the ability to think, reflect, debate, discuss and \
        evaluate the past. You have a respect for historical evidence\
        and the ability to make use of it to support your explanations \
        and judgements."""),
        ("human", "{input}")
    ]
)
prompt_4 = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a successful computer scientist.\
        You have a passion for creativity, collaboration,\
        forward-thinking, confidence, strong problem-solving capabilities,\
        understanding of theories and algorithms, and excellent communication \
        skills. You are great at answering coding questions. \
        You are so good because you know how to solve a problem by \
        describing the solution in imperative steps \
        that a machine can easily interpret and you know how to \
        choose a solution that has a good balance between \
        time complexity and space complexity."""),
        ("human", "{input}")
    ]
)

# Construct the chains we will route to. These format the input query
# into the respective prompt, run it through a chat model, and cast
# the result to a string.
chain_1 = prompt_1 | llm | StrOutputParser()
chain_2 = prompt_2 | llm | StrOutputParser()
chain_3 = prompt_3 | llm | StrOutputParser()
chain_4 = prompt_4 | llm | StrOutputParser()


# Next: define the chain that selects which branch to route to.
# Here we will take advantage of tool-calling features to force
# the output to select one of two desired branches.
route_system = (
    "Given a raw text input to a language model select the model prompt "
    "best suited for the input. You will be given the names of the available "
    "prompts and a description of what the prompt is best suited for. You may "
    "also revise the original input if you think that revising it will "
    "ultimately lead to a better response from the language model.\n\n"
    "<< FORMATTING >>\n"
    "Return a markdown code snippet with a JSON object formatted as follows:\n"
    "```json\n"
    "{{\"destination\": string, \"next_inputs\": string}}\n"
    "```\n"
    "REMEMBER: \"destination\" MUST be one of the candidate prompts\n"
    "REMEMBER: \"next_inputs\" can just be the original input if you don't need to modify it\n\n"
    "<< CANDIDATE PROMPTS >>\n{destinations}\n\n"
    "<< INPUT >>\n{{input}}\n\n"
    "<< OUTPUT (remember to include the ```json)>>"
)

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{input}"),
    ]
)


# Define schema for output:
class RouteQuery(TypedDict):
    """Route query to destination expert."""

    destination: Literal["physics", "math", "history", "computer_science"]


# For LangGraph, we will define the state of the graph to hold the query,
# destination, and final answer.
class State(TypedDict):
    query: str
    destination: RouteQuery
    answer: str


# 在定义 route_query 函数时，确保传递正确的 input 和 destinations 参数
async def route_query(state: State, config: RunnableConfig):
    destination_options = {
        "physics": "物理问题",
        "math": "数学问题",
        "history": "历史问题",
        "computer_science": "计算机科学问题"
    }
    formatted_destinations = "\n".join([f"{k}: {v}" for k, v in destination_options.items()])

    input_dict = {
        "input": state["query"],
        "destinations": formatted_destinations,
    }
    destination_result = await route_chain.ainvoke(input_dict, config)
    # 假设destination_result是一个字典，但做额外检查以防万一
    if isinstance(destination_result, list) and len(destination_result) > 0:
        destination_result = destination_result[0]  # 如果结果是一个列表，尝试获取第一个元素
    # Assuming the result contains 'destination' and 'next_inputs'
    return {"destination": destination_result.get('destination'), "query": destination_result.get('next_inputs')}


# 修改prompt_x函数以直接使用chain_x，并且不需要重新构建messages
async def prompt_1(state: State, config: RunnableConfig):
    answer = await chain_1.ainvoke({"input": state["query"]}, config)
    return {"answer": answer}


async def prompt_2(state: State, config: RunnableConfig):
    answer = await chain_2.ainvoke({"input": state["query"]}, config)
    return {"answer": answer}


async def prompt_3(state: State, config: RunnableConfig):
    answer = await chain_3.ainvoke({"input": state["query"]}, config)
    return {"answer": answer}


async def prompt_4(state: State, config: RunnableConfig):
    answer = await chain_4.ainvoke({"input": state["query"]}, config)
    return {"answer": answer}


# 更新select_node函数以返回链名而不是字符串
def select_node(state: State) -> str:
    if state["destination"] == "physics":
        return "prompt_1"
    elif state["destination"] == "math":
        return "prompt_2"
    elif state["destination"] == "history":
        return "prompt_3"
    else:
        return "prompt_4"


# 确保route_chain的定义符合预期
route_chain = route_prompt | llm.with_structured_output(RouteQuery)

graph = StateGraph(State)
graph.add_node("route_query", route_query)
graph.add_node("prompt_1", prompt_1)
graph.add_node("prompt_2", prompt_2)
graph.add_node("prompt_3", prompt_3)
graph.add_node("prompt_4", prompt_4)

graph.add_edge(START, "route_query")
graph.add_conditional_edges("route_query", select_node)
graph.add_edge("prompt_1", END)
graph.add_edge("prompt_2", END)
graph.add_edge("prompt_3", END)
graph.add_edge("prompt_4", END)
app = graph.compile()



async def test_app():
    # 定义测试查询
    test_queries = {
        0: "What is the speed of light?",
        1: "What is black body radiation?",
        2: "Who is the most popular poem in Tang Dynasty",
        3: "Why does every cell in our body contain DNA?"
    }
    # 创建初始状态
    for key,test_query in test_queries.items():
        initial_state = {
            "query": test_query,
            "destination": None,
            "answer": None,
        }
        # 运行应用并传递初始状态
        result = await app.ainvoke(initial_state, config={})
        # 打印结果
        print("Routing decision:", result.get("destination"))
        print("Answer:", result["answer"])

# 运行测试
asyncio.run(test_app())
