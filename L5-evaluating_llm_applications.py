import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
import langchain
langchain.debug = True

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')
data = loader.load()
embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://chat.aidirectlink.icu/v1"
)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings
).from_loaders([loader])

llm = ChatOpenAI(
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://chat.aidirectlink.icu/v1"
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
)
#print(data[10])
#print(data[11])
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(
    ChatOpenAI(
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://chat.aidirectlink.icu/v1"
    ))

new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

examples += new_examples

qa.invoke(examples[0]["query"])
print(qa.invoke(examples[0]["query"]))