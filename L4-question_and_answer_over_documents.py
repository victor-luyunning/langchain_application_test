import os

from IPython.display import display, Markdown
from anaconda_navigator.api.external_apps.bundle.installers import retrieve
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')
docs = loader.load()

embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://chat.aidirectlink.icu/v1"
)

embed = embeddings.embed_query("Hi my name is Harrison")

db=DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

query="Please suggest a shirt with sunblocking"

docs=db.similarity_search(query)

retriever = db.as_retriever()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://chat.aidirectlink.icu/v1"
)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.invoke(
    f"{qdocs} Question: Please list all your shirts "
    f"with sun protection in a table in markdown and summarize each one.")

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

response = qa_stuff.invoke(query)
print(response)

#一步代替
#from langchain.indexes import VectorstoreIndexCreator
#index = VectorstoreIndexCreator(
#    vectorstore_cls=DocArrayInMemorySearch,
#    embedding=embeddings
#).from_loaders([loader])
#response =index.query(query,llm=llm)
#print(response)