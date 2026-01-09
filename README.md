from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("LeavePolicy.pdf")
documents = loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
for chunk in chunks:
    chunk.metadata = {
        "document": "LeavePolicy.pdf",
        "section": "Casual Leave",
        "page": chunk.metadata.get("page", "NA")
    }
    from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.get_relevant_documents(query)
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-4")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

response = qa(query)

Answer ONLY using the provided context.
If the answer is not in the context, say "Not found in policy".
Always cite document name and section.
"Not found in current HR policies."
if not response["source_documents"]:
    return "Not found in HR policy documents."
