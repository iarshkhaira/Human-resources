from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load PDF document
loader = PyPDFLoader("LeavePolicy.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# Add metadata for citations
for chunk in chunks:
    chunk.metadata = {
        "document": "LeavePolicy.pdf",
        "section": "Casual Leave",
        "page": chunk.metadata.get("page", "NA")
    }

# Create embeddings + vector store
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# LLM + RetrievalQA chain
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Citation formatter
def format_citations(source_documents):
    citations = []
    for doc in source_documents:
        meta = doc.metadata
        citations.append(
            f"{meta.get('document', 'Unknown')} | "
            f"Section: {meta.get('section', 'N/A')} | "
            f"Page: {meta.get('page', 'N/A')}"
        )
    return citations

# Query
query = "How many casual leave days do I get?"

response = qa(query)

if not response["source_documents"]:
    print("Not found in HR policy documents.")
else:
    print("Answer:")
    print(response["result"])

    print("\nSource:")
    for cite in format_citations(response["source_documents"]):
        print(f"- {cite}")
