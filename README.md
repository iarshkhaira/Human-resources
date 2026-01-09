# 1. Imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 2. Load PDF document
loader = PyPDFLoader("LeavePolicy.pdf")
documents = loader.load()

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# 4. Add metadata for citations
for chunk in chunks:
    chunk.metadata.update({
        "document": "LeavePolicy.pdf",
        "section": "Casual Leave",
        "page": chunk.metadata.get("page", "NA")
    })

# 5. Create embeddings + vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 6. LLM + RetrievalQA chain
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Citation formatter
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

# 8. Query
query = "How many casual leave days do I get?"
response = qa(query)

# 9. Output
if not response["source_documents"]:
    print("Not found in HR policy documents.")
else:
    print("Answer:")
    print(response["result"])

    print("\nSource:")
    for cite in format_citations(response["source_documents"]):
        print(f"- {cite}")
