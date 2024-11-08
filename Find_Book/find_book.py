
import os
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage


load_dotenv()

api_key=os.getenv("PINECONE_API_KEY")
environment=os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(
    api_key=api_key
)

#Define the directory-->
current_dir=os.path.dirname(__file__)
# file_dir=os.path.join(current_dir,"books","book1.pdf")
books_dir=os.path.join(current_dir,"books")
index_name = "chatinit"



print("\nChecking where the index is already exist in PinconeStore....")
# Check if the Pinecone index exists
if index_name not in pc.list_indexes().names():
    print("\nIndex does not exist. Initializing vector store...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            print (f"File not exist. Plaese correct the path")
        )
    #Listing all the file from books folder

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".pdf")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = PyPDFLoader(file_path)
        book_docs = loader.lazy_load()
        for i, doc in enumerate(book_docs):
            # Add metadata to each document indicating its source
            doc.metadata = {
                "source": book_file,
                "page": i+1
            }
            documents.append(doc)

    #for loading text file-->
    # loader = TextLoader(file_dir,encoding="utf-8")
    # documents = loader.load()

    print("\n--- Using Recursive Character-based Splitting ---")
    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = rec_char_splitter.split_documents(documents)

    # # Display information about the split documents
    # print("\n--- Document Chunks Information ---")
    # print(f"Number of document chunks: {len(docs)}")
    # print(f"Sample chunk:\n{docs[18].page_content}\n")

    print("\nCreating embedding...")
    
    # Create the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Extract text content from the documents
    texts = [doc.page_content for doc in docs]
    
    # embeddings = embedding_model.encode(texts, show_progress_bar=True)
    print("Embedding finished\n\n")

    # Create the Pinecone index
    print("Creating Pinecone Cloud......")
    pc.create_index(
    name=index_name,
    dimension=768, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
    print("...Pinecone Cloud Created\n")
    

    # index=pc.Index("langchain-rag1")
    print("Creating vectore store\n")
    # upserts = [(str(i), embeddings[i]) for i in range(len(embeddings))]
    # index.upsert(upserts)
    index=pc.Index(index_name)
    vectore_Store=PineconeVectorStore(index=index,embedding=embedding_model)
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vectore_Store.add_documents(documents=docs,id=uuids)
    print("\nVector store created\n")
else:
    print("\nVectore store already initialized. No need to initialize\n\n")

print("\nQuery Part")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index=pc.Index(index_name)
vectore_Store=PineconeVectorStore(index=index,embedding=embedding_model)
retriver=vectore_Store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.6},
)
print("\nQuestion and answer retrieval part-->> Ask your Question")

#Promts you can use::
#1. What is the last Muslim State in the Spanish Peninsula?
#2. In 1992, where was the United Nations Conference on Environment and Development (the Earth Summit) held?

chat_history=[]
print("Type 'exit' to exit....\n")
while True:
    query=input("\nYou: ")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query))    
    relevant_doc=retriver.invoke(query)
    relevant_content = "\n".join([doc.page_content for doc in relevant_doc])
    chat_history.append(AIMessage(content=relevant_content))
    print("\nRetrieving documents.......")
    #    Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_doc, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}\n")

print("\n----------Chat History-----------")
print(chat_history)