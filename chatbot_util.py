
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import AzureOpenAIEmbeddings
# ServerlessSpec - necessary to create an index
from pinecone import ServerlessSpec

load_dotenv(find_dotenv(),override=True)

def load_document():

    print("loading json data...")
    loader = JSONLoader(file_path = 'data/productsales.json',
                        jq_schema=".data[]",
                        text_content=False)
    data = loader.load()
    return data

def chunk_document(data, chunk_size=500, chunk_overlap=0):

   print("chunking data...")
   text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   chunks = text_splitter.split_documents(data)
   return chunks

# Embedding and Uploading to a Vector DB (PINECONE)
def insert_or_fetch_embeddings(index_name, chunks):

    pc = pinecone.Pinecone() #pinecone api_key already in .env(dotenv)
    embeddings = AzureOpenAIEmbeddings(model= 'text-embedding-ada-002', dimensions=1536, api_version="2023-03-15-preview")

    if index_name in pc.list_indexes().names():
        print(f"Index name {index_name} already exists. Loading embeddings...", end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)

    else:
        print(f"Creating new index {index_name} and embeddings...", end='')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        vector_store = Pinecone.from_documents(chunks,embeddings,index_name=index_name)

    return  vector_store


def delete_pinecone_index(index_name='all'):
    pc = pinecone.Pinecone()
    if index_name == 'all':

        for index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
        print("Deleted all index.")

    else :
        pc.delete_index(index_name)
        print(f"Deleted {index_name} index.")