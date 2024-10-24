# embedding.py

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone Vector DB
def init_pinecone():
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    return pc

# Embedding model for VectorDB
def get_embeddings():
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Function to initialize the vector store
def get_vector_store():
    pc = init_pinecone()
    index_name = "kukkur"
    
    # Check if index exists, otherwise create one
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    return vector_store

# Function to scrape data from Airtel Thanks App webpage
def scrape_airtel_thanks_data():
    url = "https://www.airtel.in/airtel-thanks-app"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []
    for section in soup.find_all("section"):
        text = section.get_text(separator=" ", strip=True)
        if text:
            data.append(text)

    return data

# Function to store scraped data into Pinecone
def store_in_pinecone(data):
    vector_store = get_vector_store()  # Use get_vector_store to get the store
    
    embeddings = get_embeddings()
    
    for chunk in data:
        embedding = embeddings.embed_query(chunk)
        metadata = {
            "text": chunk
        }
        vector_store.add_texts([chunk], embeddings=[embedding], metadatas=[metadata])

    print("Data has been successfully embedded and stored in Pinecone!")

# Main function to run the embedding process
if __name__ == "__main__":
    scraped_data = scrape_airtel_thanks_data()
    store_in_pinecone(scraped_data)



# import os
# import requests
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# from pinecone import Pinecone as PineconeClient, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore  # Updated import
# from langchain_openai import OpenAIEmbeddings  # Updated import

# # Load environment variables from .env file
# load_dotenv()

# # API Keys
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize Pinecone Vector DB
# def init_pinecone():
#     pc = PineconeClient(api_key=PINECONE_API_KEY)
#     return pc

# # Embedding model for VectorDB
# def get_embeddings():
#     # Use the updated OpenAIEmbeddings class
#     return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# # Function to scrape data from Airtel Thanks App webpage
# def scrape_airtel_thanks_data():
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     # Scraping relevant sections from the website
#     data = []
#     for section in soup.find_all("section"):
#         text = section.get_text(separator=" ", strip=True)
#         if text:  # Only collect non-empty text
#             data.append(text)

#     return data

# # Function to store scraped data into Pinecone
# def store_in_pinecone(data):
#     pc = init_pinecone()  # Initialize Pinecone client
#     index_name = "kukkur"  # Your existing index name

#     # Check if index exists, otherwise create one
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=1536,
#             metric='euclidean',
#             spec=ServerlessSpec(
#                 cloud='aws',
#                 region='us-east-1'  # Use the correct region here
#             )
#         )
    
#     # Get OpenAI embeddings
#     embeddings = get_embeddings()
    
#     # Initialize Pinecone vector store
#     vector_db = PineconeVectorStore(embedding=embeddings, index_name=index_name)

#     # Process each chunk of scraped data and store in Pinecone with metadata
#     for chunk in data:
#         embedding = embeddings.embed_query(chunk)
        
#         # Create a metadata dictionary to store additional information
#         metadata = {
#             "text": chunk  # Store the scraped text as metadata
#         }
        
#         # Store the chunk and its embedding in Pinecone
#         vector_db.add_texts([chunk], embeddings=[embedding], metadatas=[metadata])

#     print("Data has been successfully embedded and stored in Pinecone!")

# # Main function to run the embedding process
# if __name__ == "__main__":
#     # Scrape Airtel Thanks App data
#     scraped_data = scrape_airtel_thanks_data()
    
#     # Store the scraped data in Pinecone
#     store_in_pinecone(scraped_data)
