import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import tool

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

# Scrape Airtel Thanks App Web Page
def scrape_airtel_thanks_data():
    url = "https://www.airtel.in/airtel-thanks-app"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Scraping relevant sections from the website
    data = []
    for section in soup.find_all("section"):
        text = section.get_text(separator=" ", strip=True)
        if text:  # Only collect non-empty text
            data.append(text)

    return data

# Basic LLM ToolNode
class BasicLLMNode(ToolNode):
    def __init__(self, *args, **kwargs):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY)
        super().__init__(*args, **kwargs)

    @tool
    def chat_tool(self, message: str) -> str:
        """Tool that interacts with ChatOpenAI."""
        response = self.llm.invoke(message)
        return response.content

    def process(self, state: MessagesState):
        last_message = state.messages[-1] if state.messages else ""
        return self.chat_tool(last_message)

# Knowledge Base Node
class KnowledgeBaseNode(ToolNode):
    def __init__(self, *args, tools=None, **kwargs):
        # Ensure tools is an iterable (empty list if None)
        if tools is None:
            tools = []
        super().__init__(*args, tools=tools, **kwargs)
        self.vector_db = PineconeVectorStore(embedding_function=get_embeddings().embed_query, index_name="kukkur")

# Fallback Node
class FallbackNode(ToolNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, state: MessagesState):
        return "Sorry, I didn't understand your query. Please ask about Airtel services."

# Sim Swap Node
class SimSwapNode(ToolNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, state: MessagesState):
        if "sim swap" in state.messages[-1].lower():
            phone_number = input("Please provide your phone number: ")
            name = input("Please provide your name: ")
            return f"Sim swap initiated for {name} with phone number {phone_number}."
        return None

# Plan Details Node
class PlanDetailsNode(ToolNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, state: MessagesState):
        if "plan details" in state.messages[-1].lower():
            phone_number = input("Please provide your phone number: ")
            return f"Your plan includes 1.5 GB/day, Unlimited Calls, 100 SMS/day for Rs. 249."
        return None

# Build StateGraph
def build_graph():
    graph = StateGraph(
        start=START,
        nodes=[
            BasicLLMNode(name="basic_llm", tools=[BasicLLMNode.chat_tool]),
            KnowledgeBaseNode(name="knowledge_base"),
            FallbackNode(name="fallback"),
            SimSwapNode(name="sim_swap"),
            PlanDetailsNode(name="plan_details"),
            END
        ]
    )

    # Transitions based on user queries
    graph.add_transition(START, "basic_llm", condition=lambda state: state.messages[-1].lower() not in ["sim swap", "plan details"])
    graph.add_transition("basic_llm", "knowledge_base", condition=lambda state: "airtel" in state.messages[-1].lower())
    graph.add_transition("knowledge_base", "fallback", condition=lambda state: state.responses[-1] == "No relevant information found.")
    graph.add_transition(START, "sim_swap", condition=lambda state: "sim swap" in state.messages[-1].lower())
    graph.add_transition(START, "plan_details", condition=lambda state: "plan details" in state.messages[-1].lower())
    graph.add_transition("sim_swap", END)
    graph.add_transition("plan_details", END)
    graph.add_transition("fallback", END)

    return graph

# Process user queries through the state graph
def process_query(query: str, graph: StateGraph):
    state = MessagesState()  # Create a new message state
    state.messages.append(query)  # Add the user query to state
    response = graph.process(state)  # Process through the graph
    return response  # Return the response

# Main Program
if __name__ == "__main__":
    # Scrape Airtel Thanks App data
    airtel_data = scrape_airtel_thanks_data()

    # Build the agent system graph
    graph = build_graph()

    # Example user queries
    user_queries = [
        "Tell me about Airtel Broadband",
        "I want to swap my SIM card",
        "What is my plan?",
        "Unrelated query"
    ]

    # Loop through the queries and print the responses
    for query in user_queries:
        print(f"User Query: {query}")
        response = process_query(query, graph)
        print(f"Response: {response}")

# import os
# import requests
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# from langgraph.graph import StateGraph, MessagesState,START, END
# from langgraph.prebuilt import ToolNode
# from langchain_openai import ChatOpenAI
# from langchain_pinecone import Pinecone
# #from langchain_community.embeddings import OpenAIEmbeddings
# from pinecone import Pinecone as PineconeClient, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore  # Updated import
# from langchain_openai import OpenAIEmbeddings  # Updated import

# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.tools import tool

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


# # Scrape Airtel Thanks App Web Page
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

# # Store scraped data into Pinecone after embedding
# #ed in Pinecone!")
# # Store scraped data into Pinecone after embedding

# # def store_in_pinecone(data):
# #     pc = init_pinecone()  # Initialize Pinecone client
# #     index_name = "kukkur"  # Your existing index name
# #     if index_name not in pc.list_indexes().names():
# #         pc.create_index(
# #             name=index_name,
# #             dimension=1536,
# #             metric='euclidean',
# #             spec=ServerlessSpec(
# #                 cloud='aws',
# #                 region='us-east-1'  # Use the correct region here
# #             )
# #         )
    
# #     embeddings = get_embeddings()  # Get the embeddings
# #     vector_db = PineconeVectorStore(embedding=embeddings, index_name=index_name)  # Updated constructor

# #     # Process each chunk of scraped data and store in Pinecone
# #     for chunk in data:
# #         embedding = embeddings.embed_query(chunk)
# #         # Create a metadata dictionary to store additional information
# #         metadata = {
# #             "text": chunk  # Use "text" as a key
# #         }
# #         vector_db.add_texts([chunk], embeddings=[embedding], metadatas=[metadata])  # Updated call with metadata

# #     print("Data has been successfully embedded and stored in Pinecone!")

# # # Basic LLM ToolNode
# # class BasicLLMNode(ToolNode):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)

# #     def process(self, state: MessagesState):
# #         llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
# #         response = llm.invoke(state.messages[-1])  # Use the last user message
# #         return response.content

# # # Knowledge Base Node - Airtel Thanks App
# # class KnowledgeBaseNode(ToolNode):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.vector_db = PineconeVectorStore(embedding_function=get_embeddings().embed_query, index_name="airtel-knowledge")  # Updated to use PineconeVectorStore

# #     def process(self, state: MessagesState):
# #         query = state.messages[-1]  # Last user message
# #         embedding = get_embeddings().embed_query(query)
# #         results = self.vector_db.similarity_search(query=embedding, k=3)
# #         if results:
# #             return results[0]['text']  # Return top result from the knowledge base
# #         return "No relevant information found in our knowledge base."
# # Basic LLM ToolNode
# class BasicLLMNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         self.llm = ChatOpenAI(api_key=OPENAI_API_KEY)
#         super().__init__(*args, **kwargs)

#     @tool
#     def chat_tool(self, message: str) -> str:
#         """Tool that interacts with ChatOpenAI."""
#         response = self.llm.invoke(message)
#         return response.content

#     def process(self, state: MessagesState):
#         last_message = state.messages[-1] if state.messages else ""
#         return self.chat_tool(last_message)

# # Update other ToolNode classes similarly if they require tools
# # Example for KnowledgeBaseNode
# class KnowledgeBaseNode(ToolNode):
#     def __init__(self, *args, tools=None, **kwargs):
#         super().__init__(*args, tools=tools, **kwargs)
#         self.vector_db = PineconeVectorStore(embedding_function=get_embeddings().embed_query, index_name="airtel-knowledge")

# # Fallback Node for random queries
# class FallbackNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process(self, state: MessagesState):
#         return "Sorry, I didn't understand your query. Please ask about Airtel services."

# # Sim Swap Workflow Node
# class SimSwapNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process(self, state: MessagesState):
#         if "sim swap" in state.messages[-1].lower():
#             phone_number = input("Please provide your phone number: ")
#             name = input("Please provide your name: ")
#             return f"Sim swap initiated for {name} with phone number {phone_number}."
#         return None

# # Plan Details Workflow Node
# class PlanDetailsNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process(self, state: MessagesState):
#         if "plan details" in state.messages[-1].lower():
#             phone_number = input("Please provide your phone number: ")
#             return f"Your plan includes 1.5 GB/day, Unlimited Calls, 100 SMS/day for Rs. 249."
#         return None

# # # Build StateGraph
# # def build_graph():
# #     graph = StateGraph(
# #         start=START,
# #         nodes=[
# #             BasicLLMNode(name="basic_llm"),
# #             KnowledgeBaseNode(name="knowledge_base"),
# #             FallbackNode(name="fallback"),
# #             SimSwapNode(name="sim_swap"),
# #             PlanDetailsNode(name="plan_details"),
# #             END
# #         ]
# #     )

# #     # Transitions based on user queries
# #     graph.add_transition(START, "basic_llm", condition=lambda state: state.messages[-1].lower() not in ["sim swap", "plan details"])
# #     graph.add_transition("basic_llm", "knowledge_base", condition=lambda state: "airtel" in state.messages[-1].lower())
# #     graph.add_transition("knowledge_base", "fallback", condition=lambda state: state.responses[-1] == "No relevant information found.")
# #     graph.add_transition(START, "sim_swap", condition=lambda state: "sim swap" in state.messages[-1].lower())
# #     graph.add_transition(START, "plan_details", condition=lambda state: "plan details" in state.messages[-1].lower())
# #     graph.add_transition("sim_swap", END)
# #     graph.add_transition("plan_details", END)
# #     graph.add_transition("fallback", END)

# #     return graph

# # Build StateGraph
# def build_graph():
#     graph = StateGraph(
#         start=START,
#         nodes=[
#             BasicLLMNode(name="basic_llm",tools=[BasicLLMNode.chat_tool]),
#             KnowledgeBaseNode(name="knowledge_base"),
#             FallbackNode(name="fallback"),
#             SimSwapNode(name="sim_swap"),
#             PlanDetailsNode(name="plan_details"),
#             END
#         ]
#     )

#     # Transitions based on user queries
#     graph.add_transition(START, "basic_llm", condition=lambda state: state.messages[-1].lower() not in ["sim swap", "plan details"])
#     graph.add_transition("basic_llm", "knowledge_base", condition=lambda state: "airtel" in state.messages[-1].lower())
#     graph.add_transition("knowledge_base", "fallback", condition=lambda state: state.responses[-1] == "No relevant information found.")
#     graph.add_transition(START, "sim_swap", condition=lambda state: "sim swap" in state.messages[-1].lower())
#     graph.add_transition(START, "plan_details", condition=lambda state: "plan details" in state.messages[-1].lower())
#     graph.add_transition("sim_swap", END)
#     graph.add_transition("plan_details", END)
#     graph.add_transition("fallback", END)

#     return graph


# # Process user queries through the state graph
# def process_query(query: str, graph: StateGraph):
#     state = MessagesState()  # Create a new message state
#     state.messages.append(query)  # Add the user query to state
#     response = graph.process(state)  # Process through the graph
#     return response  # Return the response

# # Main Program
# if __name__ == "__main__":
#     # Scrape Airtel Thanks App data and store it in Pinecone
#     airtel_data = scrape_airtel_thanks_data()
#     #store_in_pinecone(airtel_data)

#     # Build the agent system graph
#     graph = build_graph()

#     # Example user queries
#     user_queries = [
#         "Tell me about Airtel Broadband",
#         "I want to swap my SIM card",
#         "What is my plan?",
#         "Unrelated query"
#     ]

#     # Loop through the queries and print the responses
#     for query in user_queries:
#         print(f"User Query: {query}")
#         response = process_query(query, graph)
#         print(f"Response: {response}")

# # Required Libraries
# import os
# from langgraph.graph import StateGraph,MessagesState,START, END
# from langgraph.prebuilt import ToolNode
# # from langgraph.constants import START, END
# from langchain_openai import ChatOpenAI
# from langchain.vectorstores import Pinecone
# from langchain.embeddings import OpenAIEmbeddings

# # Pinecone Setup (Vector Database)
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize Pinecone Vector DB
# def init_pinecone():
#     pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
#     return Pinecone(
#         api_key=PINECONE_API_KEY,
#         index_name="airtel-knowledge",
#         environment="us-west1-gcp"
#     )

# # Embedding model for VectorDB
# def get_embeddings():
#     return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# # Basic LLM ToolNode
# class BasicLLMNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process(self, state: MessagesState):
#         llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
#         response = llm.invoke(state.messages[-1])  # Use the last user message
#         return response.content

# # Knowledge Base Node - Airtel Thanks App
# class KnowledgeBaseNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.vector_db = init_pinecone()

#     def process(self, state: MessagesState):
#         query = state.messages[-1]  # Last user message
#         embedding = get_embeddings().embed_query(query)
#         results = self.vector_db.similarity_search(query=embedding, k=3)
#         if results:
#             return results[0]['text']  # Return top result from the knowledge base
#         return "No relevant information found in our knowledge base."

# # Fallback Node for random queries
# class FallbackNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process(self, state: MessagesState):
#         return "Sorry, I didn't understand your query. Please ask about Airtel services."

# # Sim Swap Workflow Node
# class SimSwapNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process(self, state: MessagesState):
#         if "sim swap" in state.messages[-1].lower():
#             phone_number = input("Please provide your phone number: ")
#             name = input("Please provide your name: ")
#             return f"Sim swap initiated for {name} with phone number {phone_number}."
#         return None

# # Plan Details Workflow Node
# class PlanDetailsNode(ToolNode):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process(self, state: MessagesState):
#         if "plan details" in state.messages[-1].lower():
#             phone_number = input("Please provide your phone number: ")
#             return f"Your plan includes 1.5 GB/day, Unlimited Calls, 100 SMS/day for Rs. 249."
#         return None

# # Build StateGraph
# def build_graph():
#     graph = StateGraph(
#         start=START,
#         nodes=[
#             BasicLLMNode(name="basic_llm"),
#             KnowledgeBaseNode(name="knowledge_base"),
#             FallbackNode(name="fallback"),
#             SimSwapNode(name="sim_swap"),
#             PlanDetailsNode(name="plan_details"),
#             END
#         ]
#     )

#     # Transitions based on user queries
#     graph.add_transition(START, "basic_llm", condition=lambda state: state.messages[-1].lower() not in ["sim swap", "plan details"])
#     graph.add_transition("basic_llm", "knowledge_base", condition=lambda state: "airtel" in state.messages[-1].lower())
#     graph.add_transition("knowledge_base", "fallback", condition=lambda state: state.responses[-1] == "No relevant information found.")
#     graph.add_transition(START, "sim_swap", condition=lambda state: "sim swap" in state.messages[-1].lower())
#     graph.add_transition(START, "plan_details", condition=lambda state: "plan details" in state.messages[-1].lower())
#     graph.add_transition("sim_swap", END)
#     graph.add_transition("plan_details", END)
#     graph.add_transition("fallback", END)

#     return graph

# # Process user queries through the state graph
# def process_query(query: str, graph: StateGraph):
#     state = MessagesState()  # Create a new message state
#     state.messages.append(query)  # Add the user query to state
#     response = graph.process(state)  # Process through the graph
#     return response  # Return the response

# # Main Program
# if __name__ == "__main__":
#     # Build the agent system graph
#     graph = build_graph()

#     # Example user queries
#     user_queries = [
#         "Tell me about Airtel Broadband",
#         "I want to swap my SIM card",
#         "What is my plan?",
#         "Unrelated query"
#     ]

#     # Loop through the queries and print the responses
#     for query in user_queries:
#         print(f"User Query: {query}")
#         response = process_query(query, graph)
#         print(f"Response: {response}")
