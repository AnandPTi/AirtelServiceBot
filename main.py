import os
import openai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Data Loading and Processing

def load_airtel_data() -> list:
    """Scrape the Airtel website and extract relevant text data."""
    url = "https://www.airtel.in/airtel-thanks-app"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text content from paragraphs
    content = [p.get_text() for p in soup.find_all('p')]
    return content

def index_data(content: list) -> Any:
    """Index the scraped data into a FAISS vector store."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(content, embeddings)
    return vector_store

# Search Functionality

def search_vector_store(query: str, vector_store: Any) -> str:
    """Perform a search in the FAISS vector store."""
    results = vector_store.similarity_search(query, k=2)
    return f"Retrieved the following info from Airtel site: {results}"

# Tool Functionality

def airtel_help_bot(query: str) -> Dict[str, Any]:
    """Handle general queries related to Airtel."""
    return {"response": f"Handled by Airtel Help Bot: {query}"}

def retrieve_from_vectorstore(query: str, vector_store: Any) -> Dict[str, Any]:
    """Retrieve information from the vector store based on the query."""
    return {"response": search_vector_store(query, vector_store)}

def retrieve_tool(query: str, vector_store: Any) -> str:
    """Retrieve information from the vector store based on user query."""
    return retrieve_from_vectorstore(query, vector_store)

def sim_swap_workflow(query: str) -> Dict[str, Any]:
    """Initiate the SIM swap process."""
    user_phone = input("Please provide your phone number: ")
    user_name = input("Please provide your name: ")
    return {"response": f"SIM swap initiated for {user_name} ({user_phone})"}

def plan_details_workflow(query: str) -> Dict[str, Any]:
    """Provide details about the user's plan."""
    user_phone = input("Please provide your phone number to check your plan: ")
    return {"response": f"Your current plan for {user_phone} is: 'Unlimited Calls with 2GB/day'"}

# Tool Creation

def create_tools(vector_store: Any):
    """Create tool nodes for the workflow."""
    
    tools = [
        ToolNode([airtel_help_bot]),  # Airtel help bot node
        ToolNode([retrieve_tool]),  # Vector store retrieval node
        ToolNode([sim_swap_workflow]),  # SIM swap workflow node
        ToolNode([plan_details_workflow])  # Plan details workflow node
    ]
    
    return tools

# Routing Logic

def route_question(state: Dict[str, Any]) -> str:
    """Route the user's question to the appropriate workflow."""
    query = state.get("query", "").lower()
    
    if "sim swap" in query:
        return "sim_swap"
    elif "plan" in query:
        return "plan_details"
    elif "airtel" in query:
        return "airtel_help"
    else:
        return "retrieve"

# Workflow Creation

def create_workflow() -> Any:
    """Initialize the workflow graph with tools and nodes."""
    airtel_data = load_airtel_data()
    vector_store = index_data(airtel_data)
    
    tools = create_tools(vector_store)
    
    # Initialize the workflow graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes for each tool
    for i, tool in enumerate(tools):
        node_name = f"tool_node_{i}"
        workflow.add_node(node_name, tool)
        workflow.add_edge(node_name, END)

    # Build conditional edges based on user queries
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "airtel_help": "tool_node_0",
            "retrieve": "tool_node_1",
            "sim_swap": "tool_node_2",
            "plan_details": "tool_node_3"
        }
    )
    
    return workflow.compile()

# Run Workflow

# Run Workflow
def run_workflow(query: str):
    """Run the workflow with the provided query."""
    app = create_workflow()
    
    # Create initial state, adding query as a message
    initial_state = {
        "query": query,
        "messages": [{"role": "user", "content": query}]
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Print the messages
    for message in final_state.get("messages", []):
        print(message)


# Main loop to accept user input from the terminal
if __name__ == "__main__":
    while True:
        query = input("\nPlease ask your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        run_workflow(query)


# import os
# import openai
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# import requests
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langgraph.graph import END, START, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode
# from typing import Dict, Any

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Data Loading and Processing

# def load_airtel_data() -> list:
#     """Scrape the Airtel website and extract relevant text data."""
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract text content from paragraphs
#     content = [p.get_text() for p in soup.find_all('p')]
#     return content

# def index_data(content: list) -> Any:
#     """Index the scraped data into a FAISS vector store."""
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(content, embeddings)
#     return vector_store

# # Search Functionality

# def search_vector_store(query: str, vector_store: Any) -> str:
#     """Perform a search in the FAISS vector store."""
#     results = vector_store.similarity_search(query, k=2)
#     return f"Retrieved the following info from Airtel site: {results}"

# # Tool Functionality

# def airtel_help_bot(query: str) -> Dict[str, Any]:
#     """Handle general queries related to Airtel."""
#     return {"response": f"Handled by Airtel Help Bot: {query}"}

# def retrieve_from_vectorstore(query: str, vector_store: Any) -> Dict[str, Any]:
#     """Retrieve information from the vector store based on the query."""
#     return {"response": search_vector_store(query, vector_store)}

# def sim_swap_workflow(query: str) -> Dict[str, Any]:
#     """Initiate the SIM swap process."""
#     user_phone = input("Please provide your phone number: ")
#     user_name = input("Please provide your name: ")
#     return {"response": f"SIM swap initiated for {user_name} ({user_phone})"}

# def plan_details_workflow(query: str) -> Dict[str, Any]:
#     """Provide details about the user's plan."""
#     user_phone = input("Please provide your phone number to check your plan: ")
#     return {"response": f"Your current plan for {user_phone} is: 'Unlimited Calls with 2GB/day'"}
# def retrieve_tool(query: str, vector_store: Any) -> str:
#     """Retrieve information from the vector store based on user query."""
#     return retrieve_from_vectorstore(query, vector_store)

# # Tool Creation

# def create_tools(vector_store: Any):
#     """Create tool nodes for the workflow."""
    
#     # Creating tools without passing any unsupported parameters
#     tools = [
#         ToolNode([airtel_help_bot]),  # Airtel help bot node
#         ToolNode([retrieve_tool]),  # Vector store retrieval node
#         ToolNode([sim_swap_workflow]),  # SIM swap workflow node
#         ToolNode([plan_details_workflow])  # Plan details workflow node
#     ]
    
#     return tools

# # Routing Logic

# def route_question(state: Dict[str, Any]) -> str:
#     """Route the user's question to the appropriate workflow."""
#     query = state.get("query", "").lower()
    
#     if "sim swap" in query:
#         return "sim_swap"
#     elif "plan" in query:
#         return "plan_details"
#     elif "airtel" in query:
#         return "airtel_help"
#     else:
#         return "retrieve"

# # Workflow Creation

# def create_workflow() -> Any:
#     """Initialize the workflow graph with tools and nodes."""
#     airtel_data = load_airtel_data()
#     vector_store = index_data(airtel_data)
    
#     tools = create_tools(vector_store)
    
#     # Initialize the workflow graph
#     workflow = StateGraph(MessagesState)
    
#     # Add nodes for each tool
#     for i, tool in enumerate(tools):
#         node_name = f"tool_node_{i}"
#         workflow.add_node(node_name, tool)
#         workflow.add_edge(node_name, END)

#     # Build conditional edges based on user queries
#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "airtel_help": "tool_node_0",
#             "retrieve": "tool_node_1",
#             "sim_swap": "tool_node_2",
#             "plan_details": "tool_node_3"
#         }
#     )
    
#     return workflow.compile()

# # Run Workflow

# def run_workflow(query: str):
#     """Run the workflow with the provided query."""
#     app = create_workflow()
    
#     # Create initial state
#     initial_state = {
#         "query": query,
#         "messages": []
#     }
    
#     # Run the workflow
#     final_state = app.invoke(initial_state)
    
#     # Print the messages
#     for message in final_state.get("messages", []):
#         print(message)

# # Main loop to accept user input from the terminal
# if __name__ == "__main__":
#     while True:
#         query = input("\nPlease ask your question (or type 'exit' to quit): ")
#         if query.lower() == 'exit':
#             break
#         run_workflow(query)


# import os
# import openai
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# import requests
# from langchain_openai import OpenAIEmbeddings  # Updated import
# from langchain_community.vectorstores import FAISS
# from langgraph.graph import END, START, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode
# from typing import Dict, Any

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")


# # Data Loading and Processing

# def load_airtel_data() -> list:
#     """Scrape the Airtel website and extract relevant text data."""
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract text content from paragraphs
#     content = [p.get_text() for p in soup.find_all('p')]
#     return content


# def index_data(content: list) -> Any:
#     """Index the scraped data into a FAISS vector store."""
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(content, embeddings)
#     return vector_store


# # Search Functionality

# def search_vector_store(query: str, vector_store: Any) -> str:
#     """Perform a search in the FAISS vector store."""
#     results = vector_store.similarity_search(query, k=2)
#     return f"Retrieved the following info from Airtel site: {results}"


# # Tool Functionality

# def airtel_help_bot(query: str) -> Dict[str, Any]:
#     return {"response": f"Handled by Airtel Help Bot: {query}"}


# def retrieve_from_vectorstore(query: str, vector_store: Any) -> Dict[str, Any]:
#     return {"response": search_vector_store(query, vector_store)}


# def sim_swap_workflow(query: str) -> Dict[str, Any]:
#     user_phone = input("Please provide your phone number: ")
#     user_name = input("Please provide your name: ")
#     return {"response": f"SIM swap initiated for {user_name} ({user_phone})"}


# def plan_details_workflow(query: str) -> Dict[str, Any]:
#     user_phone = input("Please provide your phone number to check your plan: ")
#     return {"response": f"Your current plan for {user_phone} is: 'Unlimited Calls with 2GB/day'"}


# # Tool Creation

# def create_tools(vector_store: Any):
#     """Create tool nodes for the workflow."""
#     return [
#         ToolNode(airtel_help_bot),
#         ToolNode(lambda q: retrieve_from_vectorstore(q, vector_store)),
#         ToolNode(sim_swap_workflow),
#         ToolNode(plan_details_workflow)
#     ]


# # Routing Logic

# def route_question(state: Dict[str, Any]) -> str:
#     query = state.get("query", "").lower()
#     if "sim swap" in query:
#         return "sim_swap"
#     elif "plan" in query:
#         return "plan_details"
#     elif "airtel" in query:
#         return "airtel_help"
#     else:
#         return "retrieve"


# # Workflow Creation

# def create_workflow() -> Any:
#     """Initialize the workflow graph with tools and nodes."""
#     airtel_data = load_airtel_data()
#     vector_store = index_data(airtel_data)
    
#     tools = create_tools(vector_store)
    
#     # Initialize the workflow graph
#     workflow = StateGraph(MessagesState)
    
#     # Add nodes for each tool
#     for node in tools:
#         workflow.add_node(node.name, node)
#         workflow.add_edge(node.name, END)

#     # Build conditional edges based on user queries
#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "airtel_help": "airtel_help",
#             "retrieve": "retrieve",
#             "sim_swap": "sim_swap",
#             "plan_details": "plan_details"
#         }
#     )
    
#     return workflow.compile()


# # Run Workflow

# def run_workflow(query: str):
#     app = create_workflow()
    
#     # Create initial state
#     initial_state = {
#         "query": query,
#         "messages": []
#     }
    
#     # Run the workflow
#     final_state = app.invoke(initial_state)
    
#     # Print the messages
#     for message in final_state.get("messages", []):
#         print(message)


# # Example queries to test the workflow
# if __name__ == "__main__":
#     print("Testing SIM swap workflow:")
#     run_workflow("Can I swap my sim card?")
    
#     print("\nTesting plan details workflow:")
#     run_workflow("Tell me about my plan")
    
#     print("\nTesting general Airtel query:")
#     run_workflow("General query on Airtel")


# import os
# import openai
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# import requests
# from langchain_openai import OpenAIEmbeddings  # Updated import
# from langchain_community.vectorstores import FAISS
# from langgraph.graph import END, START, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode
# from typing import Dict, Any, Tuple
# from langchain.tools import Tool

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize FAISS vector store for Airtel data
# def load_airtel_data():
#     # Airtel Website Scraping
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract relevant text data from Airtel webpage
#     content = [p.get_text() for p in soup.find_all('p')]
#     return content

# # Indexing Airtel website data into FAISS
# def index_data(content):
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(content, embeddings)
#     return vector_store

# # Perform a search in the FAISS vector store
# def search_vector_store(query, vector_store):
#     results = vector_store.similarity_search(query, k=2)
#     return results

# # Define tool functions with proper signatures
# def airtel_help_bot(query: str) -> Dict[str, Any]:
#     return {"response": "Handled by Airtel Help Bot: " + query}

# def retrieve_from_vectorstore(query: str, vector_store: Any) -> Dict[str, Any]:
#     results = search_vector_store(query, vector_store)
#     return {"response": f"Retrieved the following info from Airtel site: {results}"}

# def sim_swap_workflow(query: str) -> Dict[str, Any]:
#     # Note: In a real application, you might want to handle user input differently
#     user_phone = input("Please provide your phone number: ")
#     user_name = input("Please provide your name: ")
#     return {"response": f"SIM swap initiated for {user_name} ({user_phone})"}

# def plan_details_workflow(query: str) -> Dict[str, Any]:
#     user_phone = input("Please provide your phone number to check your plan: ")
#     return {"response": f"Your current plan for {user_phone} is: 'Unlimited Calls with 2GB/day'"}

# # Create Tool objects
# def create_tools(vector_store):
#     tools = [
#         Tool(
#             name="airtel_help",
#             func=airtel_help_bot,
#             description="Handles general Airtel help queries"
#         ),
#         Tool(
#             name="retrieve",
#             func=lambda q: retrieve_from_vectorstore(q, vector_store),
#             description="Retrieves information from Airtel knowledge base"
#         ),
#         Tool(
#             name="sim_swap",
#             func=sim_swap_workflow,
#             description="Handles SIM swap requests"
#         ),
#         Tool(
#             name="plan_details",
#             func=plan_details_workflow,
#             description="Provides plan details for a given number"
#         )
#     ]
#     return tools

# # Define routing logic
# def route_question(state: Dict[str, Any]) -> str:
#     query = state.get("query", "").lower()
#     if "sim swap" in query:
#         return "sim_swap"
#     elif "plan" in query:
#         return "plan_details"
#     elif "airtel" in query:
#         return "airtel_help"
#     else:
#         return "retrieve"

# # Process the tool's response
# def process_response(state: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
#     if "messages" not in state:
#         state["messages"] = []
    
#     if isinstance(response, dict) and "response" in response:
#         state["messages"].append(response["response"])
#     else:
#         state["messages"].append(str(response))
    
#     return state

# # Initialize the workflow
# def create_workflow():
#     # Load Airtel data and initialize vector store
#     airtel_data = load_airtel_data()
#     vector_store = index_data(airtel_data)
    
#     # Create tools
#     tools = create_tools(vector_store)
    
#     # Initialize the workflow graph
#     workflow = StateGraph(MessagesState)
    
#     # Create tool nodes
#     for tool in tools:
#         node = ToolNode([tool])
#         workflow.add_node(tool.name, node)
#         # Add processing step for the response
#         workflow.add_edge(tool.name, END)
    
#     # Build conditional edges based on user queries
#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "airtel_help": "airtel_help",
#             "retrieve": "retrieve",
#             "sim_swap": "sim_swap",
#             "plan_details": "plan_details"
#         }
#     )
    
#     return workflow.compile()

# # Function to run the workflow with a given query
# def run_workflow(query: str):
#     app = create_workflow()
    
#     # Create initial state
#     initial_state = {
#         "query": query,
#         "messages": []
#     }
    
#     # Get the config
#     config = {"recursion_limit": 10}
    
#     # Run the workflow
#     final_state = app.invoke(initial_state, config)
    
#     # Print the messages
#     for message in final_state.get("messages", []):
#         print(message)

# # Example queries to test the workflow
# if __name__ == "__main__":
#     print("Testing SIM swap workflow:")
#     run_workflow("Can I swap my sim card?")
    
#     print("\nTesting plan details workflow:")
#     run_workflow("Tell me about my plan")
    
#     print("\nTesting general Airtel query:")
#     run_workflow("General query on Airtel")

# import os
# import openai
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# import requests
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langgraph.graph import END, START, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode
# from typing import Dict, Any, List, Tuple
# from langchain.tools import Tool
# from operator import itemgetter

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize FAISS vector store for Airtel data
# def load_airtel_data():
#     # Airtel Website Scraping
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract relevant text data from Airtel webpage
#     content = [p.get_text() for p in soup.find_all('p')]
#     return content

# # Indexing Airtel website data into FAISS
# def index_data(content):
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(content, embeddings)
#     return vector_store

# # Perform a search in the FAISS vector store
# def search_vector_store(query, vector_store):
#     results = vector_store.similarity_search(query, k=2)
#     return results

# # Define tool functions with proper signatures
# def airtel_help_bot(query: str) -> Dict[str, Any]:
#     return {"response": "Handled by Airtel Help Bot: " + query}

# def retrieve_from_vectorstore(query: str, vector_store: Any) -> Dict[str, Any]:
#     results = search_vector_store(query, vector_store)
#     return {"response": f"Retrieved the following info from Airtel site: {results}"}

# def sim_swap_workflow(query: str) -> Dict[str, Any]:
#     # Note: In a real application, you might want to handle user input differently
#     user_phone = input("Please provide your phone number: ")
#     user_name = input("Please provide your name: ")
#     return {"response": f"SIM swap initiated for {user_name} ({user_phone})"}

# def plan_details_workflow(query: str) -> Dict[str, Any]:
#     user_phone = input("Please provide your phone number to check your plan: ")
#     return {"response": f"Your current plan for {user_phone} is: 'Unlimited Calls with 2GB/day'"}

# # Create Tool objects
# def create_tools(vector_store):
#     tools = [
#         Tool(
#             name="airtel_help",
#             func=airtel_help_bot,
#             description="Handles general Airtel help queries"
#         ),
#         Tool(
#             name="retrieve",
#             func=lambda q: retrieve_from_vectorstore(q, vector_store),
#             description="Retrieves information from Airtel knowledge base"
#         ),
#         Tool(
#             name="sim_swap",
#             func=sim_swap_workflow,
#             description="Handles SIM swap requests"
#         ),
#         Tool(
#             name="plan_details",
#             func=plan_details_workflow,
#             description="Provides plan details for a given number"
#         )
#     ]
#     return tools

# # Define routing logic
# def route_question(state: Dict[str, Any]) -> Tuple[str, List[str]]:
#     query = state["query"]
#     if "sim swap" in query.lower():
#         return "sim_swap", ["sim_swap"]
#     elif "plan" in query.lower():
#         return "plan_details", ["plan_details"]
#     elif "airtel" in query.lower():
#         return "airtel_help", ["airtel_help"]
#     else:
#         return "retrieve", ["retrieve"]

# # Initialize the workflow
# def create_workflow():
#     # Load Airtel data and initialize vector store
#     airtel_data = load_airtel_data()
#     vector_store = index_data(airtel_data)
    
#     # Create tools
#     tools = create_tools(vector_store)
    
#     # Initialize the workflow graph
#     workflow = StateGraph(MessagesState)
    
#     # Create tool nodes
#     for tool in tools:
#         workflow.add_node(tool.name, ToolNode([tool]))
    
#     # Build conditional edges based on user queries
#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "airtel_help": "airtel_help",
#             "retrieve": "retrieve",
#             "sim_swap": "sim_swap",
#             "plan_details": "plan_details"
#         }
#     )
    
#     # Add end edges for each task
#     for tool in tools:
#         workflow.add_edge(tool.name, END)
    
#     # Compile the workflow
#     app = workflow.compile()
    
#     return app

# # Function to run the workflow with a given query
# def run_workflow(query: str):
#     app = create_workflow()
    
#     # Create initial state
#     initial_state = {"query": query, "messages": []}
    
#     # Get the config
#     config = {"recursion_limit": 10}
    
#     # Run the workflow
#     for output in app.stream(initial_state, config):
#         if "messages" in output:
#             for message in output["messages"]:
#                 if isinstance(message, dict) and "response" in message:
#                     print(message["response"])
#                 else:
#                     print(message)

# # Example queries to test the workflow
# if __name__ == "__main__":
#     print("Testing SIM swap workflow:")
#     run_workflow("Can I swap my sim card?")
    
#     print("\nTesting plan details workflow:")
#     run_workflow("Tell me about my plan")
    
#     print("\nTesting general Airtel query:")
#     run_workflow("General query on Airtel")
# import os
# import openai
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# import requests
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langgraph.graph import END, START, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode
# from typing import Dict, Any
# from langchain.tools import Tool

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize FAISS vector store for Airtel data
# def load_airtel_data():
#     # Airtel Website Scraping
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract relevant text data from Airtel webpage
#     content = [p.get_text() for p in soup.find_all('p')]
#     return content

# # Indexing Airtel website data into FAISS
# def index_data(content):
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(content, embeddings)
#     return vector_store

# # Perform a search in the FAISS vector store
# def search_vector_store(query, vector_store):
#     results = vector_store.similarity_search(query, k=2)
#     return results

# # Define tool functions with proper signatures
# def airtel_help_bot(query: str) -> str:
#     return "Handled by Airtel Help Bot: " + query

# def retrieve_from_vectorstore(query: str, vector_store: Any) -> str:
#     results = search_vector_store(query, vector_store)
#     return f"Retrieved the following info from Airtel site: {results}"

# def sim_swap_workflow(query: str) -> str:
#     # Note: In a real application, you might want to handle user input differently
#     user_phone = input("Please provide your phone number: ")
#     user_name = input("Please provide your name: ")
#     return f"SIM swap initiated for {user_name} ({user_phone})"

# def plan_details_workflow(query: str) -> str:
#     user_phone = input("Please provide your phone number to check your plan: ")
#     return f"Your current plan for {user_phone} is: 'Unlimited Calls with 2GB/day'"

# # Create Tool objects
# def create_tools(vector_store):
#     tools = [
#         Tool(
#             name="airtel_help",
#             func=airtel_help_bot,
#             description="Handles general Airtel help queries"
#         ),
#         Tool(
#             name="retrieve",
#             func=lambda q: retrieve_from_vectorstore(q, vector_store),
#             description="Retrieves information from Airtel knowledge base"
#         ),
#         Tool(
#             name="sim_swap",
#             func=sim_swap_workflow,
#             description="Handles SIM swap requests"
#         ),
#         Tool(
#             name="plan_details",
#             func=plan_details_workflow,
#             description="Provides plan details for a given number"
#         )
#     ]
#     return tools

# # Define routing logic
# def route_question(state: Dict[str, Any], query: str) -> str:
#     if "sim swap" in query.lower():
#         return "sim_swap"
#     elif "plan" in query.lower():
#         return "plan_details"
#     elif "airtel" in query.lower():
#         return "airtel_help"
#     else:
#         return "retrieve"

# # Initialize the workflow
# def create_workflow():
#     # Load Airtel data and initialize vector store
#     airtel_data = load_airtel_data()
#     vector_store = index_data(airtel_data)
    
#     # Create tools
#     tools = create_tools(vector_store)
    
#     # Initialize the workflow graph
#     workflow = StateGraph(MessagesState)
    
#     # Create tool nodes
#     for tool in tools:
#         workflow.add_node(tool.name, ToolNode([tool]))
    
#     # Build conditional edges based on user queries
#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "airtel_help": "airtel_help",
#             "retrieve": "retrieve",
#             "sim_swap": "sim_swap",
#             "plan_details": "plan_details"
#         }
#     )
    
#     # Add end edges for each task
#     for tool in tools:
#         workflow.add_edge(tool.name, END)
    
#     return workflow.compile()

# # Function to run the workflow with a given query
# def run_workflow(query: str):
#     app = create_workflow()
#     return app.run({"query": query})

# # Example queries to test the workflow
# if __name__ == "__main__":
#     print(run_workflow("Can I swap my sim card?"))  # SIM swap workflow
#     print(run_workflow("Tell me about my plan"))    # Plan details workflow
#     print(run_workflow("General query on Airtel"))  # Vector store retrieval
# import os
# import openai
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# import requests
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langgraph.graph import END, START, StateGraph, MessagesState
# from langgraph.prebuilt import ToolNode

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize FAISS vector store for Airtel data
# def load_airtel_data():
#     # Airtel Website Scraping
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract relevant text data from Airtel webpage
#     content = [p.get_text() for p in soup.find_all('p')]
#     return content

# # Indexing Airtel website data into FAISS
# def index_data(content):
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(content, embeddings)
#     return vector_store

# # Perform a search in the FAISS vector store
# def search_vector_store(query, vector_store):
#     results = vector_store.similarity_search(query, k=2)
#     return results

# # Define tool nodes for each task
# def airtel_help_bot(state, query):
#     return "Handled by Airtel Help Bot: " + query

# def retrieve_from_vectorstore(state, query, vector_store):
#     results = search_vector_store(query, vector_store)
#     return f"Retrieved the following info from Airtel site: {results}"

# def sim_swap_workflow(state, query):
#     state['user_phone'] = input("Please provide your phone number: ")
#     state['user_name'] = input("Please provide your name: ")
#     return f"SIM swap initiated for {state['user_name']} ({state['user_phone']})"

# def plan_details_workflow(state, query):
#     state['user_phone'] = input("Please provide your phone number to check your plan: ")
#     return f"Your current plan for {state['user_phone']} is: 'Unlimited Calls with 2GB/day'"

# # Define routing logic
# def route_question(state, query):
#     if "sim swap" in query:
#         return "sim_swap"
#     elif "plan" in query:
#         return "plan_details"
#     elif "airtel" in query:
#         return "airtel_help"
#     else:
#         return "retrieve"

# # Initialize the workflow graph
# workflow = StateGraph(MessagesState)

# # Load Airtel data and initialize vector store
# airtel_data = load_airtel_data()
# vector_store = index_data(airtel_data)

# # Define nodes for each task, wrapping functions in ToolNode
# workflow.add_node("airtel_help", ToolNode(airtel_help_bot))
# workflow.add_node("retrieve", ToolNode(lambda state, query: retrieve_from_vectorstore(state, query, vector_store)))
# workflow.add_node("sim_swap", ToolNode(sim_swap_workflow))
# workflow.add_node("plan_details", ToolNode(plan_details_workflow))

# # Build conditional edges based on user queries
# workflow.add_conditional_edges(
#     START,
#     route_question,
#     {
#         "airtel_help": "airtel_help",
#         "retrieve": "retrieve",
#         "sim_swap": "sim_swap",
#         "plan_details": "plan_details"
#     },
# )

# # End edges for each task
# workflow.add_edge("retrieve", END)
# workflow.add_edge("airtel_help", END)
# workflow.add_edge("sim_swap", END)
# workflow.add_edge("plan_details", END)

# # Compile the workflow
# app = workflow.compile()

# # Function to run the workflow with a given query
# def run_workflow(query):
#     return app.run(query=query)

# # Example queries to test the workflow
# if __name__ == "__main__":
#     print(run_workflow("Can I swap my sim card?"))  # SIM swap workflow
#     print(run_workflow("Tell me about my plan"))    # Plan details workflow
#     print(run_workflow("General query on Airtel"))  # Vector store retrieval



# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize FAISS vector store for Airtel data
# def load_airtel_data():
#     # Airtel Website Scraping
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract relevant text data from Airtel webpage
#     content = [p.get_text() for p in soup.find_all('p')]
#     return content

# # Indexing Airtel website data into FAISS
# def index_data(content):
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(content, embeddings)
#     return vector_store

# # Perform a search in the FAISS vector store
# def search_vector_store(query, vector_store):
#     results = vector_store.similarity_search(query, k=2)
#     return results

# # Define tool nodes for each task
# def airtel_help_bot(state, query):
#     return "Handled by Airtel Help Bot: " + query

# def retrieve_from_vectorstore(state, query, vector_store):
#     results = search_vector_store(query, vector_store)
#     return f"Retrieved the following info from Airtel site: {results}"

# def sim_swap_workflow(state, query):
#     state['user_phone'] = input("Please provide your phone number: ")
#     state['user_name'] = input("Please provide your name: ")
#     return f"SIM swap initiated for {state['user_name']} ({state['user_phone']})"

# def plan_details_workflow(state, query):
#     state['user_phone'] = input("Please provide your phone number to check your plan: ")
#     return f"Your current plan for {state['user_phone']} is: 'Unlimited Calls with 2GB/day'"

# # Define routing logic
# def route_question(state, query):
#     if "sim swap" in query:
#         return "sim_swap"
#     elif "plan" in query:
#         return "plan_details"
#     elif "airtel" in query:
#         return "airtel_help"
#     else:
#         return "retrieve"

# # Initialize the workflow graph
# workflow = StateGraph(MessagesState, saver=MemorySaver())

# # Load Airtel data and initialize vector store
# airtel_data = load_airtel_data()
# vector_store = index_data(airtel_data)

# # Define nodes for each task
# workflow.add_node("airtel_help", ToolNode(airtel_help_bot))
# workflow.add_node("retrieve", ToolNode(lambda state, query: retrieve_from_vectorstore(state, query, vector_store)))
# workflow.add_node("sim_swap", ToolNode(sim_swap_workflow))
# workflow.add_node("plan_details", ToolNode(plan_details_workflow))

# # Build conditional edges based on user queries
# workflow.add_conditional_edges(
#     START,
#     route_question,
#     {
#         "airtel_help": "airtel_help",
#         "vectorstore": "retrieve",
#         "sim_swap": "sim_swap",
#         "plan_details": "plan_details"
#     },
# )

# # End edges for each task
# workflow.add_edge("retrieve", END)
# workflow.add_edge("airtel_help", END)
# workflow.add_edge("sim_swap", END)
# workflow.add_edge("plan_details", END)

# # Compile the workflow
# app = workflow.compile()

# # Function to run the workflow with a given query
# def run_workflow(query):
#     return app.run(query=query)

# # Example queries to test the workflow
# print(run_workflow("Can I swap my sim card?"))  # SIM swap workflow
# print(run_workflow("Tell me about my plan"))    # Plan details workflow
# print(run_workflow("General query on Airtel"))  # Vector store retrieval







# # import os
# # import requests
# # from bs4 import BeautifulSoup
# # from langchain_community.llms import OpenAI
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import OpenAIEmbeddings

# # from langchain.chains import LLMChain

# # from langchain.prompts import PromptTemplate

# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # # from langgraph import Graph
# # from langgraph.graph import GraphNode
# # from dotenv import load_dotenv

# # # Load API Key from .env file
# # load_dotenv()
# # openai_api_key = os.getenv("OPENAI_API_KEY")

# # # Initialize OpenAI LLM
# # llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

# # # 1. Scraping Airtel Thanks App Data
# # def scrape_airtel_thanks_page():
# #     url = "https://www.airtel.in/airtel-thanks-app"
# #     response = requests.get(url)
# #     soup = BeautifulSoup(response.content, 'html.parser')
    
# #     # Extract the main content text (you can refine this to extract specific sections)
# #     page_text = " ".join([p.text for p in soup.find_all('p')])
# #     return page_text

# # airtel_thanks_data = scrape_airtel_thanks_page()

# # # 2. Building Vector DB for Knowledge Retrieval
# # def build_vector_db(data: str):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# #     docs = text_splitter.split_text(data)
# #     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# #     vector_db = FAISS.from_texts(docs, embeddings)
# #     return vector_db

# # # Build vector DB from the scraped Airtel Thanks data
# # vector_db = build_vector_db(airtel_thanks_data)

# # # Knowledge Base Agent (RetrievalQA) for Airtel Thanks App content
# # qa_chain = LLMChain(llm=llm, retriever=vector_db.as_retriever())

# # # 3. Prompts for each task (basic help, fallback, sim swap, plan details)

# # basic_help_prompt = PromptTemplate(
# #     input_variables=["question"],
# #     template="You are Airtel's help bot. Answer the user's question: {question}"
# # )
# # help_bot_chain = LLMChain(llm=llm, prompt=basic_help_prompt)

# # fallback_prompt = PromptTemplate(
# #     input_variables=["question"],
# #     template="This question doesn't seem related to Airtel services: {question}"
# # )
# # fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)

# # sim_swap_prompt = PromptTemplate(
# #     input_variables=["question"],
# #     template="The user wants to swap their SIM card. Ask for phone number and name: {question}"
# # )
# # sim_swap_chain = LLMChain(llm=llm, prompt=sim_swap_prompt)

# # plan_details_prompt = PromptTemplate(
# #     input_variables=["question"],
# #     template="""
# #     The user is asking about their active plan. First, ask for their phone number, then display this standard plan:
# #     'Your current plan is INR 509 with 6GB of data and unlimited calling, and 1000 messages.'
# #     Question: {question}
# #     """
# # )
# # plan_details_chain = LLMChain(llm=llm, prompt=plan_details_prompt)

# # # 4. Building LangGraph with the multi-agent workflow
# # graph = Graph()

# # # Adding nodes (Agents) to the Graph
# # help_node = GraphNode(name="help_bot", chain=help_bot_chain)
# # qa_node = GraphNode(name="knowledge_base", chain=qa_chain)
# # fallback_node = GraphNode(name="fallback", chain=fallback_chain)
# # sim_swap_node = GraphNode(name="sim_swap", chain=sim_swap_chain)
# # plan_details_node = GraphNode(name="plan_details", chain=plan_details_chain)

# # graph.add_node(help_node)
# # graph.add_node(qa_node)
# # graph.add_node(fallback_node)
# # graph.add_node(sim_swap_node)
# # graph.add_node(plan_details_node)

# # # --- Define the Edges and Conditional Logic ---

# # # Add edge from Help Bot to Knowledge Base (for queries related to Airtel Thanks App)
# # graph.add_conditional_edge(
# #     source=help_node,
# #     target=qa_node,
# #     condition=lambda question: "airtel thanks" in question.lower()
# # )

# # # Add edge from Help Bot to SIM Swap if the query contains "sim swap"
# # graph.add_conditional_edge(
# #     source=help_node,
# #     target=sim_swap_node,
# #     condition=lambda question: "sim swap" in question.lower()
# # )

# # # Add edge from Help Bot to Plan Details if the query contains "plan"
# # graph.add_conditional_edge(
# #     source=help_node,
# #     target=plan_details_node,
# #     condition=lambda question: "plan" in question.lower()
# # )

# # # Fallback to fallback_node if none of the conditions match
# # graph.add_edge(source=help_node, target=fallback_node)

# # # --- Process the User Query through the Graph ---
# # def process_query(question: str):
# #     # Start the graph traversal at the help bot node
# #     response = graph.run(question=question, starting_node=help_node)
# #     return response

# # # Example usage
# # if __name__ == "__main__":
# #     user_query = input("Ask your question: ")
# #     response = process_query(user_query)
# #     print(f"Response: {response}")
