import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langgraph.graph import END, StateGraph, START
from embedding import get_vector_store  # Import the vector store function from embedding.py

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model for LLM routing
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Initialize Pinecone vector store
vector_store = get_vector_store()  # Now this correctly initializes the vector store

# # Helper Functions
# def retrieve_from_knowledge_base(state):
#     """Retrieve answers from the knowledge base (Vectorstore)."""
#     print("--- Knowledge Base Retrieval ---")
#     question = state["question"]
#     documents = vector_store.similarity_search(question)
#     return {"documents": documents, "question": question}
# Modify retrieve_from_knowledge_base to format the knowledge base responses
def retrieve_from_knowledge_base(state):
    """Retrieve answers from the knowledge base (Vectorstore) and format the response."""
    print("--- Knowledge Base Retrieval ---")
    question = state["question"]
    documents = vector_store.similarity_search(question)
    
    if not documents:
        return {"generation": "No relevant information found in the knowledge base.", "question": question}
    
    # Format the first relevant document
    relevant_info = documents[0].page_content[:500]  # Only show the first 500 characters of the first document
    return {"generation": f"Here is what I found: {relevant_info}", "question": question}
# def sim_swap_workflow(state):
#     """Handle SIM swap queries."""
#     print("--- SIM Swap Workflow ---")
#     if not state.get("name"):
#         state["name"] = input("Bot: Please provide your full name for the SIM swap.\nYou: ")
#     if not state.get("phone_number"):
#         state["phone_number"] = input("Bot: Please provide your phone number for the SIM swap.\nYou: ")
    
#     return {"sim_swap_details": f"SIM swap initiated for {state['name']} with phone number {state['phone_number']}.", "question": state["question"]}
def sim_swap_workflow(state):
    """Trigger SIM swap workflow in a step-by-step manner."""
    # Check the current state of the workflow and prompt accordingly
    print("---sim workflow--- ")
    question = state["question"]
    print("We need your full name and phone number for sim swap")
    # Check if the workflow has been initiated, ask for the next step
    name = input("Please provide your full name? ")
    phone_number = input("Please provide your phone number? ")

    return {"sim_swap_details": f"Dear {name}, Your phone number -{phone_number} is successfully submitted! We will contact you soon.", "question": question}

# # Recharge Plan Workflow
# def recharge_plan_workflow(state):
#     """Handle recharge plan queries."""
#     print("--- Recharge Plan Workflow ---")
#     if state.get("step") == "ask_phone_number":
#         return {"generation": "Please provide your phone number for recharge plans.", "step": "collect_phone_number"}

#     elif state.get("step") == "collect_phone_number":
#         phone_number = state["question"]
#         url = f"https://www.airtel.in/recharge/prepaid/?siNumber={phone_number}"
#         response = requests.get(url)
#         if response.status_code == 200:
#             available_plans = response.text
#             return {"generation": f"Here are the available plans for {phone_number}: {available_plans}", "step": "completed"}
#         else:
#             return {"generation": "Sorry, there was an issue retrieving the plans. Please try again.", "step": "completed"}

#     elif "recharge plan" in state["question"].lower() and not state.get("step"):
#         return {"generation": "Please provide your phone number to retrieve available recharge plans.", "step": "ask_phone_number"}

#     return state

def recharge_plan_workflow(state):
    """Trigger Recharge Plan Workflow."""
    question = state["question"]

    print("--Triggered Recharge Plan Workflow--")
    user_phone_number = input("Please enter your phone number to check your current plan: ")

    # Hardcoded response for current plan
    current_plan = {
        "Rs 199": "This plan offers unlimited calls, 1.5 GB daily data, and 100 SMS per day for 28 days.",
        "Rs 299": "This plan includes unlimited calls, 2 GB daily data, and 100 SMS per day for 28 days.",
        "Rs 449": "This plan provides unlimited calls, 3 GB daily data, and 100 SMS per day for 56 days.",
        "Rs 599": "This plan gives unlimited calls, 4 GB daily data, and 100 SMS per day for 84 days."
    }

    # Here you can simulate getting the plan based on the phone number
    # In this example, we'll assume the user has the Rs 299 plan for simplicity.
    user_current_plan = "Rs 299"

    # Retrieve the description of the current plan
    plan_description = current_plan.get(user_current_plan, "Sorry, no details available for your current plan.")

    return {
        "generation": f"Your phone number {user_phone_number} has the current plan of {user_current_plan}. {plan_description}",
        "question": question
    }




# Fallback Agent
# def fallback_agent(state):
#     """Handle irrelevant queries."""
#     return {"fallback": "Sorry, your question is not relevant to Airtel services. Please ask something related to Airtel."}

def fallback_agent(state):
    """
    Return fallback messsage

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains fallback mechanism
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    fallback_message = "I'm sorry, but I couldn't find relevant information for your query. Could you please clarify or ask about Airtel services, plans, or issues? I'm here to help!"

    return {"documents": fallback_message, "question": question}
# Define routing logic via LLM
# class RouteQuery(BaseModel):
#     datasource: Literal["vectorstore", "workflow_sim_swap", "recharge_plan_workflow", "fallback_agent"] = Field(
#         ..., description="Route to vectorstore, workflows, or fallback."
#     )



# Routing system for LLM
system = """
You are an expert AI assistant specializing in Airtel-related queries. Your main responsibility is to accurately route user inquiries to the appropriate resources, ensuring an optimal user experience. Use the following routing guidelines:

1. **General Airtel-Related Queries**: 
   - For inquiries about Airtel's services, network issues, or company-related information, route to **knowledge_based_llm**. 
   - Example: "What are the available plans for Airtel?" or "I'm experiencing network issues."

2. **SIM Swap Queries**: 
   - If the user mentions anything related to SIM swaps, guide them through the process by routing to **swap_sim_workflow**.
   - Example: "How do I swap my SIM?" or "I need to change my SIM."

3. **Recharge or Plan Details**: 
   - Direct questions about recharges, balance inquiries, or plan information to **current_recharge_plan_workflow**.
   - Example: "What are the current recharge options?" or "How can I check my balance?"

4. **Telecom Services and Casual Conversations**: 
   - Handle casual chats, greetings, or general telecom discussions with **basic_llm**. 
   - Example: "Hi, can you tell me about DTH services?" or "What’s new in telecom?"

5. **Fallback for Ambiguous or Irrelevant Queries**: 
   - For questions that are unclear, unrelated to Airtel or telecom, or consist of random content, route them to **fallback_message**.
   - Example: "What is the capital of Japan?" or "Tell me a joke."

Always ensure to accurately determine the nature of the inquiry to provide the most relevant and helpful response. If unsure, it's better to direct users to the fallback mechanism for further assistance.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# structured_llm_router = llm.with_structured_output(RouteQuery)
# question_router = route_prompt | structured_llm_router

# def route_to_service(question: str) -> str:  # Change to return a string
#     """Route the question to the appropriate service."""
#     question_lower = question.lower()

#     if "sim swap" in question_lower:
#         return "workflow_sim_swap"  # Return string directly
#     elif "recharge" in question_lower or "plan" in question_lower:
#         return "recharge_plan_workflow"  # Return string directly
#     else:
#         return "vectorstore"  # Fallback to vectorstore for general queries
def route_to_service(question: str) -> str:
    """Route the question to the appropriate service."""
    question_lower = question.lower()

    if "sim swap" in question_lower:
        return "workflow_sim_swap"
    elif "recharge" in question_lower or "plan" in question_lower:
        return "recharge_plan_workflow"
    # Detect irrelevant queries with a fallback condition
    elif "airtel" not in question_lower:
        return "fallback_agent"
    else:
        return "vectorstore"

# Update your routing system
def question_router(state):
    """Route the question based on content."""
    question = state["question"]
    return route_to_service(question)  # Return the string directly

from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


# Build the main workflow
workflow = StateGraph(GraphState)
workflow.add_node("vectorstore", retrieve_from_knowledge_base)
workflow.add_node("workflow_sim_swap", sim_swap_workflow)
workflow.add_node("recharge_plan_workflow", recharge_plan_workflow)
workflow.add_node("fallback_agent", fallback_agent)

# Update the edges to use the new routing function
workflow.add_conditional_edges(
    START,
    question_router,
    {
        "vectorstore": "vectorstore",
        "workflow_sim_swap": "workflow_sim_swap",
        "recharge_plan_workflow": "recharge_plan_workflow",
        "fallback_agent": "fallback_agent"
    }
)

workflow.add_edge("vectorstore", END)
workflow.add_edge("workflow_sim_swap", END)
workflow.add_edge("recharge_plan_workflow", END)
workflow.add_edge("fallback_agent", END)

# Compile the workflow
app = workflow.compile()

# Chatbot runner function
def run_chatbot():
    state = {"question": "", "step": None, "name": None, "phone_number": None}

    while True:
        state["question"] = input("You: ")

        if state["question"].lower() in ["quit", "exit", "stop"]:
            print("Bot: Goodbye!")
            break

        outputs = app.stream(state)

        for output in outputs:
            for key, value in output.items():
                if "generation" in value:
                    print(f"Bot: {value['generation']}")
                elif "documents" in value:
                    for doc in value["documents"]:
                        print(f"Bot: {doc}")
                elif "sim_swap_details" in value:
                    print(f"Bot: {value['sim_swap_details']}")
                elif "fallback" in value:
                    print(f"Bot: {value['fallback']}")

        print("\n---\n")

# Run the chatbot
run_chatbot()









# import os
# import requests
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# from pinecone import Pinecone as PineconeClient, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field
# from typing import Literal
# from langgraph.graph import END, StateGraph, START
# from embedding import vector_store 
# # Load environment variables
# load_dotenv()

# # API Keys
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize Pinecone client
# def init_pinecone():
#     pc = PineconeClient(api_key=PINECONE_API_KEY)
#     return pc

# # Embedding model for VectorDB
# def get_embeddings():
#     return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# # Scrape Airtel Thanks App Data
# def scrape_airtel_thanks_data():
#     url = "https://www.airtel.in/airtel-thanks-app"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     data = [section.get_text(separator=" ", strip=True) for section in soup.find_all("section") if section.get_text(separator=" ", strip=True)]
#     return data

# # Store scraped data in Pinecone
# def store_in_pinecone(data):
#     pc = init_pinecone()
#     index_name = "kukkur"
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=1536,
#             metric='euclidean',
#             spec=ServerlessSpec(cloud='aws', region='us-east-1')
#         )
#     embeddings = get_embeddings()
#     vector_db = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    
#     for chunk in data:
#         embedding = embeddings.embed_query(chunk)
#         metadata = {"text": chunk}
#         vector_db.add_texts([chunk], embeddings=[embedding], metadatas=[metadata])
#     print("Data has been successfully embedded and stored in Pinecone!")

# # Retrieve from Pinecone
# def retrieve(state):
#     print("---RETRIEVE---")
#     question = state["question"]
#     documents = vector_store.similarity_search(question)
#     return {"documents": documents, "question": question}

# # Define Router Model for Routing Queries
# class RouteQuery(BaseModel):
#     datasource: Literal["vectorstore", "workflow_sim_swap", "recharge_plan_workflow", "conversation"] = Field(
#         ..., description="Route to vectorstore, workflows, or fallback conversation."
#     )

# # LLM setup for routing
# llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
# structured_llm_router = llm.with_structured_output(RouteQuery)

# # Define routing logic
# system = """
# You are an expert at handling Airtel-related queries and are responsible for directing them to the appropriate resource.
# Route queries about Airtel’s general services, network issues, or company-related information to the vectorstore,
# SIM swap-related queries to the SIM swap workflow, recharge or plan details to the recharge plan workflow,
# and if the query doesn't fit these categories or is ambiguous, rely on a fallback conversation model using the LLM for further assistance.
# """

# route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
# question_router = route_prompt | structured_llm_router

# # SIM Swap Workflow
# def sim_swap_workflow(state):
#     print("---SIM Swap Workflow---")
#     name = input("Please provide your full name for the SIM swap: ")
#     phone_number = input("Please provide your phone number for SIM swap: ")
#     return {"sim_swap_details": f"Name: {name}, Phone: {phone_number}", "question": state["question"]}

# # Recharge Plan Workflow
# def recharge_plan_workflow(state):
#     if state.get("step") == "ask_phone_number":
#         return {"generation": "Please provide your phone number for recharge plans.", "step": "collect_phone_number"}
#     elif state.get("step") == "collect_phone_number":
#         phone_number = state["question"]
#         # Simulate API response for plan
#         available_plans = f"Sample Plan for {phone_number}: ₹199 - 1GB/day, 28 days validity."
#         return {"generation": f"Here are the available plans: {available_plans}", "step": "completed"}
#     elif "recharge plan" in state["question"].lower():
#         return {"generation": "Please provide your phone number to retrieve available recharge plans.", "step": "ask_phone_number"}
#     return state

# # Fallback conversation handler
# def fallback_conversation(state):
#     return {"generation": "Please ask something relevant to Airtel services.", "step": "completed"}

# # Graph workflow for routing and handling user queries
# workflow = StateGraph()
# workflow.add_node("vectorstore", retrieve)
# workflow.add_node("workflow_sim_swap", sim_swap_workflow)
# workflow.add_node("recharge_plan_workflow", recharge_plan_workflow)
# workflow.add_node("conversation", fallback_conversation)

# workflow.add_conditional_edges(
#     START,
#     question_router.invoke,
#     {
#         "vectorstore": "vectorstore",
#         "workflow_sim_swap": "workflow_sim_swap",
#         "recharge_plan_workflow": "recharge_plan_workflow",
#         "conversation": "conversation"
#     }
# )
# workflow.add_edge("vectorstore", END)
# workflow.add_edge("workflow_sim_swap", END)
# workflow.add_edge("recharge_plan_workflow", END)
# workflow.add_edge("conversation", END)

# # Compile workflow
# app = workflow.compile()

# # Chatbot runner function
# def run_chatbot():
#     state = {"question": "", "step": None}
#     while True:
#         state["question"] = input("You: ")
#         if state["question"].lower() in ["quit", "exit"]:
#             print("Bot: Goodbye!")
#             break
#         outputs = app.stream(state)
#         for output in outputs:
#             for key, value in output.items():
#                 print(f"Bot: {value}")
#                 if value.get("step") == "completed":
#                     print("Bot: Thank you! The workflow is now complete.")
#                     break
#         print("\n---\n")

# # Start the chatbot
# run_chatbot()

# # import os
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import Pinecone
# # from langchain_community.embeddings import OpenAIEmbeddings
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_openai import ChatOpenAI
# # from langchain_core.documents import Document
# # from pydantic import BaseModel, Field
# # from typing import List, Literal
# # from dotenv import load_dotenv
# # import requests

# # # Set environment variables for Pinecone and OpenAI API keys
# # # os.environ['OPENAI_API_KEY'] = "sk-proj-oC6hudfyegfu3ifkekjfhjerhfuheufhkjwechu3fuywekwehjhefbuyegyheuchyuecyukiehcytekdjcuehfckjervyuejkfy3ui23eu23u836723yryd87efjyetfy4hyfgejfgty34fhjfhyuefuefhjfegE4UTanNuk9TB1sR"
# # # os.environ['PINECONE_API_KEY'] = "your-pinecone-api-key"
# # # Load environment variables from .env file
# # load_dotenv()

# # # Set your OpenAI API key
# # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # # Initialize Pinecone
# # import pinecone
# # pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1-aws")

# # # Set up embeddings and vector store
# # embedding_model = OpenAIEmbeddings()
# # index_name = "kukkur"

# # # Use Pinecone as vector store
# # vector_store = Pinecone.from_existing_index(
# #     index_name=index_name,
# #     embedding=embedding_model
# # )

# # # Retrieve documents function
# # def retrieve(state):
# #     """
# #     Retrieve documents

# #     Args:
# #         state (dict): The current graph state

# #     Returns:
# #         state (dict): New key added to state, documents, that contains retrieved documents
# #     """
# #     print("---RETRIEVE---")
# #     question = state["question"]
# #     # Retrieval
# #     documents = vector_store.similarity_search(question)
# #     return {"documents": documents, "question": question}

# # # Define your router model
# # class RouteQuery(BaseModel):
# #     datasource: Literal["vectorstore", "workflow_sim_swap", "recharge_plan_workflow", "conversation"] = Field(
# #         ..., description="Route to vectorstore, workflows, or fallback conversation."
# #     )

# # # LLM setup for routing
# # llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
# # structured_llm_router = llm.with_structured_output(RouteQuery)

# # # Define routing logic
# # system = """You are an expert at handling Airtel-related queries and are responsible for directing them to the appropriate resource.
# # Route queries about Airtel’s general services, network issues, or company-related information to the vectorstore,
# # SIM swap-related queries to the SIM swap workflow, recharge or plan details to the recharge plan workflow,
# # and if the query doesn't fit these categories or is ambiguous, rely on a fallback conversation model using the LLM for further assistance.
# # Ensure accuracy in routing to provide the best possible user experience."""

# # route_prompt = ChatPromptTemplate.from_messages(
# #     [
# #         ("system", system),
# #         ("human", "{question}"),
# #     ]
# # )
# # question_router = route_prompt | structured_llm_router

# # # SIM swap workflow
# # def sim_swap_workflow(state):
# #     """Trigger SIM swap workflow in a step-by-step manner."""
# #     print("---sim workflow--- ")
# #     question = state["question"]

# #     name = input("Please provide your full name for the SIM swap?")
# #     phone_number = input("Please provide your phone number for SIM swap")

# #     return {"sim_swap_details": name + " " + phone_number, "question": question}

# # # Recharge plan workflow
# # def recharge_plan_workflow(state):
# #     """Trigger Recharge Plan Workflow."""
# #     question = state["question"]

# #     if state.get("step") == "ask_phone_number":
# #         return {"generation": "Please provide your phone number for the recharge plans.", "step": "collect_phone_number"}

# #     elif state.get("step") == "collect_phone_number":
# #         phone_number = question
# #         url = f"https://www.airtel.in/recharge/prepaid/?siNumber={phone_number}"
# #         response = requests.get(url)
# #         if response.status_code == 200:
# #             available_plans = response.text
# #             return {"generation": f"Here are the available plans for {phone_number}: {available_plans}", "step": "completed"}
# #         else:
# #             return {"generation": "Sorry, there was an issue retrieving the plans. Please try again.", "step": "completed"}

# #     elif "recharge plan" in question.lower() and not state.get("step"):
# #         return {"generation": "Please provide your phone number to retrieve available recharge plans.", "step": "ask_phone_number"}

# #     return state

# # # Graph workflow for routing and handling user queries
# # from langgraph.graph import END, StateGraph, START

# # workflow = StateGraph()
# # workflow.add_node("vectorstore", retrieve)
# # workflow.add_node("workflow_sim_swap", sim_swap_workflow)
# # workflow.add_node("recharge_plan_workflow", recharge_plan_workflow)
# # workflow.add_conditional_edges(
# #     START,
# #     question_router.invoke,
# #     {
# #         "vectorstore": "vectorstore",
# #         "workflow_sim_swap": "workflow_sim_swap",
# #         "recharge_plan_workflow": "recharge_plan_workflow",
# #     }
# # )
# # workflow.add_edge("vectorstore", END)
# # workflow.add_edge("workflow_sim_swap", END)
# # workflow.add_edge("recharge_plan_workflow", END)

# # # Compile workflow
# # app = workflow.compile()

# # # Chatbot runner function
# # def run_chatbot():
# #     state = {"question": "", "step": None}

# #     while True:
# #         state["question"] = input("You: ")

# #         if state["question"].lower() in ["quit", "exit", "stop"]:
# #             print("Bot: Goodbye!")
# #             break

# #         outputs = app.stream(state)

# #         for output in outputs:
# #             for key, value in output.items():
# #                 print(f"Bot: {value}")
# #                 if value.get("step") == "completed":
# #                     print("Bot: Thank you! The workflow is now complete.")
# #                     break

# #         print("\n---\n")

# # run_chatbot()
