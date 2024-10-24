


import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests
from typing import List, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone
from langgraph.graph import END, StateGraph, START
from embedding import get_vector_store  # Import the vector store function from embedding.py
from langchain_openai import AzureChatOpenAI
from openai import OpenAI
client = OpenAI()
# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model for LLM routing
# llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize Azure OpenAI model for LLM routing
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# Initialize Pinecone vector store
vector_store = get_vector_store()

# Define the route query model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["basic_llm", "knowledge_based_llm", "swap_sim_workflow", "current_recharge_plan_workflow", "fallback_message"] = Field (
        ...,
        description="Route to basic_llm, knowledge_based_llm, swap_sim_workflow, current recharge plan workflow or fallback message.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

# Routing system prompt
system = """
You are an expert AI assistant specializing in Airtel-related queries. Your main responsibility is to accurately route user inquiries to the appropriate resources, ensuring an optimal user experience. Use the following routing guidelines:

1. **General Airtel-Related Queries**: 
   - For inquiries about Airtel's services, Airtel information, or company-related information, route to **knowledge_based_llm**. 
   - Example: "What are the available plans for Airtel?" or Airtel DTH plans."

2. **SIM Swap Queries**: 
   - If the user mentions anything related to SIM swaps, guide them through the process by routing to **swap_sim_workflow**.
   - Example: "How do I swap my SIM?" or "I need to change my SIM."

3. **Recharge or Plan Details**: 
   - Direct questions about recharges, balance inquiries, or plan information to **current_recharge_plan_workflow**.
   - Example: "What are the current recharge options?" or "How can I check my balance?"

4. **Telecom Services and Casual Conversations**: 
   - Handle casual chats, greetings, or general telecom discussions with **basic_llm**. 
   - Example: "Hi, can you tell me about what to do when network issue?" or "What’s new in telecom? or "what is 5G""

5. **Fallback for Ambiguous or Irrelevant Queries**: 
   - For questions that are unclear, unrelated to Airtel or telecom, or consist of random content, route them to **fallback_message**.
   - Example: "What is the capital of Japan?" or "Tell me a joke."

Always ensure to accurately determine the nature of the inquiry to provide the most relevant and helpful response. If unsure, it's better to direct users to the fallback mechanism for further assistance.
"""

route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
question_router = route_prompt | structured_llm_router

sentences = ["what is airtel?","who is president of japan", "tell me about my current plans?","how can i change my sim from jio to airtel?","who are you", "Can you tell me about airtel DTH"]
for query in sentences:
    print("Query: ", query,end=" ")
    print(
        question_router.invoke(
            {"question": query}
        )
    )


# Define the GraphState for the state graph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.datasource == "basic_llm":
        return "basic_llm"
    elif source.datasource == "knowledge_based_llm":
        return "vectorstore"
    elif source.datasource == "swap_sim_workflow":
        return "workflow_sim_swap"
    elif source.datasource == "current_recharge_plan_workflow":
        return "recharge_plan_workflow"
    elif source.datasource == "fallback_message":
        return "fallback_agent"

# Functions for each workflow step

# Modify retrieve_from_knowledge_base to combine LLM response with the vectorstore paragraph
def retrieve_from_knowledge_base(state):
    """Retrieve documents from the knowledge base."""
    print("---RETRIEVE FROM KNOWLEDGE BASE---")
    question = state["question"]

    # Search for relevant documents
    documents = vector_store.similarity_search(question)
    if not documents:
        return {"generation": "No relevant information found in the knowledge base.", "question": question}

    relevant_info = documents[0].page_content[:500]  # Get the first 500 characters of the first document

    llm_prompt = f"Here is the information I found about your query: {relevant_info}. Can you please generate a concise, customized response for the given {question}?"

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" based on your needs
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates responses based on the provided information."},
                {"role": "user", "content": llm_prompt}
            ],
            max_tokens=1000,  # Adjust max tokens as needed
            temperature=0.7
        )

        # Extract the generated response
        llm_response = response.choices[0].message.content
        combined_response = f"{llm_response}\n\nAdditional Info:\n{relevant_info}"
        return {"generation": combined_response, "question": question}

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"generation": "Error generating response. Please try again.", "question": question}

# def retrieve_from_knowledge_base(state):
#     """Retrieve documents from the knowledge base."""
#     print("---RETRIEVE FROM KNOWLEDGE BASE---")
#     question = state["question"]

#     # Search for relevant documents
#     documents = vector_store.similarity_search(question)
#     if not documents:
#         return {"generation": "No relevant information found in the knowledge base.", "question": question}

#     relevant_info = documents[0].page_content[:500]  # Get the first 500 characters of the first document

#     llm_prompt = f"Here is the information I found about your query: {relevant_info}. Can you please generate a concise, customized response for the given {question}?"

#     # Call the Azure OpenAI API
#     try:
#         headers = {
#             "Content-Type": "application/json",
#             "api-key": AZURE_OPENAI_API_KEY
#         }
#         data = {
#             "messages": [{"role": "user", "content": llm_prompt}],
#             "max_tokens": 1000,  # Adjust max tokens as needed
#             "temperature": 0.7
#         }

#         # Make the request to the Azure OpenAI API
#         response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
#         response.raise_for_status()  # Raise an error for bad responses
#         result = response.json()

#         # Extract the generated response
#         llm_response = result["choices"][0]["message"]["content"]
#         combined_response = f"{llm_response}\n\nAdditional Info:\n{relevant_info}"
#         return {"generation": combined_response, "question": question}

#     except Exception as e:
#         print(f"Error calling Azure OpenAI API: {e}")
#         return {"generation": "Error generating response. Please try again.", "question": question}


def sim_swap_workflow(state):
    """Trigger SIM swap workflow."""
    name = state.get("full_name", "Unknown User")
    phone_number = state.get("phone_number", "0000000000")
    
    # Provide confirmation message for SIM swap request
    return {"generation": f"Dear {name}, your phone number {phone_number} has been successfully submitted for SIM swap! We will contact you soon.", "question": state["question"]}

def recharge_plan_workflow(state):
    """Return recharge plan details based on user's phone number."""
    user_phone_number = state.get("phone_number", "0000000000")
    
    # Sample plan options
    current_plan = {
        "Rs 199": "Unlimited calls, 1.5 GB daily data, 100 SMS/day for 28 days.",
        "Rs 299": "Unlimited calls, 2 GB daily data, 100 SMS/day for 28 days.",
        "Rs 449": "Unlimited calls, 3 GB daily data, 100 SMS/day for 56 days.",
        "Rs 599": "Unlimited calls, 4 GB daily data, 100 SMS/day for 84 days."
    }

    user_current_plan = "Rs 299"  # Sample user's current plan
    plan_description = current_plan.get(user_current_plan, "Sorry, no details available for your current plan.")
    
    return {"generation": f"Your current plan is {user_current_plan}. {plan_description}", "question": state["question"]}

def fallback_agent(state):
    return {"generation": "I'm sorry, I couldn't find relevant information. Please ask about Airtel services or clarify your query.", "question": state["question"]}


def basic_llm(state):
    basic_prompt = """
    You are an Airtel help bot designed to assist with basic inquiries and casual conversation related to Airtel services.
    Introduce yourself as an Airtel help bot and respond to general inquiries or greetings in a friendly manner.
    """

    question = state["question"]
    conversation_prompt = f"{basic_prompt}\nHuman: {question}\nAirtel Bot:"

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(model="gpt-4o-mini",  # or "gpt-3.5-turbo" based on your needs
        messages=[
            {"role": "system", "content": basic_prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=1000,  # Adjust max tokens as needed
        temperature=0.7)

        # Extract the generated response
        llm_response = response.choices[0].message.content
        return {"documents": [{"text": llm_response}], "question": question}

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        fallback_message = "I'm sorry, I couldn't understand your request. Please try asking something else!"
        return {"documents": [{"text": fallback_message}], "question": question}
    
# def basic_llm(state):
#     basic_prompt = """
#     You are an Airtel help bot designed to assist with basic inquiries and casual conversation related to Airtel services.
#     Introduce yourself as an Airtel help bot and respond to general inquiries or greetings in a friendly manner.
#     """

#     question = state["question"]
#     conversation_prompt = f"{basic_prompt}\nHuman: {question}\nAirtel Bot:"

#     try:
#         # Prepare headers and data for the Azure API request
#         headers = {
#             "Content-Type": "application/json",
#             "api-key": AZURE_OPENAI_API_KEY
#         }
#         data = {
#             "messages": [{"role": "user", "content": conversation_prompt}],
#             "max_tokens": 1000,  # Adjust max tokens as needed
#             "temperature": 0.7
#         }

#         # Make the request to the Azure OpenAI API
#         response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
#         response.raise_for_status()  # Raise an error for bad responses
#         result = response.json()

#         # Extract the generated response
#         llm_response = result["choices"][0]["message"]["content"]
#         return {"documents": [{"text": llm_response}], "question": question}

#     except Exception as e:
#         print(f"Error calling Azure OpenAI API: {e}")
#         fallback_message = "I'm sorry, I couldn't understand your request. Please try asking something else!"
#         return {"documents": [{"text": fallback_message}], "question": question}


# Build the main workflow using StateGraph
workflow = StateGraph(GraphState)
workflow.add_node("basic_llm", basic_llm)
workflow.add_node("vectorstore", retrieve_from_knowledge_base)
workflow.add_node("workflow_sim_swap", sim_swap_workflow)
workflow.add_node("recharge_plan_workflow", recharge_plan_workflow)
workflow.add_node("fallback_agent", fallback_agent)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "basic_llm": "basic_llm",
        "vectorstore": "vectorstore",
        "workflow_sim_swap": "workflow_sim_swap",
        "recharge_plan_workflow": "recharge_plan_workflow",
        "fallback_agent": "fallback_agent"
    }
)

workflow.add_edge("basic_llm", END)
workflow.add_edge("vectorstore", END)
workflow.add_edge("workflow_sim_swap", END)
workflow.add_edge("recharge_plan_workflow", END)
workflow.add_edge("fallback_agent", END)

app = workflow.compile()

# Chatbot runner function
def run_chatbot():
    state = {"question": ""}

    while True:
        state["question"] = input("You: ")

        if state["question"].lower() in ["quit", "exit", "stop", "bye"]:
            print("Bot: Goodbye!")
            break

        outputs = app.stream(state)

        for output in outputs:
            for key, value in output.items():
                if "generation" in value:
                    print(f"Bot: {value['generation']}")
                elif "documents" in value:
                    for doc in value["documents"]:
                        print(f"Bot: {doc['text']}")
                elif "sim_swap_details" in value:
                    print(f"Bot: {value['sim_swap_details']}")

        print("\n---\n")
def get_bot_response(question: str, full_name: str = None, phone_number: str = None):
    # Prepare the initial state with user inputs
    state = {
        "question": question,
        "full_name": full_name,
        "phone_number": phone_number
    }

    # Run the compiled state graph
    outputs = app.stream(state)

    # Collect the chatbot response from the state graph
    for output in outputs:
        for key, value in output.items():
            if "generation" in value:
                return value["generation"]  # Return the generated response
            elif "documents" in value:
                return value["documents"][0]["text"]  # Return the document content
    return "Sorry, I couldn't understand your request."


# Example usage of the chatbot runner
if __name__ == "__main__":
    run_chatbot()











# import os
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field
# import requests
# from typing import List, Literal
# from typing_extensions import TypedDict
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import Pinecone
# from langgraph.graph import END, StateGraph, START
# from embedding import get_vector_store  # Import the vector store function from embedding.py
# from langchain_openai import AzureChatOpenAI
# # Load environment variables
# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize OpenAI model for LLM routing
# # llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# # Initialize Azure OpenAI model for LLM routing
# llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# # Initialize Pinecone vector store
# vector_store = get_vector_store()

# # Define the route query model
# class RouteQuery(BaseModel):
#     """Route a user query to the most relevant datasource."""
#     datasource: Literal["basic_llm", "knowledge_based_llm", "swap_sim_workflow", "current_recharge_plan_workflow", "fallback_message"] = Field (
#         ...,
#         description="Route to basic_llm, knowledge_based_llm, swap_sim_workflow, current recharge plan workflow or fallback message.",
#     )

# structured_llm_router = llm.with_structured_output(RouteQuery)

# # Routing system prompt
# system = """
# You are an expert AI assistant specializing in Airtel-related queries. Your main responsibility is to accurately route user inquiries to the appropriate resources, ensuring an optimal user experience. Use the following routing guidelines:

# 1. **General Airtel-Related Queries**: 
#    - For inquiries about Airtel's services, Airtel information, or company-related information, route to **knowledge_based_llm**. 
#    - Example: "What are the available plans for Airtel?" or Airtel DTH plans."

# 2. **SIM Swap Queries**: 
#    - If the user mentions anything related to SIM swaps, guide them through the process by routing to **swap_sim_workflow**.
#    - Example: "How do I swap my SIM?" or "I need to change my SIM."

# 3. **Recharge or Plan Details**: 
#    - Direct questions about recharges, balance inquiries, or plan information to **current_recharge_plan_workflow**.
#    - Example: "What are the current recharge options?" or "How can I check my balance?"

# 4. **Telecom Services and Casual Conversations**: 
#    - Handle casual chats, greetings, or general telecom discussions with **basic_llm**. 
#    - Example: "Hi, can you tell me about what to do when network issue?" or "What’s new in telecom? or "what is 5G""

# 5. **Fallback for Ambiguous or Irrelevant Queries**: 
#    - For questions that are unclear, unrelated to Airtel or telecom, or consist of random content, route them to **fallback_message**.
#    - Example: "What is the capital of Japan?" or "Tell me a joke."

# Always ensure to accurately determine the nature of the inquiry to provide the most relevant and helpful response. If unsure, it's better to direct users to the fallback mechanism for further assistance.
# """

# route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
# question_router = route_prompt | structured_llm_router

# # sentences = ["what is airtel?","who is president of japan", "tell me about my current plans?","how can i change my sim from jio to airtel?","who are you", "Can you tell me about airtel DTH"]
# # for query in sentences:
# #     print("Query: ", query,end=" ")
# #     print(
# #         question_router.invoke(
# #             {"question": query}
# #         )
# #     )


# # Define the GraphState for the state graph
# class GraphState(TypedDict):
#     question: str
#     generation: str
#     documents: List[str]

# def route_question(state):
#     question = state["question"]
#     source = question_router.invoke({"question": question})

#     if source.datasource == "basic_llm":
#         return "basic_llm"
#     elif source.datasource == "knowledge_based_llm":
#         return "vectorstore"
#     elif source.datasource == "swap_sim_workflow":
#         return "workflow_sim_swap"
#     elif source.datasource == "current_recharge_plan_workflow":
#         return "recharge_plan_workflow"
#     elif source.datasource == "fallback_message":
#         return "fallback_agent"

# # Functions for each workflow step

# # Modify retrieve_from_knowledge_base to combine LLM response with the vectorstore paragraph
# def retrieve_from_knowledge_base(state):
#     """Retrieve documents from the knowledge base."""
#     print("---RETRIEVE FROM KNOWLEDGE BASE---")
#     question = state["question"]

#     # Search for relevant documents
#     documents = vector_store.similarity_search(question)
#     if not documents:
#         return {"generation": "No relevant information found in the knowledge base.", "question": question}

#     relevant_info = documents[0].page_content[:500]  # Get the first 500 characters of the first document

#     llm_prompt = f"Here is the information I found about your query: {relevant_info}. Can you please generate a concise, customized response for the given {question}?"

#     # Call the Azure OpenAI API
#     try:
#         headers = {
#             "Content-Type": "application/json",
#             "api-key": AZURE_OPENAI_API_KEY
#         }
#         data = {
#             "messages": [{"role": "user", "content": llm_prompt}],
#             "max_tokens": 1000,  # Adjust max tokens as needed
#             "temperature": 0.7
#         }

#         # Make the request to the Azure OpenAI API
#         response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
#         response.raise_for_status()  # Raise an error for bad responses
#         result = response.json()

#         # Extract the generated response
#         llm_response = result["choices"][0]["message"]["content"]
#         combined_response = f"{llm_response}\n\nAdditional Info:\n{relevant_info}"
#         return {"generation": combined_response, "question": question}

#     except Exception as e:
#         print(f"Error calling Azure OpenAI API: {e}")
#         return {"generation": "Error generating response. Please try again.", "question": question}


# def sim_swap_workflow(state):
#     """Trigger SIM swap workflow."""
#     name = state.get("full_name", "Unknown User")
#     phone_number = state.get("phone_number", "0000000000")
    
#     # Provide confirmation message for SIM swap request
#     return {"generation": f"Dear {name}, your phone number {phone_number} has been successfully submitted for SIM swap! We will contact you soon.", "question": state["question"]}



# def ask_phone_number_recharge_plan(state):
#     """Ask the user for their phone number for recharge plan inquiry."""
#     return {"generation": "Please provide your phone number to check your current recharge plan.", "question": state["question"]}

# def recharge_plan_workflow(state):
#     """Return recharge plan details based on user's phone number."""
#     user_phone_number = state.get("phone_number", None)
    
#     if not user_phone_number:
#         return {"generation": "Phone number not provided. Please provide a valid phone number to check your recharge plan.", "question": state["question"]}

#     # Sample plan options
#     current_plan = {
#         "Rs 199": "Unlimited calls, 1.5 GB daily data, 100 SMS/day for 28 days.",
#         "Rs 299": "Unlimited calls, 2 GB daily data, 100 SMS/day for 28 days.",
#         "Rs 449": "Unlimited calls, 3 GB daily data, 100 SMS/day for 56 days.",
#         "Rs 599": "Unlimited calls, 4 GB daily data, 100 SMS/day for 84 days."
#     }

#     user_current_plan = "Rs 299"  # Sample user's current plan
#     plan_description = current_plan.get(user_current_plan, "Sorry, no details available for your current plan.")
    
#     return {"generation": f"Your current plan is {user_current_plan}. {plan_description}", "question": state["question"]}


# def fallback_agent(state):
#     return {"generation": "I'm sorry, I couldn't find relevant information. Please ask about Airtel services or clarify your query.", "question": state["question"]}

# def basic_llm(state):
#     basic_prompt = """
#     You are an Airtel help bot designed to assist with basic inquiries and casual conversation related to Airtel services.
#     Introduce yourself as an Airtel help bot and respond to general inquiries or greetings in a friendly manner.
#     """

#     question = state["question"]
#     conversation_prompt = f"{basic_prompt}\nHuman: {question}\nAirtel Bot:"

#     try:
#         # Prepare headers and data for the Azure API request
#         headers = {
#             "Content-Type": "application/json",
#             "api-key": AZURE_OPENAI_API_KEY
#         }
#         data = {
#             "messages": [{"role": "user", "content": conversation_prompt}],
#             "max_tokens": 1000,  # Adjust max tokens as needed
#             "temperature": 0.7
#         }

#         # Make the request to the Azure OpenAI API
#         response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
#         response.raise_for_status()  # Raise an error for bad responses
#         result = response.json()

#         # Extract the generated response
#         llm_response = result["choices"][0]["message"]["content"]
#         return {"documents": [{"text": llm_response}], "question": question}

#     except Exception as e:
#         print(f"Error calling Azure OpenAI API: {e}")
#         fallback_message = "I'm sorry, I couldn't understand your request. Please try asking something else!"
#         return {"documents": [{"text": fallback_message}], "question": question}


# # Build the main workflow using StateGraph
# workflow = StateGraph(GraphState)
# workflow.add_node("basic_llm", basic_llm)
# workflow.add_node("vectorstore", retrieve_from_knowledge_base)
# workflow.add_node("workflow_sim_swap", sim_swap_workflow)
# workflow.add_node("ask_phone_number_recharge_plan", ask_phone_number_recharge_plan)
# workflow.add_node("recharge_plan_workflow", recharge_plan_workflow)
# workflow.add_node("fallback_agent", fallback_agent)

# workflow.add_conditional_edges(
#     START,
#     route_question,
#     {
#         "basic_llm": "basic_llm",
#         "vectorstore": "vectorstore",
#         "workflow_sim_swap": "workflow_sim_swap",
#         "recharge_plan_workflow": "ask_phone_number_recharge_plan",
#         "fallback_agent": "fallback_agent"
#     }
# )

# workflow.add_edge("basic_llm", END)
# workflow.add_edge("vectorstore", END)
# workflow.add_edge("workflow_sim_swap", END)
# workflow.add_edge("ask_phone_number_recharge_plan", "recharge_plan_workflow")
# workflow.add_edge("recharge_plan_workflow", END)

# workflow.add_edge("fallback_agent", END)

# app = workflow.compile()

# # Chatbot runner function
# def run_chatbot():
#     state = {"question": ""}

#     while True:
#         state["question"] = input("You: ")

#         if state["question"].lower() in ["quit", "exit", "stop", "bye"]:
#             print("Bot: Goodbye!")
#             break

#         outputs = app.stream(state)

#         for output in outputs:
#             for key, value in output.items():
#                 if "generation" in value:
#                     print(f"Bot: {value['generation']}")
#                 elif "documents" in value:
#                     for doc in value["documents"]:
#                         print(f"Bot: {doc['text']}")
#                 elif "sim_swap_details" in value:
#                     print(f"Bot: {value['sim_swap_details']}")

#         print("\n---\n")
# def get_bot_response(question: str, full_name: str = None, phone_number: str = None):
#     # Prepare the initial state with user inputs
#     state = {
#         "question": question,
#         "full_name": full_name,
#         "phone_number": phone_number
#     }

#     # Run the compiled state graph
#     outputs = app.stream(state)

#     # Collect the chatbot response from the state graph
#     for output in outputs:
#         for key, value in output.items():
#             if "generation" in value:
#                 return value["generation"]  # Return the generated response
#             elif "documents" in value:
#                 return value["documents"][0]["text"]  # Return the document content
#     return "Sorry, I couldn't understand your request."


# # Example usage of the chatbot runner
# if __name__ == "__main__":
#     run_chatbot()






























#     # def get_bot_response(question: str, full_name: str = None, phone_number: str = None) -> str:
# #     state = {"question": question}
    
# #     if full_name:
# #         state["full_name"] = full_name
# #     if phone_number:
# #         state["phone_number"] = phone_number

# #     outputs = app.stream(state)
    
# #     response = ""
# #     for output in outputs:
# #         for key, value in output.items():
# #             if "generation" in value:
# #                 response = value['generation']
# #             elif "documents" in value:
# #                 response = value["documents"][0]["text"]
# #             elif "sim_swap_details" in value:
# #                 response = value["sim_swap_details"]
    
# #     return response