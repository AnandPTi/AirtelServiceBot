# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:29:39.219150Z","iopub.execute_input":"2024-10-23T18:29:39.219452Z","iopub.status.idle":"2024-10-23T18:29:40.176715Z","shell.execute_reply.started":"2024-10-23T18:29:39.219419Z","shell.execute_reply":"2024-10-23T18:29:40.175962Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
!pip install langchain langgraph cassio

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:32:11.382947Z","iopub.execute_input":"2024-10-23T18:32:11.383344Z","iopub.status.idle":"2024-10-23T18:32:12.434453Z","shell.execute_reply.started":"2024-10-23T18:32:11.383307Z","shell.execute_reply":"2024-10-23T18:32:12.433669Z"}}
import cassio
## connection of the ASTRA DB
ASTRA_DB_APPLICATION_TOKEN="AstraCS:NOfXjlOpnnYtdCMQjgLPxTBS:4aa2cc0555a7cbfd550dcfccd5da526278c485914192b040c48ba222c801cb1b" # enter the "AstraCS:..." string found in in your Token JSON file"
ASTRA_DB_ID="043a432f-f173-4b5d-9383-edc12a571431"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

# %% [code]
!pip install langchain_community

# %% [code]
!pip install -U langchain_community tiktoken langchain-groq langchainhub chromadb langchain langgraph langchain_huggingface

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:18:04.288517Z","iopub.execute_input":"2024-10-23T20:18:04.288887Z","iopub.status.idle":"2024-10-23T20:18:05.069194Z","shell.execute_reply.started":"2024-10-23T20:18:04.288853Z","shell.execute_reply":"2024-10-23T20:18:05.068445Z"}}
### Build Index

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma


### from langchain_cohere import CohereEmbeddings



# Docs to index
urls = [
    "https://www.airtel.in/",
    "https://www.airtel.in/airtel-thanks-app?icid=footer"
]


# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)


# %% [code]


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:39:46.594322Z","iopub.execute_input":"2024-10-23T18:39:46.595032Z","iopub.status.idle":"2024-10-23T18:39:46.598920Z","shell.execute_reply.started":"2024-10-23T18:39:46.594992Z","shell.execute_reply":"2024-10-23T18:39:46.598019Z"}}
import os
os.environ["HF_TOKEN"] = "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"

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:39:50.312836Z","iopub.execute_input":"2024-10-23T18:39:50.313718Z","iopub.status.idle":"2024-10-23T18:40:12.070986Z","shell.execute_reply.started":"2024-10-23T18:39:50.313676Z","shell.execute_reply":"2024-10-23T18:40:12.070105Z"}}
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:40:41.892838Z","iopub.execute_input":"2024-10-23T18:40:41.893610Z","iopub.status.idle":"2024-10-23T18:40:43.628252Z","shell.execute_reply.started":"2024-10-23T18:40:41.893569Z","shell.execute_reply":"2024-10-23T18:40:43.627426Z"}}
from langchain.vectorstores.cassandra import Cassandra
astra_vector_store=Cassandra(
    embedding=embeddings,
    table_name="airtel_help_bot",
    session=None,
    keyspace=None

)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:41:07.623973Z","iopub.execute_input":"2024-10-23T18:41:07.624422Z","iopub.status.idle":"2024-10-23T18:41:08.862935Z","shell.execute_reply.started":"2024-10-23T18:41:07.624379Z","shell.execute_reply":"2024-10-23T18:41:08.862065Z"}}
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
astra_vector_store.add_documents(doc_splits)
print("Inserted %i headlines." % len(doc_splits))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# %% [markdown]
# ## Note Here: Our Vector Database Retriver

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:41:17.206843Z","iopub.execute_input":"2024-10-23T18:41:17.207618Z","iopub.status.idle":"2024-10-23T18:41:17.211846Z","shell.execute_reply.started":"2024-10-23T18:41:17.207577Z","shell.execute_reply":"2024-10-23T18:41:17.210851Z"}}
retriever=astra_vector_store.as_retriever()

# %% [markdown]
# Sample Example

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:41:52.418438Z","iopub.execute_input":"2024-10-23T18:41:52.418814Z","iopub.status.idle":"2024-10-23T18:41:52.513982Z","shell.execute_reply.started":"2024-10-23T18:41:52.418778Z","shell.execute_reply":"2024-10-23T18:41:52.513081Z"}}
retriever.invoke("What is airtel",ConsistencyLevel="LOCAL_ONE")

# %% [markdown]
# ## Router

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:46:13.538133Z","iopub.execute_input":"2024-10-23T18:46:13.538537Z","iopub.status.idle":"2024-10-23T18:46:13.551027Z","shell.execute_reply.started":"2024-10-23T18:46:13.538502Z","shell.execute_reply":"2024-10-23T18:46:13.550018Z"}}

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["basic_llm", "knowledge_based_llm","swap_sim_workflow","current_recharge_plan_workflow","fallback_message"] = Field(
        ...,
        description="Route to basic_llm, knowledge_based_llm, swap_sim_workflow, current recharge plan workflow or fallback message.",
    )


# %% [markdown]
# Our LLM Now

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:48:48.417128Z","iopub.execute_input":"2024-10-23T18:48:48.418265Z","iopub.status.idle":"2024-10-23T18:48:48.423137Z","shell.execute_reply.started":"2024-10-23T18:48:48.418221Z","shell.execute_reply":"2024-10-23T18:48:48.422046Z"}}
os.environ["GROQ_API_KEY"]="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"
groq_api_key = os.environ["GROQ_API_KEY"]

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T18:48:50.447000Z","iopub.execute_input":"2024-10-23T18:48:50.447397Z","iopub.status.idle":"2024-10-23T18:48:50.538542Z","shell.execute_reply.started":"2024-10-23T18:48:50.447362Z","shell.execute_reply":"2024-10-23T18:48:50.537767Z"}}
from langchain_groq import ChatGroq


llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)



# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:02:37.535375Z","iopub.execute_input":"2024-10-23T20:02:37.536230Z","iopub.status.idle":"2024-10-23T20:02:37.542772Z","shell.execute_reply.started":"2024-10-23T20:02:37.536187Z","shell.execute_reply":"2024-10-23T20:02:37.541915Z"}}
# Prompt
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
question_router = route_prompt | structured_llm_router


# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:02:42.246802Z","iopub.execute_input":"2024-10-23T20:02:42.247667Z","iopub.status.idle":"2024-10-23T20:02:43.913040Z","shell.execute_reply.started":"2024-10-23T20:02:42.247613Z","shell.execute_reply":"2024-10-23T20:02:43.912250Z"}}
sentences = ["what is airtel?","who is president of japan", "tell me about my current plans?","how can i change my sim from jio to airtel?","who are you"]
for query in sentences:
    print("Query: ", query,end=" ")
    print(
        question_router.invoke(
            {"question": query}
        )
    )

# %% [markdown]
# # Graph

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:03:26.939916Z","iopub.execute_input":"2024-10-23T20:03:26.940718Z","iopub.status.idle":"2024-10-23T20:03:26.945635Z","shell.execute_reply.started":"2024-10-23T20:03:26.940676Z","shell.execute_reply":"2024-10-23T20:03:26.944656Z"}}

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

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:03:27.222737Z","iopub.execute_input":"2024-10-23T20:03:27.223272Z","iopub.status.idle":"2024-10-23T20:03:27.228366Z","shell.execute_reply.started":"2024-10-23T20:03:27.223235Z","shell.execute_reply":"2024-10-23T20:03:27.227527Z"}}
from langchain.schema import Document


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:03:27.509953Z","iopub.execute_input":"2024-10-23T20:03:27.510620Z","iopub.status.idle":"2024-10-23T20:03:27.515781Z","shell.execute_reply.started":"2024-10-23T20:03:27.510583Z","shell.execute_reply":"2024-10-23T20:03:27.514883Z"}}
def sim_swap_workflow(state):
    """Trigger SIM swap workflow in a step-by-step manner."""
    # Check the current state of the workflow and prompt accordingly
    print("---sim workflow--- ")
    question = state["question"]
    print("We need your full name and phone number for sim swap")
    # Check if the workflow has been initiated, ask for the next step
    name = input("Please provide your full name? ")
    phone_number = input("Please provide your phone number? ")

    return {"documents": f"Dear {name}, Your phone number -{phone_number} is successfully submitted! We will contact you soon.", "question": question}


# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:03:27.914194Z","iopub.execute_input":"2024-10-23T20:03:27.914509Z","iopub.status.idle":"2024-10-23T20:03:27.921520Z","shell.execute_reply.started":"2024-10-23T20:03:27.914477Z","shell.execute_reply":"2024-10-23T20:03:27.920553Z"}}
### Edges ###


def route_question(state):
    """
    Route question to "basic_llm", "knowledge_based_llm","swap_sim_workflow","current_recharge_plan_workflow","fallback_message"

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "basic_llm":
        print("---ROUTE QUESTION TO basic_llm---")
        return "basic_llm"
    
    elif source.datasource == "knowledge_based_llm":
        print("---ROUTE QUESTION TO knowledge_based_llm---")
        return "knowledge_based_llm"
    
    elif source.datasource == "swap_sim_workflow":
        print("---ROUTE QUESTION TO swap_sim_workflow ---")
        return "swap_sim_workflow"
    
    elif source.datasource == "current_recharge_plan_workflow":
        print("---ROUTE QUESTION TO current_recharge_plan_workflow---")
        return "current_recharge_plan_workflow"
    
    elif source.datasource == "fallback_message":
        print("---ROUTE QUESTION TO fallback_message---")
        return "fallback_message"
    

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:06:48.426624Z","iopub.execute_input":"2024-10-23T20:06:48.427191Z","iopub.status.idle":"2024-10-23T20:06:48.434274Z","shell.execute_reply.started":"2024-10-23T20:06:48.427156Z","shell.execute_reply":"2024-10-23T20:06:48.433271Z"}}
# import requests

# def current_recharge_plan_workflow(state):
#     """Trigger Recharge Plan Workflow."""
#     question = state["question"]

#     print("--Triggered Recharge Plan Workflow--")
#     user_phone_number = input("Please Enter your phone number to check your current plan?")
# #     url = f"https://www.airtel.in/recharge/prepaid/?siNumber={user_phone_number}&icid=recharge_rail"

#     # Simulate an API call to get the available plans (you can replace this with actual scraping if necessary)
# #     response = requests.get(url)
# #     if response.status_code == 200:
# #         available_plans = response.text  # Or parse the response if it's JSON/HTML
# #         return {"documents": f"Here are the available plans for {user_phone_number}: {available_plans}", "question": question}
# #     return {"documents": "Sorry, there was an issue retrieving the plans. Please try again.", "question": question}
#     return {"documents": f"Your phone number {user_phone_number} have current plan of Rs 299. ", "question": question}
    
    
def current_recharge_plan_workflow(state):
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
        "documents": f"Your phone number {user_phone_number} has the current plan of {user_current_plan}. {plan_description}",
        "question": question
    }



# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:06:48.935002Z","iopub.execute_input":"2024-10-23T20:06:48.935809Z","iopub.status.idle":"2024-10-23T20:06:48.940870Z","shell.execute_reply.started":"2024-10-23T20:06:48.935774Z","shell.execute_reply":"2024-10-23T20:06:48.939846Z"}}

def fallback_message(state):
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

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:06:49.777473Z","iopub.execute_input":"2024-10-23T20:06:49.777820Z","iopub.status.idle":"2024-10-23T20:06:49.784882Z","shell.execute_reply.started":"2024-10-23T20:06:49.777785Z","shell.execute_reply":"2024-10-23T20:06:49.783875Z"}}
def basic_llm(state):
    """
    Return a conversation message related to Airtel services in a friendly manner.
    
    Args:
        state (dict): The current graph state.
        
    Returns:
        dict: Contains the 'documents' key with the response message and 'question' key with the original question.
    """
    # Airtel-specific basic conversation prompt
    basic_prompt = """
    You are an Airtel help bot designed to assist with basic inquiries and casual conversation related to Airtel services.
    You can answer questions such as greetings, inquiries about Airtel’s general services, or any simple questions related to Airtel.
    If the question is outside your scope or requires detailed technical assistance, kindly refer the user to more specific help channels.

    Always introduce yourself as an Airtel help bot. For example, if someone asks a general question, you can respond with a friendly greeting and let them know that you're here to help with basic Airtel-related queries.
    """

    question = state["question"]

    # Format the conversation prompt with the user's question
    conversation_prompt = f"{basic_prompt}\nHuman: {question}\nAirtel Bot:"

    try:
        # Call the LLM to generate a response using the formatted conversation prompt
        response = llm.generate(conversation_prompt)
        # Return the generated response in the desired format
        return {"documents": [{"text": response}], "question": question}
    except Exception as e:
        # Fallback message in case of any error
        fallback_message = "I'm sorry, but I couldn't find relevant information for your query. Could you please clarify or ask about Airtel services, plans, or issues? I'm here to help!"
#         print(f"Error: {e}")
        # Return fallback message in the same format
        return {"documents": [{"text": fallback_message}], "question": question}


# %% [markdown]
# # Langgraph

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:23:28.508152Z","iopub.execute_input":"2024-10-23T20:23:28.508862Z","iopub.status.idle":"2024-10-23T20:23:28.519255Z","shell.execute_reply.started":"2024-10-23T20:23:28.508825Z","shell.execute_reply":"2024-10-23T20:23:28.518288Z"}}
from langgraph.graph import END, StateGraph, START

# Workflow
workflow = StateGraph(GraphState)

# Add nodes for different workflows
workflow.add_node("basic_llm", basic_llm)
workflow.add_node("knowledge_based_llm", retrieve)
workflow.add_node("swap_sim_workflow", sim_swap_workflow)
workflow.add_node("current_recharge_plan_workflow", current_recharge_plan_workflow) 
workflow.add_node("fallback_message", fallback_message) 

# Conditional routing based on the question intent
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "basic_llm": "basic_llm",
        "knowledge_based_llm": "knowledge_based_llm",
        "swap_sim_workflow": "swap_sim_workflow",
        "current_recharge_plan_workflow": "current_recharge_plan_workflow",
        "fallback_message": "fallback_message",
    }
)

# End nodes
workflow.add_edge("basic_llm", END)
workflow.add_edge("knowledge_based_llm", END)
workflow.add_edge("swap_sim_workflow", END)
workflow.add_edge("current_recharge_plan_workflow", END)
workflow.add_edge("fallback_message", END)

app = workflow.compile()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:23:29.007973Z","iopub.execute_input":"2024-10-23T20:23:29.008319Z","iopub.status.idle":"2024-10-23T20:23:29.092237Z","shell.execute_reply.started":"2024-10-23T20:23:29.008287Z","shell.execute_reply":"2024-10-23T20:23:29.091397Z"}}
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %% [code] {"execution":{"iopub.status.busy":"2024-10-23T20:10:02.192515Z","iopub.execute_input":"2024-10-23T20:10:02.193470Z","iopub.status.idle":"2024-10-23T20:10:04.707218Z","shell.execute_reply.started":"2024-10-23T20:10:02.193426Z","shell.execute_reply":"2024-10-23T20:10:04.706270Z"}}
from pprint import pprint

while True:
    
    print("You: ",end=" ")
    user_input = input("")
    if "exit" in user_input:
        break
    inputs = {
        "question": user_input
    }
    for output in app.stream(inputs):
        for key,value in output.items():
            pprint(f"Node {key} :")
            
        pprint("\n---\n")
    
        print("Assistant: ",end=" ")
        try:
            pprint(value['documents'][0].dict()['metadata']['description'])
        except Exception as e:
            print(value["documents"])
#             print(e)

# %% [code]
