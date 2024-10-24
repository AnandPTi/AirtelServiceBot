
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
from chatbot import get_bot_response  # Importing the bot response function
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS settings for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (Authorization, etc.)
)

# Dummy database to hold session data
session_data: Dict[str, Dict] = {}

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"  # We use a user_id to track individual sessions

# Dummy recharge plans (you can fetch this dynamically or from a database in a real-world scenario)
recharge_plans = {
    "basic": "Basic Plan: 199 INR for 28 days, 1GB data per day, unlimited calls",
    "standard": "Standard Plan: 399 INR for 56 days, 1.5GB data per day, unlimited calls",
    "premium": "Premium Plan: 799 INR for 84 days, 2GB data per day, unlimited calls"
}

@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id

    # Check if user exists in the session data, if not, initialize them
    if user_id not in session_data:
        session_data[user_id] = {"state": "default", "full_name": "", "phone_number": "", "selected_plan": ""}

    # Extract the user session
    user_session = session_data[user_id]
    message = request.message.lower()

    # Call chatbot response function from chatbot.py
    bot_response = get_bot_response(message)

    # SIM swap process
    if user_session["state"] == "awaiting_details":
        if not user_session["full_name"]:
            user_session["full_name"] = message  # Assume the next message is the full name
            return {"response": "Please provide your phone number."}

        elif not user_session["phone_number"]:
            user_session["phone_number"] = message  # Assume the next message is the phone number
            # SIM swap process complete, return confirmation message
            response = (f"Dear {user_session['full_name']}, your phone number "
                        f"{user_session['phone_number']} has been successfully submitted for SIM swap! "
                        "We will contact you soon.")
            # Reset session state
            session_data[user_id] = {"state": "default", "full_name": "", "phone_number": "", "selected_plan": ""}
            return {"response": response}

    # Detect "SIM swap" request in user message
    if "sim swap" in message or "swap sim" in message or "change sim" in message or "new sim" in message or "replace sim" in message or "swap my sim" in message or "change my sim" in message: 
        # Ask for full name first
        user_session["state"] = "awaiting_details"
        return {"response": "For SIM swap, please provide your full name."}

    # Recharge plan process
    if user_session["state"] == "awaiting_plan_details":
        if not user_session["phone_number"]:
            user_session["phone_number"] = message  # Assume the next message is the phone number
            # Plan inquiry complete, return confirmation message with the selected plan
            response = (f"Dear user, your phone number {user_session['phone_number']} has been submitted for recharge "
                        f"on the {user_session['selected_plan']}! We will contact you shortly for confirmation.")
            # Reset session state
            session_data[user_id] = {"state": "default", "full_name": "", "phone_number": "", "selected_plan": ""}
            return {"response": response}

    # Recharge plan inquiry

    if "recharge plan" in message or "plan details" in message:
        # List available plans
        available_plans = "\n".join(recharge_plans.values())
        return {"response": f"Here are our available recharge plans:\n{available_plans}\nWhich one would you like to choose?"}

    # Specific plan request (e.g., "Tell me about the basic plan")
    for plan_name, plan_details in recharge_plans.items():
        if plan_name in message:
            user_session["selected_plan"] = plan_name  # Set the selected plan in the session
            user_session["state"] = "awaiting_plan_details"
            return {"response": f"You selected the {plan_details}. Please provide your phone number for recharge."}

    # Default response if no specific keyword detected, use bot's response
    return {"response": bot_response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)









# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from typing import Dict
# from chatbot import get_bot_response  # Importing the bot response function
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # CORS settings for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allow all headers (Authorization, etc.)
# )

# # Dummy database to hold session data
# session_data: Dict[str, Dict] = {}

# class ChatRequest(BaseModel):
#     message: str
#     user_id: str = "default_user"  # We use a user_id to track individual sessions

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     user_id = request.user_id

#     # Check if user exists in the session data, if not, initialize them
#     if user_id not in session_data:
#         session_data[user_id] = {"state": "default", "full_name": "", "phone_number": ""}

#     # Extract the user session
#     user_session = session_data[user_id]
#     message = request.message.lower()

#     # Call chatbot response function from chatbot.py
#     bot_response = get_bot_response(message)

#     # SIM swap process
#     if user_session["state"] == "awaiting_details":
#         if not user_session["full_name"]:
#             user_session["full_name"] = message  # Assume the next message is the full name
#             return {"response": "Please provide your phone number."}

#         elif not user_session["phone_number"]:
#             user_session["phone_number"] = message  # Assume the next message is the phone number
#             # SIM swap process complete, return confirmation message
#             response = (f"Dear {user_session['full_name']}, your phone number "
#                         f"{user_session['phone_number']} has been successfully submitted for SIM swap! "
#                         "We will contact you soon.")
#             # Reset session state
#             session_data[user_id] = {"state": "default", "full_name": "", "phone_number": ""}
#             return {"response": response}

#     # Detect "SIM swap" request in user message
#     if "sim swap" in message or "swap sim" in message or "change sim" in message or "new sim" in message or "replace sim" in message or "swap my sim" in message or "change my sim" in message: 
#         # Ask for full name first
#         user_session["state"] = "awaiting_details"
#         return {"response": "For SIM swap, please provide your full name."}

#     # Default response if no specific keyword detected, use bot's response
#     return {"response": bot_response}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from typing import Dict
# from chatbot import get_bot_response  # Importing the bot response function
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allow all headers (Authorization, etc.)
# )
# # Dummy database to hold session data
# session_data: Dict[str, Dict] = {}

# class ChatRequest(BaseModel):
#     message: str
#     user_id: str = "default_user"  # We use a user_id to track individual sessions

# # Dummy recharge plans (you can fetch this dynamically or from a database in a real-world scenario)
# recharge_plans = {
#     "basic": "Basic Plan: 199 INR for 28 days, 1GB data per day, unlimited calls",
#     "standard": "Standard Plan: 399 INR for 56 days, 1.5GB data per day, unlimited calls",
#     "premium": "Premium Plan: 799 INR for 84 days, 2GB data per day, unlimited calls"
# }

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     user_id = request.user_id

#     # Check if user exists in the session data, if not, initialize them
#     if user_id not in session_data:
#         session_data[user_id] = {"state": "default", "full_name": "", "phone_number": ""}

#     # Extract the user session
#     user_session = session_data[user_id]
#     message = request.message.lower()

#     # Call chatbot response function from chatbot.py
#     bot_response = get_bot_response(message)

#     # SIM swap process
#     if user_session["state"] == "awaiting_details":
#         if not user_session["full_name"]:
#             user_session["full_name"] = message  # Assume the next message is the full name
#             return {"response": "Please provide your phone number."}

#         elif not user_session["phone_number"]:
#             user_session["phone_number"] = message  # Assume the next message is the phone number
#             # SIM swap process complete, return confirmation message
#             response = (f"Dear {user_session['full_name']}, your phone number "
#                         f"{user_session['phone_number']} has been successfully submitted for SIM swap! "
#                         "We will contact you soon.")
#             # Reset session state
#             session_data[user_id] = {"state": "default", "full_name": "", "phone_number": ""}
#             return {"response": response}

#     # Detect "SIM swap" request in user message
#     if "sim swap" in message or "swap sim" in message:
#         # Ask for full name first
#         user_session["state"] = "awaiting_details"
#         return {"response": "For SIM swap, please provide your full name."}

#     # Recharge plan inquiry
#     if "recharge plan" in message or "plan details" in message:
#         # You can enhance this to fetch specific plans dynamically if needed
#         available_plans = "\n".join(recharge_plans.values())
#         return {"response": f"Here are our available recharge plans:\n{available_plans}\nWhich one would you like to choose?"}

#     # Specific plan request (e.g., "Tell me about the basic plan")
#     for plan_name, plan_details in recharge_plans.items():
#         if plan_name in message:
#             return {"response": plan_details}

#     # Default response if no specific keyword detected, use bot's response
#     return {"response": bot_response}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from chatbot import get_bot_response  # Import the chatbot function
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # Allow CORS for specific domains (your frontend URL)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allow all headers (Authorization, etc.)
# )

# # Define the input schema to include additional fields
# class ChatRequest(BaseModel):
#     message: str
#     full_name: str = None  # Optional: Needed for SIM swap
#     phone_number: str = None  # Optional: Needed for SIM swap and recharge plan

# # Define the output schema
# class ChatResponse(BaseModel):
#     response: str

# # Endpoint to process chatbot messages
# @app.post("/chat", response_model=ChatResponse)
# async def chat_with_bot(request: ChatRequest):
#     try:
#         # Pass the message, name, and phone number to the chatbot
#         bot_response = get_bot_response(request.message, request.full_name, request.phone_number)
#         return ChatResponse(response=bot_response)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Home route to check the API status
# @app.get("/")
# async def read_root():
#     return {"message": "Welcome to the Airtel chatbot API!"}

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from chatbot import get_bot_response  # Import the chatbot function
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()
# # Allow CORS for specific domains or all domains
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allow all headers (Authorization, etc.)
# )

# # Define the input schema
# class ChatRequest(BaseModel):
#     message: str
#     full_name: str = None
#     phone_number: str = None

# # Define the output schema
# class ChatResponse(BaseModel):
#     response: str

# @app.post("/chat", response_model=ChatResponse)
# async def chat_with_bot(request: ChatRequest):
#     try:
#         # Pass the message, name, and phone number to the chatbot
#         bot_response = get_bot_response(request.message, request.full_name, request.phone_number)
#         return ChatResponse(response=bot_response)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Example home route
# @app.get("/")
# async def read_root():
#     return {"message": "Welcome to the Airtel chatbot API!"}

# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from chatbot import get_bot_response  # Import the chatbot function
# # from fastapi.middleware.cors import CORSMiddleware


# # app = FastAPI()
# # # Allow CORS for specific domains or all domains
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["http://localhost:3000"],  # Frontend URL
# #     allow_credentials=True,
# #     allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
# #     allow_headers=["*"],  # Allow all headers (Authorization, etc.)
# # )

# # # Define the input schema
# # # Update the input schema to include additional fields
# # class ChatRequest(BaseModel):
# #     message: str
# #     full_name: str = None
# #     phone_number: str = None


# # # Define the output schema
# # class ChatResponse(BaseModel):
# #     response: str

# # @app.post("/chat", response_model=ChatResponse)
# # async def chat_with_bot(request: ChatRequest):
# #     try:
# #         # Pass the message, name, and phone number to the chatbot
# #         bot_response = get_bot_response(request.message, request.full_name, request.phone_number)
# #         return ChatResponse(response=bot_response)
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # # # Route for posting a user message and getting a chatbot response
# # # @app.post("/chat", response_model=ChatResponse)
# # # async def chat_endpoint(request: ChatRequest):
# # #     try:
# # #         # Call the chatbot function to get the response
# # #         response = get_bot_response(request.message)
# # #         return ChatResponse(response=response)
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=str(e))

# # # Example home route
# # @app.get("/")
# # async def read_root():
# #     return {"message": "Welcome to the Airtel chatbot API!"}
