from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio
import json
import os
import numpy as np
import faiss
from llama_index.core import VectorStoreIndex, Settings, ChatPromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage, MessageRole
from datetime import datetime
import re

# Import vector database functions
from vectordb import generate_embedding, store_in_supabase

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TogetherLLM
current_time = datetime.now()
Settings.llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo", 
    api_key=os.getenv("")  # Use env var for API key
)

# Initialize FAISS index
dimension = 384  # This should match the dimension of your embeddings
index = faiss.IndexFlatL2(dimension)
content_store = []  # This will store the content corresponding to each embedding

# Create the prompt template
character_creation_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(f"""You are an AI that will handle Museum ticket booking requests by the user.
                Your name is Museo AI.
                The user will have the list of Museums in the city they are by an API.
                You will need the following information from the user:

               1. Ask the user exact address where they are living.

               2. Just ask the user in which city they want to visit the museum in short.

               3. List the top 5 to 10 museums in that city.

               4. Ask the user which museum they want to visit.

               5. Ask how many adults and children will be visiting.

               5. Ask the date of traveling.

               6. Ask the user if he wants to book traveling tickets.

               7. Use maps and provide shortest route through cab, train, walking and metro train and flight if needed with traveling time from the user's location to the given museum address and provide a path for selected mode of transport.

               8. Also allow to book ticket by giving cab, flight and train tickets when asked by the user.

               9. Give the directions if requested by the user in short. 
               
               10. Display the price chart of the museum by web scrapping or from an api.

               11. Ask how many adults and children will be visiting.

               11. Give total cost of the bill including traveling and museum tickets.

               12. Ask the user at what date and time will they arrive at the Museum. User may enter in any format. Interpret it correctly.
               Check if the date is valid, that is, if it is a future date. If past date is entered, notify user and ask again.
            """
        + f" Current time: {current_time} " + 
            """
               13. Any Add-ons like a tour guide at the museum OR and Audio guide at the museum.

               14. Check for parking for 2-wheeler or 4-wheeler at the museum and get the parking rates for that museum as well.
               15. Be flexible and allow booking changes OR cancellations: Changes to Museum choice, number of adults or children, date or time or optionals
               16. Do not print bold or italics or any other fancy formatting. Just plain text will do.
               17. Keep your responses short and crisp. No puns, use simple words and be professional.
               18. Important key note: Expect multi-lingual input. For example: hindi, gujarati, marathi, spanish, german, french, russian. also expect the languages to be written in english characters.
               19. user wants to speak in a particular language, respond purely in that language using that language's script and characters.

               Ask only one question at a time.

               Ask user to select the flights from given real time flights from google, ask user to select the Cab from given real time Cabs from google and ask user to select the train and its classes and to change the station as well from given real time train tickets from google ask the user twice before booking traveling tickets and include prices of each tickets.
               Please ask the date before booking traveling tickets.
            Ask the email ID for booking traveling and museum tickets and payments.
               Expect multiple information in one response. See which information is missing and ask for it accordingly.
               
               Use the chat history to maintain continuity:
               {history}
            """)
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="{question}"
    ),
]
character_creation_template = ChatPromptTemplate(character_creation_msgs)

# In-memory history for each session
session_histories = {}

class ChatRequest(BaseModel):
    user_input: str
    session_id: str

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = ""
    awaiting_email = False
    
    try:
        while True:
            data = await websocket.receive_text()
            user_input = json.loads(data)["message"]

            # Perform vector search using FAISS
            query_vector = np.array([generate_embedding(user_input)]).astype('float32')
            k = 5  # number of nearest neighbors to retrieve
            distances, indices = index.search(query_vector, k)

            context = ""
            for idx in indices[0]:
                if idx < len(content_store):
                    context += f"{content_store[idx]}\n"

            prompt = f"""Based on the following context and chat history, please respond to the user's question:

Context:
{context}

Chat History:
{chat_history}

User Question: {user_input}

Please provide a relevant and informative response:"""

            response = Settings.llm.stream_complete(character_creation_template.format(question=prompt, history=chat_history))
            full_response = ""
            
            for r in response:
                full_response += r.delta
                await websocket.send_text(json.dumps({"delta": r.delta}))
            
            chat_history += f"User: {user_input}\nCharacter: {full_response}\n"
            
            if "<DONE>" in full_response:
                await websocket.send_text(json.dumps({"complete": True}))

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await websocket.close()

def add_to_index(content: str):
    embedding = generate_embedding(content)
    index.add(np.array([embedding]).astype('float32'))
    content_store.append(content)

def load_initial_data():
    initial_data = [
        "This is some initial content",
        "Another piece of initial content",
        # Add more initial content here if needed
    ]
    for content in initial_data:
        add_to_index(content)

load_initial_data()

@app.post("/chat/")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_input = request.user_input

    chat_history = session_histories.get(session_id, "")
    
    query_vector = np.array([generate_embedding(user_input)]).astype('float32')
    k = 5
    distances, indices = index.search(query_vector, k)

    context = ""
    for idx in indices[0]:
        if idx < len(content_store):
            context += f"{content_store[idx]}\n"

    prompt = f"""Based on the following context and chat history, please respond to the user's question:

Context:
{context}

Chat History:
{chat_history}

User Question: {user_input}

Please provide a relevant and informative response:"""

    response = Settings.llm.stream_complete(character_creation_template.format(question=prompt, history=chat_history))
    full_response = ""
    
    for r in response:
        full_response += r.delta

    full_response = re.sub(r" (\d)\. ", r"\n \1. ", full_response)

    chat_history += f"User: {user_input}\nBot: {full_response}\n"
    session_histories[session_id] = chat_history

    return {"response": full_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
