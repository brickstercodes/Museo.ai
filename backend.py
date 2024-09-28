import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
import supabase
from supabase import Client, create_client

load_dotenv("/Users/anugrahshetty/Desktop/Local e3/apiKeys.env")

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Import vector database functions
from vectordb import generate_embedding, store_in_supabase

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Keeping it open for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TogetherLLM
current_time = datetime.now()
Settings.llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo", 
    api_key=os.getenv("TOGETHER_API_KEY")  # Make sure the environment variable is set
)

# Initialize FAISS index
dimension = 768  # Ensure this matches the actual embedding dimension
index = faiss.IndexFlatL2(dimension)
content_store = []  # This will store the content corresponding to each embedding

# Add some sample data to the index
sample_data = [
    "The Louvre is a famous museum in Paris.",
    "The Metropolitan Museum of Art is located in New York City.",
    "The British Museum in London has a vast collection of world art and artifacts.",
    "The Uffizi Gallery in Florence is renowned for its collection of Renaissance art.",
    "The Prado Museum in Madrid houses one of the world's finest collections of European art."
]

for content in sample_data:
    embedding = generate_embedding(content)
    index.add(np.array([embedding]))
    content_store.append(content)

logging.info(f"Initialized FAISS index with {len(sample_data)} sample entries")

# Create the prompt template
character_creation_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            """
               You are Museo AI, an AI assistant that handles museum ticket booking requests.
                Your primary goal is to assist users in booking museum tickets based on the most up-to-date information available.

                Key instructions:
                1. Always use the most recent information provided to you, especially when it comes from the 'Relevant information' section.
                2. If new museum information is provided that wasn't in your original list, incorporate it into your knowledge and use it.
                3. Be adaptive and don't stick rigidly to information from earlier in the conversation if new details are provided.
                4. If the user expresses interest in a museum, use any available information about that museum to assist them.

                Remember:
                - The user's preferences and choices are paramount.
                - Always provide accurate information based on the most recent data given to you.
                - If you receive new information about a museum, treat it as valid and use it to help the user.

                
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

async def get_response_from_model(prompt: str, session_id: str, websocket: WebSocket, user_input: str):
    logger.info(f"Starting response generation for session {session_id}")
    try:
        user_input_embedding = generate_embedding(user_input)
        logger.info(f"Generated embedding for user input: {user_input[:50]}...")

        similar_content = await query_supabase(user_input_embedding)

        if similar_content:
            additional_context = f"\n\nNEW RELEVANT INFORMATION: {similar_content}\n\nPlease incorporate this new information into your knowledge and use it to respond to the user's request about {user_input}. This information supersedes any previous list of museums you may have provided."
            prompt += additional_context
            logger.info(f"Added context to prompt: {additional_context[:100]}...")

        logger.info("Sending prompt to LLM for response generation")
        logger.debug(f"Full prompt sent to LLM: {prompt}")
        response = Settings.llm.stream_complete(prompt=prompt, temperature=0.7)
        
        full_response = await stream_response_to_client(response, websocket, session_id)

        session_histories[session_id] += f"User: {user_input}\nMuseo AI: {full_response}\n"
        logger.info(f"Updated session history for {session_id}")

        if "<DONE>" in full_response:
            await websocket.send_text(json.dumps({"complete": True}))

    except Exception as e:
        logger.error(f"Error in get_response_from_model: {str(e)}", exc_info=True)
        await websocket.send_text(json.dumps({"error": "Error processing your request."}))

async def query_supabase(user_input_embedding):
    try:
        logger.info("Querying Supabase for similar content")
        query_result = supabase.rpc('match_documents', {
            'query_embedding': user_input_embedding,
            'match_threshold': 0.5,
            'match_count': 1,
            'table_name': 'table1'
        }).execute()
        
        logger.info(f"Supabase query result: {query_result}")
        
        if query_result.data:
            similar_content = query_result.data[0]['content']
            logger.info(f"Found similar content in Supabase: {similar_content[:100]}...")
            return similar_content
        else:
            logger.info("No matching documents found in Supabase")
            return None
    except Exception as supabase_error:
        logger.error(f"Error querying Supabase: {str(supabase_error)}", exc_info=True)
        return None

async def stream_response_to_client(response, websocket, session_id):
    full_response = ""
    try:
        for chunk in response:
            delta = extract_delta(chunk)
            if delta:
                full_response += delta
                await websocket.send_text(json.dumps({"delta": delta}))
                logger.debug(f"Sent chunk to client for session {session_id}: {delta[:50]}...")
                await asyncio.sleep(0.01)
        
        logger.info(f"Full response for session {session_id}: {full_response[:100]}...")
        return full_response
    except Exception as e:
        logger.error(f"Error streaming response to client: {str(e)}", exc_info=True)
        raise

def extract_delta(chunk):
    if isinstance(chunk, str):
        return chunk
    elif hasattr(chunk, 'delta'):
        return chunk.delta
    elif isinstance(chunk, dict):
        if 'choices' in chunk and chunk['choices']:
            return chunk['choices'][0].get('delta', {}).get('content', '')
        elif 'content' in chunk:
            return chunk['content']
    return ""

@app.on_event("startup")
async def startup_event():
    try:
        # Test Supabase connection
        result = supabase.table('table1').select("*").limit(1).execute()
        logger.info(f"Supabase connection test successful. Result: {result}")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")
        # You might want to raise an exception here to prevent the app from starting

# In your main function
try:
    # Ensure user_input is defined before this block
    user_input = "sample input"  # Replace with actual user input
    user_input_embedding = generate_embedding(user_input)
    
    query_result = supabase.rpc('match_documents', {
        'query_embedding': user_input_embedding,
        'match_threshold': 0.5,
        'match_count': 1
    }).execute()
    
    # More detailed logging
    logger.info(f"Supabase query result: {query_result}")
    
    similar_content = None
    if query_result.data:
        similar_content = query_result.data[0]['content']
        logger.info(f"Found similar content in Supabase: {similar_content[:100]}...")
except Exception as supabase_error:
    logger.error(f"Error querying Supabase: {str(supabase_error)}", exc_info=True)
    similar_content = None

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    session_id = None

    try:
        while True:
            data = await websocket.receive_text()
            logging.info(f"Received data: {data}")
            
            try:
                json_data = json.loads(data)
                user_input = json_data.get("message", "").strip()
                session_id = json_data.get("session_id", None)
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"JSON parsing error: {str(e)}")
                await websocket.send_text(json.dumps({"error": "Invalid message format"}))
                continue

            if not session_id:
                logging.error("Session ID missing from the request")
                await websocket.send_text(json.dumps({"error": "Session ID is required"}))
                continue

            if session_id not in session_histories:
                session_histories[session_id] = ""
            
            logging.info(f"User input: {user_input}")

            prompt = character_creation_template.format(question=user_input, history=session_histories[session_id])
            logging.info(f"Constructed prompt: {prompt}")

            if not prompt:
                logging.error("Constructed prompt is empty.")
                await websocket.send_text(json.dumps({"error": "Prompt is required."}))
                continue
            logger.info(f"Processing request for session {session_id}")
            await get_response_from_model(prompt, session_id, websocket, user_input)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    finally:
        await websocket.close()
        if session_id in session_histories:
            logging.info(f"Cleaning up session {session_id}")
            del session_histories[session_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
