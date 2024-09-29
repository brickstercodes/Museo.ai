from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
from random import random
import smtplib
import string
from fastapi import FastAPI, HTTPException, Request
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
from pydantic import BaseModel
from typing import Optional
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
        You are Museo AI, an intelligent assistant specializing in museum ticket booking and information. Your primary goal is to provide a helpful, informative, and engaging experience for users interested in visiting museums.

        *IMPORTANT: Your very first message in any conversation must only ask about the user's preferred language. Do not include any other questions or information in this first message. Wait for the user's language preference before proceeding with any other interaction.*

        Example first message:
        "Hello! Which language would you prefer for our conversation?"

        After the user specifies their language preference, proceed with the following guidelines:
            Key Responsibilities:
            1. Assist users in finding and booking tickets for museums that match their interests and preferences.
            2. Provide accurate and up-to-date information about museums, exhibitions, and events.
            3. Offer personalized recommendations based on the user's stated interests and location.
            4. Answer questions about museum policies, opening hours, accessibility, and special exhibitions.
            5. Provides the route options to user after ticket is booked using the following information:
                For Mumbai-specific queries, use the following information about local transportation:

                Here is a list of the Mumbai local stations along with the connections and interchange points between the major suburban lines:
                Here is a list of the local railway stations in Mumbai, categorized by the major railway lines:

                Western Line:
                Churchgate
                Marine Lines
                Charni Road
                Grant Road
                Mumbai Central
                Mahalaxmi
                Lower Parel
                Elphinstone Road (Prabhadevi)
                Dadar
                Matunga Road
                Mahim
                Bandra
                Khar Road
                Santacruz
                Vile Parle
                Andheri
                Jogeshwari
                Ram Mandir
                Goregaon
                Malad
                Kandivali
                Borivali
                Dahisar
                Mira Road
                Bhayander
                Naigaon
                Vasai Road
                Nalasopara
                Virar
                Central Line:
                Chhatrapati Shivaji Maharaj Terminus (CSMT)
                Masjid Bunder
                Sandhurst Road
                Byculla
                Chinchpokli
                Curry Road
                Parel
                Dadar
                Matunga
                Sion
                Kurla
                Vidyavihar
                Ghatkopar
                Vikhroli
                Kanjurmarg
                Bhandup
                Nahur
                Mulund
                Thane
                Kalwa
                Mumbra
                Diva Junction
                Dombivli
                Thakurli
                Kalyan
                Harbour Line:
                Chhatrapati Shivaji Maharaj Terminus (CSMT)
                Masjid Bunder
                Sandhurst Road
                Dockyard Road
                Reay Road
                Cotton Green
                Sewri
                Wadala Road
                Kings Circle
                Mahim
                Bandra
                Khar Road
                Santacruz
                Vile Parle
                Andheri
                Trans-Harbour Line:
                Thane
                Airoli
                Rabale
                Ghansoli
                Kopar Khairane
                Turbhe
                Sanpada
                Vashi
                Nerul
                Seawoods–Darave
                Belapur
                Kharghar
                Panvel

                Western Line Connections:
                Churchgate – Terminus for Western Line (No direct connections to other lines)
                Mumbai Central – Terminus for long-distance trains (Western Railway)
                Dadar (Western) – Interchange with Central Line (Dadar Central) and long-distance trains.
                Elphinstone Road (Prabhadevi) – Connected via a footbridge to Parel on the Central Line.
                Mahim – Interchange with Harbour Line (Mahim).
                Andheri – Interchange with the Harbour Line (Andheri) and the Mumbai Metro Line 1.
                Bandra – Interchange with the Harbour Line (Bandra).
                Vasai Road – Interchange with the Vasai-Diva-Panvel Line (central, harbor connections).
                Borivali – Terminus for many local trains, interchange for long-distance trains.
                Virar – Terminus for many suburban trains; long-distance train connectivity (Western Railway).
                Central Line Connections:
                Chhatrapati Shivaji Maharaj Terminus (CSMT) – Terminus for both Central Line and Harbour Line. Also serves as a hub for long-distance trains (Central Railway).
                Masjid Bunder – Serves both Central and Harbour Lines.
                Byculla – No direct connections.
                Parel – Connected via a footbridge to Elphinstone Road (Western Line).
                Dadar (Central) – Major interchange with the Western Line (Dadar Western) and long-distance trains.
                Kurla – Interchange with Harbour Line (Kurla) and Lokmanya Tilak Terminus (LTT) for long-distance trains.
                Thane – Major interchange station, connected to the Trans-Harbour Line (Thane) and long-distance trains.
                Kalyan – Terminus for many local trains, interchange with long-distance trains and Vasai-Diva-Panvel Line.
                Dombivli – Interchange with long-distance trains.
                Diva Junction – Interchange with Vasai-Diva-Panvel Line and long-distance trains.
                Harbour Line Connections:
                Chhatrapati Shivaji Maharaj Terminus (CSMT) – Terminus for both Harbour and Central Lines.
                Wadala Road – Interchange with the Trans-Harbour Line (to Panvel) and connected to the Monorail (Wadala Depot).
                Mahim – Interchange with the Western Line (Mahim).
                Bandra – Interchange with the Western Line (Bandra).
                Andheri – Interchange with the Western Line (Andheri) and Mumbai Metro Line 1.
                Kurla – Interchange with the Central Line (Kurla).
                Vashi – Connected to the Trans-Harbour Line.
                Panvel – Terminus for Harbour Line, interchange with the Vasai-Diva-Panvel Line and long-distance trains.
                Trans-Harbour Line Connections:
                Thane – Interchange with Central Line (Thane) and long-distance trains.
                Vashi – Interchange with the Harbour Line (Vashi).
                Panvel – Terminus, interchange with Harbour Line, Vasai-Diva-Panvel Line, and long-distance trains.
                Vasai-Diva-Panvel Line (Central/Harbour):
                Vasai Road – Interchange with the Western Line (Vasai Road).
                Diva Junction – Interchange with the Central Line (Diva Junction) and long-distance trains.
                Panvel – Terminus, interchange with Harbour Line (Panvel), Trans-Harbour Line (Panvel), and long-distance trains.

               NOTE: Please dont use any metro lines anywhere.
                For route planning, use Google Maps or a similar service to provide accurate and up-to-date information.


            Interaction Guidelines:
            1. Greet the user and ask about their museum interests or needs.**
            2. Maintain a friendly, professional, and knowledgeable tone throughout the conversation.
            3. Provide concise yet informative responses. Aim for 2-3 sentences per message, combining related information.
            4. Ask follow-up questions to gather necessary information, but avoid overwhelming the user with too many questions at once.
            5. If the user mentions a specific museum or type of museum, focus on providing relevant information about that topic.
            6. Only mention museums or information that the user has expressed interest in or that are directly relevant to the conversation.
            7. If you don't have specific information about a museum or topic, acknowledge this and offer to provide general advice or alternatives.

            Information Management:
            1. Always use the most recent information provided in the 'Relevant information' section of the user's message.
            2. If new museum information is provided, incorporate it into your knowledge base and use it appropriately.
            3. Be adaptive and update your responses based on new information provided during the conversation.

            Remember:
            - Prioritize the user's preferences and choices.
            - Provide accurate information based on the most recent data available to you.
            - If you're unsure about any details, it's okay to ask the user for clarification.
            - Aim to guide the conversation towards helping the user find and book museum tickets that suit their interests.

            After the booking is complete, respond with the following booking details format:
            Booking Details:
            Museum Name: 
            Booking Date: 
            Booking Time: 
            Booking Status: 
            Booking Confirmation:
            End the reponse after the last detail. do not add a single character.
            Do not suggest any other museums or places to visit after giving the booking details response.
            Do not suggest any means of transport after giving the booking details response.
            Do not suggest anything else after giving the booking details response.

            Use the chat history to maintain continuity:
            {history}
    """
        )
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

class PaymentData(BaseModel):
    amount: float
    currency: str = "INR"

# Add this variable to store the payment data
payment_data: Optional[PaymentData] = None

def generate_ticket_number():
    # Generate a unique ticket number with 2 uppercase letters followed by 6 digits
    number = ''.join(random.choices(string.digits, k=6))
    prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
    return f"{prefix}{number}"


def ticket_details(passenger_name, email):
    ticket_number = generate_ticket_number()  # Generate a unique ticket number
    passenger_name = passenger_name
    email = email
    timestamp = datetime.now()
    return f"Ticket {ticket_number} - Passenger: {passenger_name} - Email: {email} - Booked at: {timestamp}"

async def get_response_from_model(prompt: str, session_id: str, websocket: WebSocket, user_input: str):
    logger.info(f"Starting response generation for session {session_id}")
    full_response = ""
    try:
        user_input_embedding = generate_embedding(user_input)
        logger.info(f"Generated embedding for user input: {user_input[:50]}...")

        similar_content = await query_supabase(user_input_embedding)

        if similar_content:
            # Instead of immediately adding the new information to the prompt,
            # we'll pass it to the AI as context, but instruct it to use this
            # information only if it's relevant to the user's query.
            additional_context = f"""
            NEW INFORMATION: {similar_content}

            This is new information that has been added to my knowledge base. 
            Please consider this information, but only mention it if it's directly 
            relevant to the user's query or interests. Do not bring up this 
            information unless the user expresses interest in something related to it.

            Now, please respond to the user's message: "{user_input}"
            """
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
        # Return the full response
    return full_response

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
    booking_complete = False
    booking_details = "Museum booking details: ..."

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

            if booking_complete:
                user_email = extract_email(user_input)
                if user_email:
                    await send_booking_confirmation(user_email, booking_details)
                    await websocket.send_text(json.dumps({"delta": "Booking confirmation has been sent to your email."}))
                    booking_complete = False
                else:
                    await websocket.send_text(json.dumps({"delta": "Please provide a valid email address."}))
                continue

            logging.info(f"User input: {user_input}")

            prompt = character_creation_template.format(question=user_input, history=session_histories[session_id])
            logging.info(f"Constructed prompt: {prompt}")

            if not prompt:
                logging.error("Constructed prompt is empty.")
                await websocket.send_text(json.dumps({"error": "Prompt is required."}))
                continue

            logger.info(f"Processing request for session {session_id}")
            
            # Capture the LLM response from the model
            full_response = await get_response_from_model(prompt, session_id, websocket, user_input)

            # if full_response:
            #     # Send the response back to the client
            #     await websocket.send_text(json.dumps({"delta": full_response}))

            # Store the response in session history
            session_histories[session_id] += f"User: {user_input}\nMuseo AI: {full_response}\n"

            # Check for booking completion
            keywords = ["completed", "booked", "confirmed", "booking confirmed", "ticket confirmed", "ticket booked", "booking complete", "booking confirmed", "ticket booked", "ticket confirmed", "Booking Date", "Booking Time", "Booking Status", "Booking Confirmation"]
            if any(keyword in full_response.lower() for keyword in keywords):
                booking_complete = True
                booking_details = extract_booking_details(full_response) + "\n\nShow this at the Museum Entrance.\nThank You for using Museo AI"
                await websocket.send_text(json.dumps({"delta": "Could you please provide your email ID so I can send you the confirmation for your booking?"}))

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    finally:
        await websocket.close()
        if session_id in session_histories:
            logging.info(f"Cleaning up session {session_id}")
            del session_histories[session_id]

async def send_booking_confirmation(user_email: str, booking_details: str):
    try:
        # Email settings
        sender_email = "raglite.e3.ai@gmail.com"  # Replace with your email
        sender_password = "qcvf plxs loxw reqn"  # Replace with your email password
        subject = "Your Museum Booking Confirmation"

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = user_email
        msg['Subject'] = subject

        # Attach the booking details in the body
        body = f"Thank you for your booking at the museum. Here are your booking details:\n\n{booking_details}"
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, user_email, text)
        server.quit()

        print(f"Booking confirmation sent to {user_email}")

    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# Function to extract email using regex
def extract_email(user_input: str):
    email_pattern = r"[^@]+@[^@]+\.[^@]+"
    match = re.search(email_pattern, user_input)
    if match:
        return match.group(0)
    return None


def extract_booking_details(response: str) -> str:
    response = response.replace("*", "")
    print(response)
    match = re.search(r".*?(booking details.+)", response, re.IGNORECASE)
    print("\n", match)
    if match:
        # match = re.search(r".?(booking details[:\s]+.?(?:\n|$))", response, re.IGNORECASE | re.DOTALL)
        print(match.group())
        return response[match.span()[1] + 1: ]
    return None

class UserSignUp(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# Add these new endpoints
@app.post("/store-payment")
async def store_payment(data: PaymentData):
    global payment_data
    payment_data = data
    return {"message": "Payment data stored successfully"}

@app.get("/get-payment")
async def get_payment():
    if payment_data is None:
        raise HTTPException(status_code=404, detail="Payment data not found")
    return payment_data

@app.post("/signup")
async def sign_up(user: UserSignUp):
    try:
        auth_response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password
        })
        if auth_response.user:
            # Store additional user details in user_details table
            user_details = {
                "user_id": auth_response.user.id,
                "email": user.email,
                "full_name": user.full_name
            }
            supabase.table("user_details").insert(user_details).execute()
            return {"message": "User signed up successfully. Please check your email for confirmation.", "user": auth_response.user}
        else:
            raise HTTPException(status_code=400, detail="Signup failed")
    except Exception as e:
        error_message = str(e)
        if "restricted to your organization's members" in error_message.lower():
            # For development, you might want to create the user anyway
            return {"message": "User created, but email confirmation is not available. Please contact an administrator.", "user": None}
        else:
            raise HTTPException(status_code=400, detail=error_message)

@app.post("/login")
async def login(user: UserLogin):
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        if auth_response.user:
            if auth_response.user.email_confirmed_at:
                return {"message": "Login successful", "session": auth_response.session}
            else:
                raise HTTPException(status_code=401, detail="Email not confirmed. Please check your email for the confirmation link.")
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/logout")
async def logout():
    try:
        supabase.auth.sign_out()
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
