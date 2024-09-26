from fastapi import FastAPI, Request
from llama_index.core import VectorStoreIndex, Settings, ChatPromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.supabase import SupabaseVectorStore
from datetime import datetime
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Configuration
current_time = datetime.now()
Settings.llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo", api_key="6f81d9a1dc6d93c9105d5827b3fc9c9717d24e462fc04c74369d49ab85dc03b6"
)

# Supabase Vector Store Configuration
supabase_url = "your-supabase-url"
supabase_key = "your-supabase-key"
vector_store = SupabaseVectorStore(
    postgres_connection_string=f"postgresql://postgres.ckwxnombsirouyhkmxiu:UKlsp51ZGLXmZ662@aws-0-ap-south-1.pooler.supabase.com:6543/postgres",
    table_name="museums"
)

# Create the index with the Supabase vector store
index = VectorStoreIndex.from_vector_store(vector_store)

# Create the prompt template
character_creation_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            """You are an AI that will handle Museum ticket booking requests by the user.
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

               7. Use maps and provide shortest route through cab, train, walking and metro train and flight if needed with traveling time from the user's location to the given museum address and provide a path for selectedd mode of transport.

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
               13. Be flexible and allow booking changes OR cancellations: Changes to Museum choice, number of adults or children, date or time or optionals
               14. Do not print bold or italics or any other fancy formatting. Just plain text will do.
               15. Keep your responses short and crisp. No puns, use simple words and be proffestional.
               16. Important key note: Expect multi-lingual input. For example: hindi, gujarati, marathi, spanish, german, french, russian. also expect the languages to be wriiten in english characters.
               If user wants to speak in a particular language, respond purely in that language using that language's script and characters.

               Ask only one question at a time.

               Ask user to select the fligts from given real time flights from google, ask user to select the Cab from given real time Cabs from google and ask user to select the train and its classes and to change the station as well from given real time trainn tickets from google ask the user twice before booking traveling tickets and include prices of each tickets.
                Please ask the date before booking traveling tickets.
                Ask the email ID for booking traveling and museum tickets and payments.
               Expect multiple information in one response. See which information is missing and ask for it accordingly.
               
               Use the chat history to maintain continuity:
               {history}
            """
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            """{question}"""
        ),
    ),
]

character_creation_template = ChatPromptTemplate(character_creation_msgs)

# In-memory history for each session
session_histories = {}

# Create a Pydantic model for request body
class ChatRequest(BaseModel):
    user_input: str
    session_id: str


# Define an endpoint to handle chat interactions
@app.post("/chat/")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_input = request.user_input

    # Fetch chat history or start a new one
    chat_history = session_histories.get(session_id, "")
    
    # Query the vector store
    query_engine = index.as_query_engine()
    response = query_engine.query(user_input)
    
    # Get response from LLM
    llm_response = Settings.llm.stream_complete(
        character_creation_template.format(question=user_input, history=chat_history, context=response.response)
    )
    
    full_response = ""
    for r in llm_response:
        full_response += r.delta

    full_response = re.sub(r" (\d)\. ", r"\n \1. ", full_response)

    # Update chat history
    chat_history += f"User: {user_input}\nBot: {full_response}\n"
    session_histories[session_id] = chat_history

    return {"response": full_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
