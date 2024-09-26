import wikipedia
from typing import List, Dict
import os
from openai import OpenAI
import openai
from supabase import create_client, Client

def scrape_indian_museums() -> List[Dict[str, str]]:
    museums = []
    # Search for a list of museums in India
    search_results = wikipedia.search("List of museums in India", results=1)
    if not search_results:
        print("Couldn't find a page about museums in India")
        return museums
    
    try:
        # Get the page about museums in India
        page = wikipedia.page(search_results[0])
        # Split the content into lines
        lines = page.content.split('\n')
        current_museum = {}
        for line in lines:
            if line.strip().startswith('==') and not line.strip().startswith('==='):
                # This line is likely a museum name
                if current_museum:
                    museums.append(current_museum)
                current_museum = {"name": line.strip().replace('==', '').strip()}
            elif ':' in line:
                # This line might contain location or feature information
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ['location', 'city', 'state']:
                    current_museum['location'] = value
                elif key not in ['established', 'website']:  # Exclude some common irrelevant keys
                    if 'features' not in current_museum:
                        current_museum['features'] = []
                    current_museum['features'].append(f"{key}: {value}")
        
        # Add the last museum
        if current_museum:
            museums.append(current_museum)
    
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e.options}")
    except wikipedia.exceptions.PageError:
        print("Page not found for museums in India")
    
    return museums

# Configure OpenAI and Supabase
openai.api_key = "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"
supabase_url = "https://ckwxnombsirouyhkmxiu.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkNzQwZjQwZi1mZjQwLTQwZjUtYjIwZi1mZjQwZjUwZjQwZjUiLCJpYXQiOjE2MzQwNjYwNzIsInVzZXJfaWQiOiJkNzQwZjQwZi1mZjQwLTQwZjUtYjIwZi1mZjQwZjUwZjQwZjUiLCJlbWFpbCI6ImFkbWluQGdtYWlsLmNvbSIsInJvbGUiOiJ1c2VyIn0.7"
supabase: Client = create_client(supabase_url, supabase_key)

# OPENAI_API_KEY = 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
# SUPABASE_KEY = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkNzQwZjQwZi1mZjQwLTQwZjUtYjIwZi1mZjQwZjUwZjQwZjUiLCJpYXQiOjE2MzQwNjYwNzIsInVzZXJfaWQiOiJkNzQwZjQwZi1mZjQwLTQwZjUtYjIwZi1mZjQwZjUwZjQwZjUiLCJlbWFpbCI6ImFkbWluQGdtYWlsLmNvbSIsInJvbGUiOiJ1c2VyIn0.7

client = OpenAI(api_key=openai.api_key)

def generate_embedding(text: str) -> List[float]:
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def store_in_supabase(museums: List[Dict[str, str]]):
    for museum in museums:
        content = f"{museum.get('name', '')} - {museum.get('location', '')}"
        if 'features' in museum:
            content += " " + " ".join(museum['features'])
        
        embedding = generate_embedding(content)
        supabase.table("indian_museums").insert({
            "name": museum.get('name', ''),
            "location": museum.get('location', ''),
            "features": museum.get('features', []),
            "content": content,
            "embedding": embedding
        }).execute()

# Example usage
indian_museums = scrape_indian_museums()

# Store the scraped data in Supabase
store_in_supabase(indian_museums)

# Print the results
for museum in indian_museums:
    print(f"Name: {museum.get('name', 'N/A')}")
    print(f"Location: {museum.get('location', 'N/A')}")
    print("Features:")
    for feature in museum.get('features', []):
        print(f" - {feature}")
    print()
