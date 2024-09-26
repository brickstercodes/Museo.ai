import wikipedia
from typing import List, Dict
import os
import google.generativeai as genai
from supabase import create_client, Client
import csv

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_embedding(text: str) -> list:
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="Embedding for vector database"
    )
    return embedding['embedding']

def scrape_wikipedia(topic: str) -> List[Dict[str, str]]:
    items = []
    search_results = wikipedia.search(topic, results=1)
    if not search_results:
        print(f"Couldn't find a page about {topic}")
        return items
    try:
        page = wikipedia.page(search_results[0])
        content = page.content
        items.append({
            "title": page.title,
            "content": content,
            "source": "wikipedia"
        })
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e.options}")
    except wikipedia.exceptions.PageError:
        print(f"Page not found for {topic}")
    return items

def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    items = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['source'] = 'csv'
            items.append(row)
    return items

def store_in_supabase(items: List[Dict[str, str]], table_name: str):
    for item in items:
        content = f"{item.get('title', '')} {item.get('content', '')}"
        embedding = generate_embedding(content)
        supabase.table(table_name).insert({
            "title": item.get('title', ''),
            "content": item.get('content', ''),
            "source": item.get('source', ''),
            "embed": embedding
        }).execute()

def main():
    table_name = "table1"  # Use the existing table name

    # Choose data source
    source = input("Choose data source (wikipedia/csv): ").lower()

    if source == "wikipedia":
        topic = input("Enter the topic to search on Wikipedia: ")
        items = scrape_wikipedia(topic)
    elif source == "csv":
        file_path = input("Enter the path to your CSV file: ")
        items = read_csv_file(file_path)
    else:
        print("Invalid source. Please choose 'wikipedia' or 'csv'.")
        return

    store_in_supabase(items, table_name)
    
    # Print the results
    for item in items:
        print(f"Title: {item.get('title', 'N/A')}")
        print(f"Content: {item.get('content', 'N/A')[:100]}...")  # Print first 100 characters
        print(f"Source: {item.get('source', 'N/A')}")
        print()

if __name__ == "__main__":
    main()
