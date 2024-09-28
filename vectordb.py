import wikipedia
from typing import List, Dict, Union
import os
import google.generativeai as genai
from supabase import create_client, Client
import csv
from dotenv import load_dotenv
import asyncio

load_dotenv("/Users/anugrahshetty/Desktop/Local e3/apiKeys.env")

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Supabase
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

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
        # Optionally let the user choose or retry
        choice = input("Choose one of the options above or type 'retry' to enter a new topic: ").strip()
        if choice.lower() == 'retry':
            new_topic = input("Enter the new topic to search on Wikipedia: ")
            return scrape_wikipedia(new_topic)
        elif choice in e.options:
            return scrape_wikipedia(choice)
        else:
            print("Invalid choice. Please try again.")
    except wikipedia.exceptions.PageError:
        print(f"Page not found for {topic}")
    return items

def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    items = []
    if not os.path.exists(file_path):
        print(f"CSV file not found at {file_path}. Please enter a valid path.")
        return items

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Creating a concatenated content for embedding
            content = (f"Museum Name: {row['Museum Name']}, Location: {row['Location']}, City: {row['City']}, "
                       f"Type: {row['Type']}, Year Established: {row['Year Established']}, Visitors per Year: {row['Visitors per Year']}, "
                       f"Child Entry Fee: {row['Child Entry Fee']}, Adult Entry Fee: {row['Adult Entry Fee']}, "
                       f"Opening Time: {row['Opening Time']}, Closing Time: {row['Closing Time']}")
            
            items.append({
                "title": row['Museum Name'],  # The museum name will be the 'title'
                "content": content,            # All combined information will be the 'content'
                "source": 'csv'                # Mark it as a CSV source
            })
    return items

async def store_in_supabase(item: Union[str, List[Dict[str, str]]], embedding: Union[list, None] = None, table_name: str = "table1"):
    try:
        if isinstance(item, str):
            # Single item case (used in chat)
            data = {
                "content": item,
                "embed": embedding,
                "source": "chat"
            }
            result = supabase.table(table_name).insert(data).execute()
            return {"success": True, "data": result.data}
        elif isinstance(item, list):
            # Multiple items case (used in data ingestion)
            data = []
            for i in item:
                content = f"{i.get('title', '')} {i.get('content', '')}"
                embed = generate_embedding(content) if embedding is None else embedding
                data.append({
                    "title": i.get('title', ''),
                    "content": i.get('content', ''),
                    "source": i.get('source', ''),
                    "embed": embed
                })
            result = supabase.table(table_name).insert(data).execute()
            return {"success": True, "data": result.data}
        else:
            raise ValueError("Invalid input type. Expected string or list of dictionaries.")
    except Exception as e:
        return {"success": False, "error": str(e)}

async def main():
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

    result = await store_in_supabase(items, table_name=table_name)
    if result["success"]:
        print("Data successfully stored in Supabase")
    else:
        print(f"Failed to store data in Supabase: {result['error']}")
    
    # Print the results
    for item in items:
        print(f"Title: {item.get('title', 'N/A')}")
        print(f"Content: {item.get('content', 'N/A')[:100]}...")  # Print first 100 characters
        print(f"Source: {item.get('source', 'N/A')}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
