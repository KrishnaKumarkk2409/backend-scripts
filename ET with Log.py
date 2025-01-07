from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json
import openai
import os
import time
from datetime import datetime
import csv
import pandas as pd

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ask the user to input the CSV filename
csv_filename = input("Please enter the name of your CSV file (including .csv extension): ")

# Read the CSV file
try:
    df = pd.read_csv(csv_filename)
    # Convert DataFrame to list of dictionaries and ensure all required columns exist
    required_columns = ['Root Node', 'Root Link', 'P1 Name', 'P1 Link', 
                       'P2 Name', 'P2 Link', 'P3 Name', 'P3 Link',
                       'P4 Name', 'P4 Link', 'Leaf name', 'Leaf Link']
    
    if not all(col in df.columns for col in required_columns):
        print("Error: CSV file is missing required columns. Please ensure all required columns exist:")
        print(required_columns)
        exit(1)
        
    leaf_data = df[required_columns].to_dict('records')
except FileNotFoundError:
    print(f"Error: File '{csv_filename}' not found in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Set up headless browser options
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Initialize the browser
driver = webdriver.Chrome(options=chrome_options)

# Function to scrape text from a given URL
def scrape_text(url):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            driver.get(url)
            time.sleep(50)  # Consider making this dynamic based on page load
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            main_content = soup.find('div', {'class': 'ArticleDetailLeftContainer__box'})
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                return text
            else:
                print(f"Warning: No content found for {url}")
                return ""
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to scrape {url} after {max_retries} attempts")
                return ""
            time.sleep(30)  # Wait before retry

def chunk_text(text, chunk_size=800):
    if not text or not isinstance(text, str):
        print("Warning: Invalid text input for chunking")
        return []
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [chunk for chunk in chunks if chunk.strip()]

def embed_text_openai(text):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response["data"][0]["embedding"]
        except openai.error.RateLimitError:
            if attempt == max_retries - 1:
                print("Rate limit reached, failing after max retries")
                return None
            print("Rate limit reached, waiting 60 seconds...")
            time.sleep(60)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

def save_embeddings_to_json(embeddings, file_count):
    folder_path = "Chunks"
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"embeddings_batch_{file_count}.json"
    file_path = os.path.join(folder_path, file_name)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(embeddings, json_file, ensure_ascii=False, indent=4)
        print(f"Saved {len(embeddings)} embeddings to {file_path}")
    except Exception as e:
        print(f"Error saving embeddings to JSON: {e}")

# Create a log CSV file
def create_log_file():
    log_filename = 'processing_log.csv'
    if not os.path.exists(log_filename):
        with open(log_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Leaf name', 'Leaf Link', 'Chunk Status', 'Embedding Status', 'Num Chunks', 'Timestamp'])
    return log_filename

# Modified log function to write to CSV
def log_to_csv(log_filename, leaf_name, leaf_link, chunk_status, embedding_status, num_chunks, timestamp):
    try:
        with open(log_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([leaf_name, leaf_link, chunk_status, embedding_status, num_chunks, timestamp])
        print(f"Updated log for {leaf_name}: Chunk:{chunk_status}, Embedding:{embedding_status}, Chunks:{num_chunks}, Time:{timestamp}")
    except Exception as e:
        print(f"Error updating log in CSV: {e}")

def scrape_chunk_and_embed(leaf_data):
    total_leaves = len(leaf_data)
    print(f"Starting processing of {total_leaves} leaves...")
    
    embeddings_batch = []
    batch_size = 50
    file_count = 1
    id_counter = 1
    log_filename = create_log_file()

    for idx, leaf in enumerate(leaf_data, start=1):
        print(f"Processing leaf {idx} of {total_leaves} ({(idx/total_leaves)*100:.1f}%)")
        # Get all metadata from the current leaf record
        leaf_name = leaf.get('Leaf name')
        leaf_link = leaf.get('Leaf Link')
        root_name = leaf.get('Root name')
        root_link = leaf.get('Root Link')
        p1_name = leaf.get('P1 name')
        p1_link = leaf.get('P1 Link')
        p2_name = leaf.get('P2 name')
        p2_link = leaf.get('P2 Link')
        p3_name = leaf.get('P3 name')
        p3_link = leaf.get('P3 Link')
        p4_name = leaf.get('P4 name')
        p4_link = leaf.get('P4 Link')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if leaf_link and leaf_link != 'No Leaf Link':
            print(f"Scraping data from: {leaf_link}")
            scraped_text = scrape_text(leaf_link)

            if scraped_text:
                chunks = chunk_text(scraped_text, chunk_size=800)
                print(f"Data for {leaf_name} broken into {len(chunks)} chunks.")
                chunk_status = "YES"
                log_to_csv(log_filename, leaf_name, leaf_link, chunk_status, "PENDING", len(chunks), timestamp)

                for chunk in chunks:
                    embedding = embed_text_openai(chunk)
                    if embedding:
                        embeddings_batch.append({
                            "id": id_counter,
                            "combined_chunk": f"Root: {root_name}\nP1: {p1_name}\nP2: {p2_name}\nP3: {p3_name}\nP4: {p4_name}\nLeaf: {leaf_name}\nChunk: {chunk}",
                            "embedding": embedding,
                            "metadata": {
                                "root_name": root_name,
                                "root_link": root_link,
                                "p1_name": p1_name,
                                "p1_link": p1_link,
                                "p2_name": p2_name,
                                "p2_link": p2_link,
                                "p3_name": p3_name,
                                "p3_link": p3_link,
                                "p4_name": p4_name,
                                "p4_link": p4_link,
                                "leaf_name": leaf_name,
                                "leaf_link": leaf_link
                            }
                        })
                        id_counter += 1

                        if len(embeddings_batch) >= batch_size:
                            save_embeddings_to_json(embeddings_batch, file_count)
                            file_count += 1
                            embeddings_batch = []
                            print("Pausing for 10 minutes to avoid IP restrictions...")
                            time.sleep(600)  # 10 minutes pause
                            print("Resuming operations...")

                log_to_csv(log_filename, leaf_name, leaf_link, chunk_status, "YES", len(chunks), timestamp)
            else:
                log_to_csv(log_filename, leaf_name, leaf_link, "NO", "NO", 0, timestamp)
        else:
            log_to_csv(log_filename, leaf_name, leaf_link, "NO", "NO", 0, timestamp)

    # Save any remaining embeddings in the batch
    if embeddings_batch:
        save_embeddings_to_json(embeddings_batch, file_count)

# Main execution
try:
    scrape_chunk_and_embed(leaf_data)
finally:
    driver.quit()
    print("Browser session closed successfully") 