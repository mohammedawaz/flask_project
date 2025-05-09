# responder.py
import json
import os
from core.web_scraper import scrape_google, scrape_wikipedia
from core.neural_network import SimpleNeuron, train_model, save_model, load_model, predict
from core.nlp_utils import clean_text
import torch
from core.semantic_memory import SemanticMemory


DB_PATH = "query_database.json"

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_database(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)

def process_query(query):
    semantic_memory = SemanticMemory()

    # 1. Semantic search
    result = semantic_memory.search(query)
    if result:
        return f"[From memory]\n{result['response']} (source: {result['source']})"

    # 2. Exact match from database
    database = load_database()
    if query in database:
        return database[query]["response"]

    # 3. Web scrape + process
    wiki = scrape_wikipedia(query)
    google_data = scrape_google(query)
    all_text = wiki + ' ' + ' '.join(google_data)
    cleaned = clean_text(all_text)

    input_data = [[float(len(cleaned))]]
    model = SimpleNeuron(input_size=1)
    train_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    labels = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    trained_model = train_model(model, train_data, labels)
    save_model(trained_model)
    loaded_model = load_model(SimpleNeuron(1))
    prediction = predict(loaded_model, input_data)

    response = f"Based on my research, here's what I found: {wiki[:300]}... (predicted value: {prediction})"
    
    # Save to local DB
    database[query] = {"response": response}
    save_database(database)

    # Save to semantic memory
    semantic_memory.add_entry(query, response, source="web_scraping")

    return response


