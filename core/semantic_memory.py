# core/semantic_memory.py

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

KB_PATH = "knowledge_base.json"
EMBEDDING_DIM = 384  # Model-specific (MiniLM)

model = SentenceTransformer('all-MiniLM-L6-v2')

class SemanticMemory:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.queries = []
        self.responses = []
        self.sources = []
        self.load_knowledge()

    def load_knowledge(self):
        if not os.path.exists(KB_PATH):
            with open(KB_PATH, 'w') as f:
                json.dump([], f)

        with open(KB_PATH, 'r') as f:
            data = json.load(f)

        self.queries = [item['query'] for item in data]
        self.responses = [item['response'] for item in data]
        self.sources = [item['source'] for item in data]

        if self.queries:
            embeddings = model.encode(self.queries)
            self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query, top_k=1):
        if not self.queries:
            return None

        query_embedding = model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        if distances[0][0] < 0.5:  # Threshold for similarity
            idx = indices[0][0]
            return {
                "query": self.queries[idx],
                "response": self.responses[idx],
                "source": self.sources[idx],
                "score": distances[0][0]
            }
        return None

    def add_entry(self, query, response, source):
        new_entry = {
            "query": query,
            "response": response,
            "source": source,
            "tags": [],
            "date": datetime.now().isoformat()
        }

        with open(KB_PATH, 'r') as f:
            data = json.load(f)
        data.append(new_entry)

        with open(KB_PATH, 'w') as f:
            json.dump(data, f, indent=4)

        self.queries.append(query)
        self.responses.append(response)
        self.sources.append(source)

        emb = model.encode([query]).astype('float32')
        self.index.add(emb)
