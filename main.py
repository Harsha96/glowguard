# main code
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
app = Flask(__name__)
CORS(app)

# Load typo correction map
try:
    with open("typos.json") as f:
        TYPO_MAP = json.load(f)
except FileNotFoundError:
    TYPO_MAP = {}
    print("Warning: typos.json not found. Typo correction disabled.")

print("Loading GlowGuard knowledge base...")
emb_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./glowguard_db")
collection = client.get_collection("skincare_kb")
print("API ready!")

def correct_typos(text: str) -> str:
    """Fix typos in terms that appear in your knowledge base."""
    words = text.split()
    corrected = []
    for word in words:
        clean = word.lower().strip(".,!?")
        if clean in TYPO_MAP:
            corrected.append(TYPO_MAP[clean])
        else:
            corrected.append(word)
    return " ".join(corrected)

def clean_context(text: str) -> str:
    clean = text.split("##")[0].strip()
    return "\n".join([line.strip() for line in clean.split("\n") if line.strip()])

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Please provide a 'question' field."}), 400
    question = data['question'].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    corrected = correct_typos(question)
    query_emb = emb_model.encode(corrected).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    if not results['documents'][0] or not results['documents'][0][0].strip():
        return jsonify({
            "best_answer": "I couldn't find reliable information on this topic in my knowledge base.",
            "source": "None",
            "url": "",
            "related_articles": [],
            "corrected_question": corrected
        })

    # Best answer = top result
    best_doc = results['documents'][0][0]
    best_source = results['metadatas'][0][0].get('source', 'Unknown')
    best_url = results['metadatas'][0][0].get('url', '').strip()

    best_answer = clean_context(best_doc)

    # Related articles = next 2 non-empty results
    related = []
    for i in range(1, len(results['documents'][0])):
        doc = results['documents'][0][i]
        if doc.strip():
            related.append({
                "answer": clean_context(doc),
                "source": results['metadatas'][0][i].get('source', 'Unknown'),
                "url": results['metadatas'][0][i].get('url', '').strip()
            })
        if len(related) == 2:
            break

    return jsonify({
        "best_answer": best_answer,
        "source": best_source,
        "url": best_url,
        "related_articles": related,
        "corrected_question": corrected
    })

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "GlowGuard API is running!"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)