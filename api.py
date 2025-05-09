from flask import Flask, request, jsonify
import pyttsx3
from core.responder import process_query  # ✅ Import your actual AI logic

app = Flask(__name__)

def speak(text):
    """Safely speak using a fresh engine per call."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS ERROR] {e}")

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    query = data.get('query', '')
    response = process_query(query)  # ✅ Use the imported logic
    speak(response)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# from flask import Flask, request, jsonify
# import pyttsx3

# app = Flask(__name__)

# def speak(text):
#     """Safely speak using a fresh engine per call"""
#     try:
#         engine = pyttsx3.init()
#         engine.say(text)
#         engine.runAndWait()
#         engine.stop()
#     except Exception as e:
#         print(f"[TTS ERROR] {e}")

# def process_query(query):
#     """Basic mock AI logic"""
#     if 'hello' in query.lower():
#         return "Hi there!"
#     elif 'your name' in query.lower():
#         return "I'm ARC, your assistant."
#     else:
#         return "Sorry, I don't understand that yet."

# @app.route('/process', methods=['POST'])
# def process():
#     data = request.get_json()
#     query = data.get('query', '')
#     response = process_query(query)
#     speak(response)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

