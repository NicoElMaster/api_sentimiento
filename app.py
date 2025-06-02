from flask import Flask, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')  # solo la primera vez
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"mensaje": "API de análisis de sentimientos activa ✅"})

@app.route('/analizar', methods=['POST'])
def analizar_sentimiento():
    data = request.get_json()
    texto = data.get("texto", "")
    
    if not texto:
        return jsonify({"error": "Texto vacío"}), 400

    resultado = analyzer.polarity_scores(texto)
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
