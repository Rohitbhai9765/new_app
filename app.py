from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the Question-Answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.route("/answer", methods=["POST"])
def answer_question():
    try:
        # Extract question and context from the JSON body
        data = request.json
        question = data.get("question")
        context = data.get("context")

        if not question or not context:
            return jsonify({"error": "Both 'question' and 'context' fields are required"}), 400

        # Get the answer from the model
        result = qa_pipeline(question=question, context=context)
        return jsonify({"answer": result['answer'], "score": result['score']})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
