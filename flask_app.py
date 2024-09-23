from flask import Flask, request, jsonify
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import io

app = Flask(__name__)

# Load the VILT processor and model for visual question answering
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


@app.route('/vqa', methods=['POST'])
def vqa():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({"error": "Image and question are required"}), 400

    try:
        # Get the image from the request
        image_file = request.files['image']
        image = Image.open(image_file)

        # Get the question from the request
        question = request.form['question']

        # Process the image and question
        encoding = processor(image, question, return_tensors="pt")

        # Forward pass through the model
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()

        # Get the predicted answer
        answer = model.config.id2label[idx]

        # Return the answer as a JSON response
        return jsonify({"question": question, "answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
