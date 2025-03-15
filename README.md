# Image-Based Question Answering System

## Overview
This repository contains two projects:
1. **Complete Web Application** – A full-stack web app built using Streamlit for both frontend and backend.
2. **Flask API Backend** – A standalone Flask-based backend API.

Both implementations allow users to upload an image and ask questions about it. The system uses the **dandelin/vilt-b32-finetuned-vqa** model to analyze and respond to queries based on the provided image.

## Features
- Users can upload an image.
- Users can ask questions related to the uploaded image.
- The model processes the image and answers questions based on its content.
- Two implementations:
  - **Streamlit Web App:** A complete frontend and backend application.
  - **Flask API:** A RESTful API for backend processing.

## Technology Stack
- **Frontend:** Streamlit (for the web app UI)
- **Backend:** Flask (for the API)
- **Model:** `dandelin/vilt-b32-finetuned-vqa`
- **Libraries:** PyTorch, Transformers, Pillow, OpenCV, Requests

---

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/image-vqa.git
cd image-vqa
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web App
```bash
streamlit run stream.py
```

### 4. Run the Flask API
```bash
python flask_app.py
```

---

## API Endpoints (For Flask Backend)
### 1. Visual Question Answering (VQA)
**Endpoint:** `POST /vqa`
- **Description:** Processes an image and a question, returning an answer.
- **Request Format:** Multipart form-data
  - `image`: The uploaded image file.
  - `question`: The question related to the image.
- **Response Format:** JSON

**Example Request (cURL):**
```bash
curl -X POST "http://127.0.0.1:5000/vqa" \
     -F "image=@path/to/image.jpg" \
     -F "question=What is in the image?"
```

**Example Response:**
```json
{
  "question": "What is in the image?",
  "answer": "A cat sitting on a table."
}
```

---

## Testing with Postman
### Steps to Test the Flask API in Postman
1. Open **Postman**.
2. Select **POST** request.
3. Enter the request URL: `http://127.0.0.1:5000/vqa`.
4. Navigate to the **Body** tab and select **form-data**.
5. Add two key-value pairs:
   - **Key:** `image` → Select an image file.
   - **Key:** `question` → Enter a text question related to the image.
6. Click **Send**.
7. View the response containing the model's answer in JSON format.

---

## Example Usage
### Streamlit Web App
1. Open the app in the browser.
2. Upload an image.
3. Enter a question.
4. View the model's response.

### Flask API
1. Send a `POST` request to `/vqa` with an image and a question.
2. Receive the model-generated answer in JSON format.

---

## Model Information
- **Name:** `dandelin/vilt-b32-finetuned-vqa`
- **Functionality:** Vision-and-Language Transformer (ViLT) model fine-tuned for Visual Question Answering (VQA).
- **Source:** [Hugging Face Model Hub](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)

---

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

---

## License
This project is licensed under the MIT License.

