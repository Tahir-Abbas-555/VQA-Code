import streamlit as st
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# Load the VILT processor and model for visual question answering
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Streamlit app UI
st.title("Visual Question Answering (VQA) with VILT")

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Question input
question = st.text_input("Enter your question about the image:")

# A button to trigger the VQA task
if st.button("Get Answer"):
    if uploaded_image is None:
        st.error("Please upload an image.")
    elif question == "":
        st.error("Please enter a question.")
    else:
        try:
            # Load the image from the uploader
            image = Image.open(uploaded_image)

            # Show the uploaded image in the app
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process the image and question
            encoding = processor(image, question, return_tensors="pt")

            # Forward pass through the model
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()

            # Get the predicted answer
            answer = model.config.id2label[idx]

            # Show the answer
            st.success(f"Predicted Answer: {answer}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
