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

# Sidebar info with custom profile section
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <style>
        .custom-sidebar {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            width: 650px;
            padding: 10px;
        }
        .profile-container {
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            width: 100%;
        }
        .profile-image {
            width: 200px;
            height: auto;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            margin-right: 15px;
        }
        .profile-details {
            font-size: 14px;
            width: 100%;
        }
        .profile-details h3 {
            margin: 0 0 10px;
            font-size: 18px;
            color: #333;
        }
        .profile-details p {
            margin: 10px 0;
            display: flex;
            align-items: center;
        }
        .profile-details a {
            text-decoration: none;
            color: #1a73e8;
        }
        .profile-details a:hover {
            text-decoration: underline;
        }
        .icon-img {
            width: 18px;
            height: 18px;
            margin-right: 6px;
        }
    </style>

    <div class="custom-sidebar">
        <div class="profile-container">
            <img class="profile-image" src="https://res.cloudinary.com/dwhfxqolu/image/upload/v1744014185/pnhnaejyt3udwalrmnhz.jpg" alt="Profile Image">
            <div class="profile-details">
                <h3>👨‍💻 Developed by:<br> Tahir Abbas Shaikh</h3>
                <p>
                    <img class="icon-img" src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail">
                    <strong>Email:</strong> <a href="mailto:tahirabbasshaikh555@gmail.com">tahirabbasshaikh555@gmail.com</a>
                </p>
                <p>📍 <strong>Location:</strong> Sukkur, Sindh, Pakistan</p>
                <p>
                    <img class="icon-img" src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="GitHub">
                    <strong>GitHub:</strong> <a href="https://github.com/Tahir-Abbas-555" target="_blank">Tahir-Abbas-555</a>
                </p>
                <p>
                    <img class="icon-img" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace">
                    <strong>HuggingFace:</strong> <a href="https://huggingface.co/Tahir5" target="_blank">Tahir5</a>
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Button to trigger the Visual Question Answering (VQA) task
if st.button("🧠 Generate Answer"):
    if uploaded_image is None:
        st.error("🚫 **No Image Uploaded**\n\nPlease upload an image to proceed with visual question answering.")
    elif question.strip() == "":
        st.error("❓ **Question Missing**\n\nPlease enter a valid question related to the uploaded image.")
    else:
        try:
            # Load and display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

            # Encode image and question
            st.info("🔄 Processing image and question using the VQA model...")
            encoding = processor(image, question, return_tensors="pt")

            # Run model inference
            outputs = model(**encoding)
            logits = outputs.logits
            predicted_index = logits.argmax(-1).item()

            # Decode predicted answer
            answer = model.config.id2label[predicted_index]

            # Display answer with success message
            st.success("✅ **Answer Generated Successfully**")

            # Beautiful answer block with dark/light mode support
            st.markdown(f"""
                <div style='
                    background-color: rgba(26, 115, 232, 0.2);
                    border-left: 6px solid #1a73e8;
                    padding: 1rem;
                    margin-top: 0.75rem;
                    border-radius: 8px;
                    font-size: 1.15rem;
                    font-weight: 600;
                    color: #ffffff;
                '>
                    📌 <strong>Predicted Answer:</strong> {answer}
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ **An unexpected error occurred:**\n\n`{str(e)}`\n\nPlease ensure the uploaded image is valid and the model is properly loaded.")
