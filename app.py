import streamlit as st
from groq import Groq
import base64
from PIL import Image
import io
import os

# Your Groq API key (hardcoded for now - replace with your actual key)
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to classify MRI image using Groq API
def classify_mri_image(image):
    base64_image = encode_image(image)
    
    # Prompt for the Groq API
    prompt = "Analyze this MRI image and determine if it shows a brain tumor. Provide a clear classification (e.g., 'Tumor detected' or 'No tumor detected') and a brief explanation on why it has tunour. Be very brief about it and straightforward."
    
    # Call Groq API with Llama 3.2-90B Vision Preview
    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=300
    )
    
    # Extract the response
    result = response.choices[0].message.content
    return result

# Streamlit app
def main():
    st.title("MRI Brain Tumor Classifier")
    st.write("Upload an MRI image to classify whether it contains a brain tumor.")
    
    # File uploader for MRI image
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)
        
        # Add a button to trigger classification
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                try:
                    # Get classification result
                    result = classify_mri_image(image)
                    st.success("Classification Complete!")
                    st.write("### Result:")
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()