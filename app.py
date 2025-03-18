# app.py
import streamlit as st
import cv2
import torch
import numpy as np
import base64
import requests
import os
import io
import time
from PIL import Image
from gtts import gTTS
from openai import OpenAI
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Vision Assistant - Real-time Image Captioning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SSL Certificate handling - fixes "OSError: [Errno 22] Invalid argument" error
# Unset problematic SSL environment variables if they exist
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
if "SSL_CERT_DIR" in os.environ:
    del os.environ["SSL_CERT_DIR"]

# Initialize OpenAI client with proper error handling
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Check if API key exists
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please add it to your .env file as OPENAI_API_KEY=your_key_here")
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    st.info("If you're seeing SSL certificate errors, try setting these environment variables in your .env file:")
    st.code("PYTHONHTTPSVERIFY=0\nREQUESTS_CA_BUNDLE=\nSSL_CERT_FILE=")
    client = None

def get_image_caption(image):
    """
    Use OpenAI's Vision API to generate a detailed caption for the image.
    """
    if client is None:
        return "Error: OpenAI client not initialized. Please check your API key and SSL settings."
        
    # Convert numpy array to bytes
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer)
    
    # Get the selected model from the sidebar, or use fallback list
    if "selected_model" in st.session_state:
        vision_models = [st.session_state.selected_model]
    else:
        # Updated list of models to try (current as of March, 2025)
        vision_models = [
            "gpt-4o",              # GPT-4o model that supports vision
            "gpt-4-turbo-vision",  # Potential turbo model name for vision
            "gpt-4-vision-2024",   # Potential newer model name
            "gpt-4-turbo",         # Another option
            "gpt-4-1106-vision-preview"  # Older but possibly still working
        ]
    
    last_error = None
    
    # Try each model in order until one works
    for model in vision_models:
        try:
            st.info(f"Attempting to use model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a detailed image descriptor for visually impaired users. Describe the image focusing on key elements, spatial relationships, colors, people, actions, text, and important contextual details. Be concise but thorough.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image for a visually impaired person:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(io_buf.getvalue()).decode('utf-8')}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            # If we reach here, the model worked
            # Store the working model for future use
            st.session_state.working_model = model
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            st.warning(f"Model {model} failed: {last_error}")
            continue  # Try the next model
    
    # If we get here, all models failed
    error_msg = last_error
    st.error(f"All vision models failed. Last error: {error_msg}")
    
    # Provide more helpful error messages for common issues
    if "api_key" in error_msg.lower():
        return "Error: Invalid or missing OpenAI API key. Please check your .env file."
    elif "ssl" in error_msg.lower() or "certificate" in error_msg.lower():
        return "Error: SSL certificate validation failed. Please check the troubleshooting section in the sidebar."
    elif "model_not_found" in error_msg.lower() or "deprecated" in error_msg.lower():
        return "Error: All available vision models failed. Your OpenAI account might not have access to the required models."
    else:
        return f"Error generating caption: {error_msg}"

def text_to_speech(text):
    """
    Convert text to speech using Google's TTS API
    """
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
            tts.save(temp_filename)
            
        return temp_filename
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

def autoplay_audio(file_path):
    """
    Autoplay the audio file
    """
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

def main():
    st.title("Vision Assistant")
    st.subheader("Real-time Image Captioning for the Visually Impaired")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # Model selector in sidebar
    available_models = [
        "gpt-4o",
        "gpt-4-turbo-vision",
        "gpt-4-vision-2024",
        "gpt-4-turbo",
        "gpt-4-1106-vision-preview"
    ]
    
    # Use the previously working model if available
    default_model = st.session_state.get("working_model", "gpt-4-vision")
    selected_model = st.sidebar.selectbox(
        "OpenAI Vision Model", 
        options=available_models,
        index=available_models.index(default_model) if default_model in available_models else 0
    )
    
    # Store the selected model in session state
    st.session_state.selected_model = selected_model
    
    # Troubleshooting section in sidebar
    with st.sidebar.expander("Troubleshooting"):
        st.write("### Common Issues")
        st.write("#### SSL Certificate Errors")
        st.write("If you're experiencing SSL certificate errors, add these lines to your .env file:")
        st.code("PYTHONHTTPSVERIFY=0\nREQUESTS_CA_BUNDLE=\nSSL_CERT_FILE=")
        st.write("#### API Key Issues")
        st.write("Make sure your OpenAI API key is correctly set in the .env file:")
        st.code("OPENAI_API_KEY=your_key_here")
        st.write("#### Windows-specific SSL Issues")
        st.write("On Windows, you might need to set additional environment variables:")
        st.code("set PYTHONHTTPSVERIFY=0")
        st.write("Or try installing the certifi package:")
        st.code("pip install certifi")
    
    # Image source selection
    image_source = st.sidebar.radio("Select Image Source", ["Webcam", "Upload Image"])
    
    # Detail level for captions
    detail_level = st.sidebar.slider("Caption Detail Level", 1, 5, 3)
    
    # Voice settings
    voice_speed = st.sidebar.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)
    
    # Auto-captioning interval (for webcam)
    if image_source == "Webcam":
        auto_caption = st.sidebar.checkbox("Enable Auto-Captioning", value=False)
        if auto_caption:
            caption_interval = st.sidebar.slider("Caption Interval (seconds)", 2, 30, 10)
    
    # Main area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        if image_source == "Webcam":
            st.write("Camera Feed")
            img_placeholder = st.empty()
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Please check your camera settings.")
                return
            
            # Take snapshot button
            if st.button("Take Snapshot"):
                ret, frame = cap.read()
                if ret:
                    img_placeholder.image(frame, channels="BGR", use_container_width=True)
                    
                    with st.spinner("Generating caption..."):
                        caption = get_image_caption(frame)
                        st.session_state.last_caption = caption
                        st.session_state.last_frame = frame
                    
                    # Convert caption to speech
                    with st.spinner("Converting to speech..."):
                        speech_file = text_to_speech(caption)
                        if speech_file:
                            autoplay_audio(speech_file)
            
            # Webcam feed display
            while True:
                ret, frame = cap.read()
                if ret:
                    img_placeholder.image(frame, channels="BGR", use_container_width=True)
                    
                    # If auto-captioning is enabled, periodically generate captions
                    if auto_caption and 'last_caption_time' not in st.session_state:
                        st.session_state.last_caption_time = time.time()
                    
                    if auto_caption and time.time() - st.session_state.last_caption_time > caption_interval:
                        with st.spinner("Auto-generating caption..."):
                            caption = get_image_caption(frame)
                            st.session_state.last_caption = caption
                            st.session_state.last_frame = frame
                            st.session_state.last_caption_time = time.time()
                        
                        # Convert caption to speech
                        with st.spinner("Converting to speech..."):
                            speech_file = text_to_speech(caption)
                            if speech_file:
                                autoplay_audio(speech_file)
                
                time.sleep(0.1)  # Small delay to reduce CPU usage
                
        else:  # Upload Image option
            uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                
                st.image(frame, channels="BGR", use_container_width=True)
                
                with st.spinner("Generating caption..."):
                    caption = get_image_caption(frame)
                    st.session_state.last_caption = caption
                    st.session_state.last_frame = frame
                
                # Convert caption to speech
                with st.spinner("Converting to speech..."):
                    speech_file = text_to_speech(caption)
                    if speech_file:
                        autoplay_audio(speech_file)
    
    with col2:
        st.subheader("Generated Caption")
        
        # Display the last generated caption if available
        if 'last_caption' in st.session_state:
            st.write(st.session_state.last_caption)
            
            # Options for the caption
            st.button("Read Again", on_click=lambda: autoplay_audio(text_to_speech(st.session_state.last_caption)))
            
            # Save caption as audio file
            if st.button("Save Audio"):
                audio_file = text_to_speech(st.session_state.last_caption)
                with open(audio_file, "rb") as file:
                    btn = st.download_button(
                        label="Download Audio File",
                        data=file,
                        file_name="caption_audio.mp3",
                        mime="audio/mp3"
                    )
            
            # Save image with caption
            if st.button("Save Image with Caption"):
                if 'last_frame' in st.session_state:
                    # Add caption to image
                    img_with_caption = st.session_state.last_frame.copy()
                    # Convert to PIL for easier text manipulation
                    pil_img = Image.fromarray(cv2.cvtColor(img_with_caption, cv2.COLOR_BGR2RGB))
                    # TODO: Add text to image using PIL
                    
                    # Provide download button
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="JPEG")
                    btn = st.download_button(
                        label="Download Image",
                        data=buffered.getvalue(),
                        file_name="captioned_image.jpg",
                        mime="image/jpeg"
                    )
        else:
            st.write("No caption generated yet. Take a snapshot or upload an image.")
        
        # Display additional information about the scene
        st.subheader("Image Analysis")
        if 'last_frame' in st.session_state:
            # Placeholder for future image analysis features
            st.write("Additional scene information will appear here.")

if __name__ == "__main__":
    main()