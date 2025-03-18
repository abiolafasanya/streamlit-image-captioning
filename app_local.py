import streamlit as st
import base64
import tempfile
import numpy as np
from PIL import Image
import io
import os
from gtts import gTTS
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup page - MUST come before any other Streamlit commands
st.set_page_config(
    page_title="Image Captioning for the Visually Impaired",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API key from environment or let user enter it
api_key = os.getenv("OPENAI_API_KEY", "")

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

def text_to_speech(text):
    """
    Convert text to speech using Google's TTS
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

def get_image_caption_from_openai(image, api_key):
    """
    Get image caption using OpenAI's Vision API
    """
    if not api_key:
        return "Error: OpenAI API key is missing"
    
    try:
        # Convert PIL image to bytes for the API request
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a detailed description of this image for a visually impaired person. Focus on what's visually important and keep it to 2-3 sentences maximum."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            caption = result["choices"][0]["message"]["content"]
            return caption
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return f"Error getting caption: API returned status code {response.status_code}"

    except Exception as e:
        st.error(f"Error in OpenAI caption generation: {str(e)}")
        return f"Error generating caption: {str(e)}"

def main():
    st.title("Vision Assistant with OpenAI")
    st.subheader("Image Captioning for the Visually Impaired")
    
    # API Key Input in sidebar
    st.sidebar.title("API Settings")
    
    # Use stored API key or let user enter it
    api_key_input = st.sidebar.text_input("OpenAI API Key", 
                                         value=api_key if api_key else "",
                                         type="password",
                                         help="Enter your OpenAI API key here")
    
    # Update the API key for use in the app
    current_api_key = api_key_input if api_key_input else api_key
    
    # Voice settings
    st.sidebar.title("Voice Settings")
    voice_speed = st.sidebar.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1, 
                                   help="Adjust the reading speed (future feature)")
    
    # Main area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Upload Image option
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Open image with PIL
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Check if we have an API key
            if not current_api_key:
                st.error("Please enter your OpenAI API key in the sidebar to get image descriptions.")
            else:
                with st.spinner("Generating caption with OpenAI..."):
                    caption = get_image_caption_from_openai(image, current_api_key)
                    st.session_state.last_caption = caption
                
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
            if st.button("Read Again"):
                speech_file = text_to_speech(st.session_state.last_caption)
                if speech_file:
                    autoplay_audio(speech_file)
            
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
        else:
            st.write("No caption generated yet. Upload an image.")

    # Add information on what's happening
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses OpenAI's Vision API to generate detailed descriptions of images.
    The descriptions are then converted to speech for visually impaired users.
    
    To use this app:
    1. Enter your OpenAI API key in the sidebar
    2. Upload an image
    3. The app will generate a description and read it aloud
    """)

if __name__ == "__main__":
    main()