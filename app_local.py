import streamlit as st

# Setup page - MUST come before any other Streamlit commands
st.set_page_config(
    page_title="Image Captioning for the Visually Impaired",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from gtts import gTTS

st.title("Vision Assistant")
st.subheader("Image Captioning for the Visually Impaired")

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

def get_sample_caption(image):
    """
    This is a placeholder function that returns a sample caption
    In a real app, you would call a captioning API or model here
    """
    # Get image dimensions for a slightly more customized caption
    width, height = image.size
    
    # Create a sample caption with the dimensions
    caption = f"An image with dimensions {width}x{height} pixels."
    
    # You can add more logic here if needed
    if width > height:
        caption += " The image appears to be in landscape orientation."
    elif height > width:
        caption += " The image appears to be in portrait orientation."
    else:
        caption += " The image appears to be square."
    
    return caption

# Main area
col1, col2 = st.columns([2, 3])

with col1:
    # Upload Image option
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Open image with PIL
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        with st.spinner("Generating caption..."):
            caption = get_sample_caption(image)
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
This is a minimal version of the image captioning app.
Currently, it generates a placeholder caption based on the image dimensions.
To enable real captioning, you would need to integrate with a captioning API
or incorporate a local model.
""")

# Add a voice setting option in the sidebar for future enhancement
st.sidebar.title("Settings")
voice_speed = st.sidebar.slider("Speech Rate (Future Feature)", 0.5, 2.0, 1.0, 0.1)