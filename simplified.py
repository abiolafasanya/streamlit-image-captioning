import streamlit as st
import base64
import io
import tempfile
from PIL import Image
import requests
from io import BytesIO

# Setup page
st.set_page_config(
    page_title="Image Captioning for the Visually Impaired",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_image_caption(image):
    """
    Generate caption using OpenAI Vision API or HuggingFace API
    """
    # Placeholder for actual API call - replace with your implementation
    # This is just a dummy function to simulate image captioning
    return "A sample image caption would appear here."

def text_to_speech(text):
    """
    Convert text to speech using gTTS
    """
    try:
        from gtts import gTTS
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
    st.subheader("Image Captioning for the Visually Impaired")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # Voice settings
    voice_speed = st.sidebar.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)
    
    # Image source selection
    image_source = st.sidebar.radio("Select Image Source", ["Upload Image"])
    
    # Main area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Only implement upload image option for simplicity
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Open image with PIL
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            with st.spinner("Generating caption..."):
                caption = get_image_caption(image)
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

if __name__ == "__main__":
    main()