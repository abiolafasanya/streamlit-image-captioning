import streamlit as st

# Setup page - MUST come before any other Streamlit commands
st.set_page_config(
    page_title="Local Image Captioning for the Visually Impaired",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import the rest of the libraries
import numpy as np
import base64
import os
import io
import time
import tempfile

# Try importing OpenCV with error handling
try:
    import cv2
except ImportError:
    st.error("Failed to import OpenCV (cv2). Please make sure 'opencv-python-headless' is installed.")
    st.info("Run: pip install opencv-python-headless")
    cv2 = None

# Try importing PIL with error handling
try:
    from PIL import Image
except ImportError:
    st.error("Failed to import PIL. Please make sure 'Pillow' is installed.")
    st.info("Run: pip install Pillow")
    Image = None

# Try importing gTTS with error handling
try:
    from gtts import gTTS
except ImportError:
    st.error("Failed to import gTTS. Please make sure 'gTTS' is installed.")
    st.info("Run: pip install gTTS")
    gTTS = None

# Try importing transformers with error handling
try:
    from transformers import pipeline
except ImportError:
    st.error("Failed to import transformers. Please make sure 'transformers' is installed.")
    st.info("Run: pip install transformers torch")
    pipeline = None

# Initialize image captioning model (load once at startup)
@st.cache_resource
def load_image_captioning_model():
    if pipeline is None:
        return None
    
    try:
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        return captioner
    except Exception as e:
        st.error(f"Error loading image captioning model: {str(e)}")
        st.info("Please run: pip install transformers torch")
        return None

# Load the model
captioner = load_image_captioning_model()

def get_image_caption(image):
    """
    Generate caption using local Hugging Face model
    """
    if captioner is None:
        return "Error: Image captioning model not loaded."
    
    try:
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Generate caption
        result = captioner(pil_image)
        caption = result[0]['generated_text']
        
        # Post-process the caption to make it more descriptive
        enhanced_caption = f"The image shows {caption}"
        return enhanced_caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def text_to_speech(text):
    """
    Convert text to speech using Google's TTS
    """
    if gTTS is None:
        st.error("gTTS module not available. Cannot convert text to speech.")
        return None
    
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
    # Check if required modules are available
    if cv2 is None or Image is None or gTTS is None or pipeline is None:
        st.error("Some required modules are missing. Please install all dependencies:")
        st.code("pip install streamlit opencv-python-headless torch Pillow gTTS transformers")
        return
    
    st.title("Local Vision Assistant")
    st.subheader("Offline Image Captioning for the Visually Impaired")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # Voice settings
    voice_speed = st.sidebar.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)
    
    # Image source selection
    image_source = st.sidebar.radio("Select Image Source", ["Webcam", "Upload Image"])
    
    # Main area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        if image_source == "Webcam":
            st.write("Camera Feed")
            
            # Camera device selection
            camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0, 
                                             help="Try different numbers if your camera isn't detected (usually 0 for built-in webcam)")
            
            # Initialize webcam with the selected index
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                st.error(f"Cannot open webcam at index {camera_index}. Please check your camera settings or try a different index.")
                st.info("Common issues: Privacy settings blocking access, camera in use by another application, or incorrect camera index.")
                return
            
            # Take snapshot button
            if st.button("Take Snapshot"):
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB (Streamlit expects RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb_frame, use_container_width=True)
                    
                    with st.spinner("Generating caption..."):
                        caption = get_image_caption(frame)
                        st.session_state.last_caption = caption
                        st.session_state.last_frame = frame
                    
                    # Convert caption to speech
                    with st.spinner("Converting to speech..."):
                        speech_file = text_to_speech(caption)
                        if speech_file:
                            autoplay_audio(speech_file)
                else:
                    st.error("Failed to take snapshot. Please check your camera.")
            
            # Create a placeholder for webcam feed
            webcam_placeholder = st.empty()
            
            # Display a single frame as a preview (not continuous stream which can cause issues)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB (Streamlit expects RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(rgb_frame, caption="Camera Preview (Click 'Take Snapshot' to capture)",
                                       use_container_width=True)
            else:
                st.error("Failed to get preview from webcam")
                
            # Release the camera when this section is done
            cap.release()
                    
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
            st.write("No caption generated yet. Take a snapshot or upload an image.")

if __name__ == "__main__":
    main()