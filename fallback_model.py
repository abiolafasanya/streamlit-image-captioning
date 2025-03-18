# For a fallback that doesn't require OpenAI API, you could implement a local model using Hugging Face Transformers

# In app.py, add this function:

def get_image_caption_local(image):
    """
    Use a local model from Hugging Face for image captioning.
    This is a fallback if OpenAI API fails.
    """
    try:
        # Convert the OpenCV image to PIL format
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Import libraries only when needed
        from transformers import pipeline
        
        # Use a pre-trained image captioning model
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        
        # Generate the caption
        caption = captioner(img)[0]['generated_text']
        
        # Enhance the basic caption to be more descriptive
        return f"Image shows {caption}. [Generated using local model]"
    except Exception as e:
        return f"Error using local captioning model: {str(e)}"


# Then in the main get_image_caption function, add this at the end:

    # If we get here, all models failed, try local model
    try:
        st.warning("All OpenAI models failed. Trying local model...")
        return get_image_caption_local(image)
    except Exception as e:
        st.error(f"Local model also failed: {str(e)}")
        return f"All image captioning methods failed. Last error: {error_msg}"