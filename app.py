import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for better styling with background images
st.markdown("""
    <style>
    /* Background styling */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.95));
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        margin: 1rem 0 2rem 0;
    }
    
    .logo-container img {
        max-width: 150px;
        height: auto;
    }
    
    /* Prediction box styling */
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f1f8f4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .healthy {
        background-color: #c8e6c9;
        border-left: 5px solid #4caf50;
    }
    
    .disease {
        background-color: #ffccbc;
        border-left: 5px solid #ff5722;
    }
    
    /* Decorative cashew images in corners */
    .cashew-corner {
        position: fixed;
        opacity: 0.15;
        z-index: -1;
        pointer-events: none;
    }
    
    .cashew-top-left {
        top: 100px;
        left: 20px;
        width: 200px;
        transform: rotate(-15deg);
    }
    
    .cashew-bottom-right {
        bottom: 20px;
        right: 20px;
        width: 200px;
        transform: rotate(15deg);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f8f4 0%, #ffffff 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Disease classes
DISEASE_CLASSES = {
    0: "Anthracnose",
    1: "Gummosis",
    2: "Healthy",
    3: "Leaf Miner",
    4: "Not Cashew",
    5: "Parasitic Plant",
    6: "Physiological Disorder",
    7: "Powdery Mildew",
    8: "Red Rust",
    9: "Termite"
}

# Disease information
DISEASE_INFO = {
    "Anthracnose": {
        "description": "A fungal disease causing dark lesions on leaves, stems, and fruits.",
        "symptoms": "Dark, sunken lesions with pink spore masses; leaf spots; fruit rot",
        "treatment": "Apply copper-based fungicides; remove infected plant parts; improve air circulation"
    },
    "Gummosis": {
        "description": "Excessive gum exudation from bark, often caused by stress or infection.",
        "symptoms": "Gum oozing from trunk or branches; bark discoloration; cankers",
        "treatment": "Improve drainage; avoid mechanical injury; apply appropriate fungicides if fungal"
    },
    "Healthy": {
        "description": "The plant appears healthy with no visible disease symptoms.",
        "symptoms": "No visible disease symptoms",
        "treatment": "Continue regular maintenance and monitoring"
    },
    "Leaf Miner": {
        "description": "Insect larvae that tunnel through leaves creating distinctive trails.",
        "symptoms": "Serpentine trails or blotches on leaves; reduced photosynthesis",
        "treatment": "Remove affected leaves; use neem oil or appropriate insecticides; encourage natural predators"
    },
    "Not Cashew": {
        "description": "The uploaded image does not appear to be a cashew plant.",
        "symptoms": "N/A",
        "treatment": "Please upload an image of a cashew plant for accurate diagnosis"
    },
    "Parasitic Plant": {
        "description": "Parasitic plants that attach to and extract nutrients from the host plant.",
        "symptoms": "Visible parasitic plant growth; yellowing leaves; stunted growth",
        "treatment": "Manually remove parasitic plants; prevent seed dispersal; monitor regularly"
    },
    "Physiological Disorder": {
        "description": "Non-infectious problems caused by environmental or nutritional factors.",
        "symptoms": "Varied symptoms including discoloration, wilting, or abnormal growth patterns",
        "treatment": "Identify and correct environmental issues; adjust fertilization; improve growing conditions"
    },
    "Powdery Mildew": {
        "description": "Fungal disease characterized by white powdery growth on leaves.",
        "symptoms": "White powdery coating on leaves; leaf curling; stunted growth",
        "treatment": "Apply sulfur or potassium bicarbonate; improve air circulation; remove infected parts"
    },
    "Red Rust": {
        "description": "Fungal disease causing rust-colored pustules on plant tissues.",
        "symptoms": "Orange or rust-colored spots on leaves; premature leaf drop",
        "treatment": "Apply appropriate fungicides; remove infected leaves; ensure good air circulation"
    },
    "Termite": {
        "description": "Insect damage from termites attacking the plant.",
        "symptoms": "Hollowed stems or roots; wilting; presence of termite galleries",
        "treatment": "Apply termiticides; remove severely damaged parts; use termite barriers"
    }
}

@st.cache_resource
def load_model(model_path):
    """Load the trained Keras model"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(model, image):
    """Make prediction on the preprocessed image"""
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img, verbose=0)
    
    # Get class with highest probability
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Get all probabilities for display
    all_probabilities = {DISEASE_CLASSES[i]: predictions[0][i] * 100 
                        for i in range(len(DISEASE_CLASSES))}
    
    return predicted_class, confidence, all_probabilities

def main():
    # Load images for decoration and logo
    try:
        logo_img = Image.open("icon_complete.png")
        cashew_img1 = None
        cashew_img2 = None
    except Exception as e:
        logo_img = None
        cashew_img1 = None
        cashew_img2 = None
    
    # Display logo at the top center
    if logo_img:
        col_left, col_center, col_right = st.columns([1, 1, 1])
        with col_left:
            st.image(logo_img, width=500, use_container_width=True)

    # Header
    st.markdown('<h1 class="main-header">🌿 Cashew Disease Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
        <p style='text-align: center; font-size: 1.2rem; color: #555;'>
        Upload an image of a cashew plant to detect diseases and get treatment recommendations
        </p>
    """, unsafe_allow_html=True)
    
    # Add decorative cashew images with low opacity
    if cashew_img1 and cashew_img2:
        st.markdown("<br>", unsafe_allow_html=True)
        deco_col1, deco_col2, deco_col3 = st.columns([1, 10, 1])
        with deco_col1:
            st.markdown("""
                <div style='opacity: 0.2; filter: blur(1px);'>
            """, unsafe_allow_html=True)
            st.image(cashew_img1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with deco_col3:
            st.markdown("""
                <div style='opacity: 0.2; filter: blur(1px);'>
            """, unsafe_allow_html=True)
            st.image(cashew_img2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Fixed model path - place your keras_model.h5 in the same directory
        model_path = "keras_model.h5"
        
        st.markdown("---")
        
        # Information
        st.subheader("📋 About")
        st.info("""
        This application is developed by the **UENR Team** funded by **ATPS Grant**. 
        
        This platform uses deep learning to identify plant diseases in cashew plants.
        
        **Supported Diseases:**
        - Anthracnose
        - Gummosis
        - Leaf Miner
        - Parasitic Plant
        - Physiological Disorder
        - Powdery Mildew
        - Red Rust
        - Termite
        
        **Plus:** Healthy plant detection and non-cashew plant identification
        """)
        
        st.subheader("💡 Tips")
        st.markdown("""
        - Use clear, well-lit images
        - Focus on the affected area
        - Ensure the plant is in focus
        - Image should show disease symptoms clearly
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Plant Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the plant showing any disease symptoms"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        if uploaded_file is not None:
            # Load model
            with st.spinner("Loading model..."):
                model = load_model(model_path)
            
            if model is not None:
                # Make prediction
                with st.spinner("Analyzing image..."):
                    try:
                        predicted_class, confidence, all_probabilities = predict_disease(model, image)
                        disease_name = DISEASE_CLASSES[predicted_class]
                        
                        # Display prediction
                        if disease_name == "Healthy":
                            st.markdown(f"""
                                <div class="prediction-box healthy">
                                    <h2>✅ {disease_name}</h2>
                                    <h3>Confidence: {confidence:.2f}%</h3>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="prediction-box disease">
                                    <h2>⚠️ {disease_name}</h2>
                                    <h3>Confidence: {confidence:.2f}%</h3>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Disease information
                        st.markdown("---")
                        st.subheader("📊 Detailed Information")
                        
                        info = DISEASE_INFO[disease_name]
                        
                        with st.expander("🔬 Description", expanded=True):
                            st.write(info['description'])
                        
                        with st.expander("🩺 Symptoms"):
                            st.write(info['symptoms'])
                        
                        with st.expander("💊 Treatment"):
                            st.write(info['treatment'])
                        
                        # Show probability distribution
                        st.markdown("---")
                        st.subheader("📈 Probability Distribution")
                        
                        # Sort probabilities
                        sorted_probs = sorted(all_probabilities.items(), 
                                            key=lambda x: x[1], reverse=True)
                        
                        for disease, prob in sorted_probs[:5]:  # Show top 5
                            st.write(f"**{disease}:** {prob:.2f}%")
                            st.progress(prob / 100)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            else:
                st.error("❌ Model file 'keras_model.h5' not found. Please place your model file in the same directory as this script.")
        else:
            st.info("👆 Please upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #777; padding: 2rem;'>
            <p style='font-size: 1.1rem;'>🌱 <strong>Cashew Disease Detection System</strong></p>
            <p style='margin: 0.5rem 0;'>Developed by UENR Team | Funded by ATPS Grant</p>
            <p style='margin: 0.5rem 0;'>Powered by Deep Learning & AI</p>
            <p style='font-size: 0.9rem; margin-top: 1rem;'>
                <strong>Note:</strong> This is a diagnostic tool. For severe infections, 
                please consult with agricultural experts or plant pathologists.
            </p>
            <p style='font-size: 0.85rem; margin-top: 1rem; color: #555;'>
                📱 For our Mobile App access, contact: 
                <a href="mailto:patrick.mensah@uenr.edu.gh" style="color: #2e7d32;">patrick.mensah@uenr.edu.gh</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()