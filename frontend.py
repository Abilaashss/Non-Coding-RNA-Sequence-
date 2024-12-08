import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="RNA Sequence Classifier",
    page_icon="🧬",
    layout="wide"
)

def prepare_sequence(sequence):
    """Prepare a single RNA sequence for model inference"""
    # Convert to 2-mers
    K = 2
    seq_str = str(sequence).strip()
    kmers = [seq_str[i:i+K] for i in range(len(seq_str)) if len(seq_str[i:i+K]) == K]
    
    # Tokenize
    tokenizer = Tokenizer(num_words=30000)
    tokenizer.fit_on_texts([kmers])
    sequences = tokenizer.texts_to_sequences([kmers])
    
    # Pad sequence
    padded_seq = pad_sequences(sequences, maxlen=224, padding="post")
    
    return padded_seq

def get_prediction(model, sequence):
    """Get model prediction for a sequence"""
    # Class labels
    class_labels = [
        '5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box',
        'miRNA', 'Intron_gpI', 'Intron_gpII', 'HACA-box',
        'riboswitch', 'IRES', 'leader', 'scaRNA'
    ]
    
    # Prepare sequence and get prediction
    processed_seq = prepare_sequence(sequence)
    prediction = model.predict(processed_seq)
    
    # Get top 3 predictions
    top_3_indices = prediction[0].argsort()[-3:][::-1]
    top_3_predictions = [
        (class_labels[i], float(prediction[0][i]) * 100)
        for i in top_3_indices
    ]
    
    return top_3_predictions

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
        }
        .prediction-box {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
            background-color: #00796b;
            border: 2px solid #004d40;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🧬 RNA Sequence Classifier")
    st.markdown("""
        This application classifies RNA sequences into different RNA types.
        Enter your RNA sequence below to get started.
    """)
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model('best_model.h5')
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Input section
    st.subheader("Input RNA Sequence")
    sequence = st.text_area(
        "Enter the RNA sequence:",
        height=150,
        help="Enter your RNA sequence. All characters will be accepted."
    )
    
    # Process sequence
    if st.button("Classify Sequence"):
        if not sequence:
            st.warning("Please enter a sequence.")
        else:
            # Clean sequence - just remove whitespace and newlines
            sequence = sequence.upper().replace('\n', '').replace(' ', '')
            
            # Get prediction
            with st.spinner("Analyzing sequence..."):
                try:
                    predictions = get_prediction(model, sequence)
                    
                    # Display results
                    st.subheader("Classification Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Top Predictions")
                        for i, (rna_type, confidence) in enumerate(predictions, 1):
                            st.markdown(
                                f"""
                                <div class="prediction-box">
                                    <h4>#{i}: {rna_type}</h4>
                                    <p>Confidence: {confidence:.2f}%</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        # Create a bar chart
                        chart_data = {
                            "RNA Type": [p[0] for p in predictions],
                            "Confidence (%)": [p[1] for p in predictions]
                        }
                        st.bar_chart(chart_data, x="RNA Type", y="Confidence (%)")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Add information section
    with st.expander("About this Classifier"):
        st.markdown("""
            This RNA sequence classifier uses a deep learning model to identify different types of RNA sequences.
            The model can classify sequences into 13 different RNA types:
            
            - 5S rRNA
            - 5.8S rRNA
            - tRNA
            - Ribozyme
            - CD-box
            - miRNA
            - Intron gpI
            - Intron gpII
            - HACA-box
            - Riboswitch
            - IRES
            - Leader
            - scaRNA
            
            The classifier now accepts all sequence characters, including non-standard nucleotides and special characters.
        """)

if __name__ == "__main__":
    main()