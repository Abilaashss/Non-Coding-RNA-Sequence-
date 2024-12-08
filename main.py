import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization
from keras.layers import LSTM, LayerNormalization, GRU, Bidirectional
from keras.optimizers import Adam
import os


# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def prepare_data(features, labels):
    # Create one-hot encoded labels
    label_mapping = {
        '5S_rRNA': [1,0,0,0,0,0,0,0,0,0,0,0,0],
        '5_8S_rRNA': [0,1,0,0,0,0,0,0,0,0,0,0,0],
        'tRNA': [0,0,1,0,0,0,0,0,0,0,0,0,0],
        'ribozyme': [0,0,0,1,0,0,0,0,0,0,0,0,0],
        'CD-box': [0,0,0,0,1,0,0,0,0,0,0,0,0],
        'miRNA': [0,0,0,0,0,1,0,0,0,0,0,0,0],
        'Intron_gpI': [0,0,0,0,0,0,1,0,0,0,0,0,0],
        'Intron_gpII': [0,0,0,0,0,0,0,1,0,0,0,0,0],
        'HACA-box': [0,0,0,0,0,0,0,0,1,0,0,0,0],
        'riboswitch': [0,0,0,0,0,0,0,0,0,1,0,0,0],
        'IRES': [0,0,0,0,0,0,0,0,0,0,1,0,0],
        'leader': [0,0,0,0,0,0,0,0,0,0,0,1,0],
        'scaRNA': [0,0,0,0,0,0,0,0,0,0,0,0,1]
    }
    
    encoded_labels = np.array([label_mapping[label] for label in labels])
    
    # Process sequences using 2-mer encoding
    K = 2
    str_array = []
    for seq in features:
        seq_str = str(seq).strip('[]\'')
        kmers = [seq_str[i:i+K] for i in range(len(seq_str)) if len(seq_str[i:i+K]) == K]
        str_array.append(kmers)
    
    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=30000)
    tokenizer.fit_on_texts(str_array)
    sequences = tokenizer.texts_to_sequences(str_array)
    sequences = pad_sequences(sequences, maxlen=224, padding="post")
    
    return sequences, encoded_labels

def create_model():
    sequence_input = Input(shape=(224,))
    embedding_layer = Embedding(30000, 16, input_length=224)
    embedded_sequences = embedding_layer(sequence_input)
    
    # Batch normalization
    embedded_sequences = BatchNormalization(epsilon=1e-6)(embedded_sequences)
    
    # Stem section
    stem = Conv1D(filters=96, kernel_size=8, padding="same", activation="gelu")(embedded_sequences)
    lstm = Bidirectional(LSTM(16, return_sequences=True))(embedded_sequences)
    lstm = layers.Activation("gelu")(lstm)
    stem = concatenate([stem, lstm], axis=2)
    stem = BatchNormalization(epsilon=1e-6)(stem)
    stem = Dropout(0.5)(stem)

    
    # Fully Connected Module
    mlp = Dense(256, activation="gelu")(stem)
    mlp = BatchNormalization(epsilon=1e-6)(mlp)
    mlp = Dropout(0.2)(mlp)
    mlp = Dense(128, activation="gelu")(mlp)
    mlp = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(mlp, stem)
    stem = stem + mlp

    # Simplified Multi-Window Convolution
    cnn1 = Conv1D(filters=64, kernel_size=16, activation="gelu", padding='same')(stem)
    cnn2 = Conv1D(filters=32, kernel_size=10, activation="gelu", padding='same')(stem)
    merge = concatenate([cnn1, cnn2], axis=2)
    merge = BatchNormalization(epsilon=1e-6)(merge)
    merge = Dropout(0.2)(merge)
    
    # Additional convolutional layer with attention
    cnn3 = Conv1D(filters=128, kernel_size=16, activation="gelu", padding='same')(merge)
    cnn3 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(merge, cnn3)
    merge = merge + cnn3
    merge = BatchNormalization(epsilon=1e-6)(merge)
    
    # Final layers
    flat = Flatten()(merge)
    flat = Dropout(0.5)(flat)
    dense = Dense(200, activation="sigmoid")(flat)
    dense = Dropout(0.5)(dense)
    output = Dense(13, activation="softmax")(dense)
    
    model = Model(sequence_input, output)
    return model

def train_model():
    # Load data
    dataset = pd.read_csv('./dataset.csv', header=None)
    features = dataset[1][:]
    labels = dataset[2][:]
    
    # Split into train and validation sets (80-20 split)
    split_idx = int(len(features) * 0.8)
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]
    
    # Prepare data
    X_train, y_train = prepare_data(train_features, train_labels)
    X_val, y_val = prepare_data(val_features, val_labels)
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Create model checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        batch_size=64,  # Reduced batch size for CPU
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback],
        verbose=1
    )
    
    return history, model

if __name__ == "__main__":
    # Train the model
    history, model = train_model()
    
    # Print final results
    print("\nTraining completed!")
    print("Best validation accuracy:", max(history.history['val_accuracy']))
    print("Best model saved as 'best_model.h5'")



