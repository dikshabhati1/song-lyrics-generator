## Import libs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import string
import numpy as np
from tensorflow.keras.models import  Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,LSTM ,Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense, Bidirectional 
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pickle
import time
import pandas as pd

import base64



## Read file
def main():
    st.markdown("<h1 style='text-align: center; color: red;'>Song Lyrics Generator</h1>", unsafe_allow_html=True)

    
    # Tokenization
    df = pd.read_csv("/content/final_song_df.csv")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Lyric'].astype(str).str.lower())

    
    # create model
    model = Sequential()
    model.add(Embedding(8556, 40, input_length=656-1))
    model.add(Bidirectional(LSTM(250)))
    model.add(Dropout(0.1))
    model.add(Dense(8556, activation='softmax'))
   


    ## Load model using checkpoint
    checkpoint_path = "lstm-model.ckpt"
    model.load_weights(checkpoint_path)


    ## predict func
    def complete_this_song(seed_text, next_words):
        for _ in range(next_words):
            # tokenizer = Tokenizer()
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=656-1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
        
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text

    st.markdown("<h4 >Start your song lyrics:</h4>", unsafe_allow_html=True)
    seed_text = st.text_area('', height=200, max_chars=800)
    st.markdown("<h4 >Max text length (in characters)</h4>", unsafe_allow_html=True)
    next_words = slider = st.slider('', 5, 800)
    button = st.button('Generate')


    if button:
        st.markdown(f"""<h3 style='text-align: center; color: white;background :rgba(0, 0, 255, 0.9);'> {complete_this_song(seed_text, next_words)}</h3>""", unsafe_allow_html=True)
 

if __name__ == "__main__":
    main()