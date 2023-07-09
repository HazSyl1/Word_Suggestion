import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional



def initialise():
    print("Setting up the Model")

    tokenizer = Tokenizer()
    with open('Book 6 - The Half Blood Prince.txt',encoding="utf-8") as f:
        data = f.read()

    # Convert to lower case and save as a list
    corpus = data.lower().split("\n")


    new_corpus = []
    for line in corpus:
        my_new_string = line.translate({ord(i): None for i in "!?,"})
        new_corpus.append(my_new_string)

    tokenizer.fit_on_texts(new_corpus)

    print("Done!")
    return seed_text(tokenizer)

def seed_text(tokenizer):
    seed=input("Enter Text:")
    return suggest(seed,tokenizer)



def suggest(seed_text,tokenizer):

    number_of_sug =1
    # imporing the model
    new_model = tf.keras.models.load_model('Word_suggestion.h5')
    for _ in range(number_of_sug):


        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences([token_list], maxlen=15 - 1, padding='pre')

        predicted = new_model.predict(token_list, verbose=0)
        output = []
        for i in range(3):

            predicted_word = np.argmax(predicted, axis=-1).item()

            output_word = tokenizer.index_word[predicted_word]

            output.append(output_word)

            predicted[0][predicted_word]=0

        seed_text += " " + str(output)
        return seed_text

Suggestions=initialise()

print(Suggestions)
