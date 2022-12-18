import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import *
import numpy as np
import pickle
import io

def load_file(file_path):
    data = open(file_path).read()

    return data


def tokenizer_data(data):
    tokenizer = Tokenizer()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    #save tokenizer
    with open('main/PM/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return corpus, tokenizer, total_words


def gen_train_data(corpus, tokenizer, total_words):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    x_train, labels = input_sequences[:,:-1],input_sequences[:,-1]
    y_train = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    return x_train, y_train, max_sequence_len


def build_model(total_words, max_sequence_len, x_train, y_train):
    model = Sequential()
    model.add(Embedding(total_words, 200, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=50, verbose=1)

    return model, history


def gen_poem(user_text, max_sequence_len, len_gen_words):
    # loading tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    #load model
    model = tf.keras.models.load_model("model.h5")

    #generation
    for i in range(len_gen_words):
        token_list = tokenizer.texts_to_sequences([user_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        user_text += " " + output_word

    return user_text

# if __name__ =="__main__":
#     #load data
#     data = load_file("SHAKESPEARE.txt")
#     #tokenizing
#     corpus, tokenizer, total_words = tokenizer_data(data)
#     #get train data
#     x_train, y_train, max_sequence_len = gen_train_data(corpus, tokenizer, total_words)
#     #build model
#     model, history = build_model(total_words, max_sequence_len, x_train, y_train)
#     #save model
#     model.save("model.h5")
#
#     #generate poem
#     user_text = "when I see your face"
#     len_gen_words = 50
#     user_text = gen_poem(user_text, max_sequence_len, len_gen_words)
#
#     print(user_text)


