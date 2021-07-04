import json
from keras.preprocessing.text import Tokenizer
from string import punctuation, digits

from flatbuffers.builder import np
from keras_preprocessing.sequence import pad_sequences
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow import keras


def load_data():
    texts, sentiments = [], []
    with open('../data/dataset2.json') as file:
        for line in file:
            json_text = json.loads(line)
            texts.append(json_text["headline"])
            sentiments.append(json_text["is_sarcastic"])

    return texts, sentiments


def preprocess(text):
    hl_cleansed = []
    for hl in text:
        #     Remove punctuations
        clean = hl.translate(str.maketrans('', '', punctuation))
        #     Remove digits/numbers
        clean = clean.translate(str.maketrans('', '', digits))
        hl_cleansed.append(clean)

    stop = ['a', 'an', 'the', 'and', 'for', 'of']
    ps = PorterStemmer()
    preprocessed_text = ''
    preprocessed_text_list = []
    for returned in hl_cleansed:
        for word in returned.split(" "):
            if word not in stop:
                word = ps.stem(word)
                preprocessed_text += word + ' '
        preprocessed_text_list.append(preprocessed_text.strip())
        preprocessed_text = ''

    return preprocessed_text_list


def tokenize(text):
    text = preprocess(text)
    tokens = []
    for hl in text:
        tokens.append(hl.split(" "))

    return tokens


def preparing_data(text):
    max_features = 2000
    max_token = len(max(text))
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    X = pad_sequences(sequences, maxlen=max_token)

    return X

def split_dataset():
    texts, is_sarcastic = load_data()
    texts = tokenize(texts)
    texts = preparing_data(texts)

    Y = np.vstack(is_sarcastic)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, Y, test_size=0.1, random_state=42)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2223, random_state=42)

    return  X_train, X_test, X_val, Y_train, Y_test, Y_val

def fit_model(model, X_train, Y_train, X_val, Y_val):
    # class_weights = {0: 1,
    #                  1: 1.3}
    y_ints = [y[0] for y in Y_train]
    class_weights = compute_class_weight("balanced", np.unique(y_ints), y_ints)
    class_weight_dict = dict(enumerate(class_weights))

    # Fitting the data onto model
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=32, batch_size=256, verbose=2, class_weight=class_weight_dict)
    model.summary()
    model.save_weights("../data/weights.h5")

if __name__ == '__main__':
    X_train, X_test, X_val, Y_train, Y_test, Y_val = split_dataset()

    model = keras.models.load_model("../model")

    fit_model(model, X_train, Y_train, X_val, Y_val)