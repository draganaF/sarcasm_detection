from keras import Sequential
from keras.layers import Normalization, Embedding, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow import keras

from src.train_cnn_model import split_dataset


def create_model(x_train_size):
    # inputs = Input(shape=(size,), dtype='int32')
    model = Sequential()
    model.add(Normalization())
    model.add(Embedding(input_dim=x_train_size, output_dim=256, input_length=13))
    model.add(Conv1D(128, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(256, 5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # sgd = optimizers.SGD(learning_rate=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.build(input_shape=(None, 13))

    model.save("../model")

    return model

if __name__ == '__main__':
    X_train, X_test, X_val, Y_train, Y_test, Y_val = split_dataset()
    create_model(len(X_train))