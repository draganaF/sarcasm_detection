from flatbuffers.builder import np
from tensorflow import keras

from src.train_cnn_model import split_dataset


def evaluate_model(model, X_test, Y_test):
    scores = model.evaluate(X_test, Y_test, verbose=0)

    print("Accuracy: %.2f%%" % (scores[1] * 100))

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_test)):

        result = model.predict(X_test[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=0)[0]

        if np.around(result) == np.around(Y_test[x]):
            if np.around(Y_test[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.around(Y_test[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1
    print("Sarcasm accuracy\t: ", round(pos_correct / pos_cnt * 100, 3), "%")
    print("Non-sarcasm accuracy\t: ", round(neg_correct / neg_cnt * 100, 3), "%")

if __name__ == '__main__':
    X_train, X_test, X_val, Y_train, Y_test, Y_val = split_dataset()

    model = keras.models.load_model("../model")
    model.load_weights("../data/weights.h5")

    evaluate_model(model, X_test, Y_test)