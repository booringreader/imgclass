# evaluation factors

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

def evaluate_model(model, test_data):
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test_data.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    precision = pre.result().numpy()
    recall = re.result().numpy()
    accuracy = acc.result().numpy()

    return precision, recall, accuracy
