import numpy as np
from keras._tf_keras.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score

model = load_model('models/rec_model.h5')

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

print("Model Accuracy:", accuracy_score(y_test, predictions))
print("Model ROC AUC:", roc_auc_score(y_test, predictions))