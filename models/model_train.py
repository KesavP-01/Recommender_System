import pandas as pd
import numpy as np
import model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical


def preprocessing(filepath):
    df = pd.read_csv(filepath)
    encoder = LabelEncoder()

    df['user_id'] = encoder.fit_transform(df['user_id'])
    df['item_id'] = encoder.fit_transform(df['item_id'])

    X = df[['user_id', 'item_id']].values
    y = df['like'].values

    y = to_categorical(y)

    return X, y

def main():
    X, y = preprocessing('data/interactions.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

    r_model = model.model_build(input=X_train.shape[1])
    r_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    r_model.save('models/rec_model.h5')


if __name__ == '__main__':
    main()