# MLP
import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import metrics
from keras.layers import Dense, Dropout, Activation
from keras import models
from keras.optimizers import Adam


def mean_relative_error(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    mre = 0.0
    for i in range(len(y_pred)):
        mre = mre + abs((y_pred[i] - y_test[i]) / (y_test[i]))
    mre = mre / len(y_pred)
    return mre

def buildModel():
    model = models.Sequential()
    l1 = Dense(512)
    l2 = Activation('relu')
    l3 = Dropout(rate=0)
    l4 = Dense(128)
    l5 = Activation('relu')
    l6 = Dropout(rate=0)
    l7 = Dense(32)
    l8 = Activation('relu')
    l9 = Dropout(rate=0)
    l10 = Dense(1)
    layers = [
            l1, l2, l3,
            l4, l5, l6,
            l7, l8, l9,
            l10
    ]
    for i in range(len(layers)):
        model.add(layers[i])
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])
    return model

def Regression(data_x_df, data_y_df):
    print("shape_x", data_x_df.shape)
    print("shape_y", data_y_df.shape)
    # Normalization
    min_max_scaler_X = MinMaxScaler()
    min_max_scaler_X.fit(data_x_df)
    x_trans1 = min_max_scaler_X.transform(data_x_df)    # normalized input
    min_max_scaler_y = MinMaxScaler()
    min_max_scaler_y.fit(data_y_df)
    y_trans1 = min_max_scaler_y.transform(data_y_df)    # normalized output
    # Train the model
    MAEs = []           # list of mean absolute errors
    RMSEs = []           # list of mean relative errors
    for i in range(1):
        kf = KFold(n_splits=10, shuffle=True, random_state=i)
        for train_index, test_index in kf.split(x_trans1):
            X_train = x_trans1[train_index, :]
            y_train = y_trans1[train_index, :]
            X_test = x_trans1[test_index, :]
            y_test = y_trans1[test_index, :]
            model_mlp = buildModel()
            model_mlp.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)
            result = model_mlp.predict(X_test)
            y_test = np.reshape(y_test, (-1, 1))
            result = result.reshape(-1, 1)
            mae = metrics.mean_absolute_error(y_test, result)
            rmse = metrics.mean_squared_error(y_test, result, squared=False)
            MAEs.append(mae)
            RMSEs.append(rmse)
    return sum(MAEs)/len(MAEs), max(MAEs), min(MAEs), sum(RMSEs)/len(RMSEs), max(RMSEs), min(RMSEs)
