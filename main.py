# %%
# ライブラリインポート
from enum import auto
from get_teaching_data import get_teaching_data
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Reshape, Softmax
import numpy as np


# %%
# 教師データ取得
n = 100000
(teaching_x, teaching_y) = get_teaching_data()

x_train = teaching_x[:n]
y_train = teaching_y[:n]

x_test = teaching_x[n:]
y_test = teaching_y[n:]


# %%
# モデル定義
def make_model():
    model: tf.keras.Model = tf.keras.Sequential([
        Input(shape=(2, 8, 8,)),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(96),
        Activation('relu'),
        Dense(96),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dense(64),
        Reshape((8, 8)),
        Softmax()
    ])
    model.summary()
    return model


def metrics(y_true, y_pred):
    # (None, 8, 8) to (None, 64)
    reshaped_t = tf.reshape(y_true, (-1, 64))
    reshaped_p = tf.reshape(y_pred, (-1, 64))
    # (None): 64個のマスのうち、一番大きい要素の添字を格納
    argmax_t = tf.math.argmax(reshaped_t, 1)
    argmax_p = tf.math.argmax(reshaped_p, 1)
    # 添字が同じ場所をTrueにした行列へ
    is_equal = tf.math.equal(argmax_t, argmax_p)
    # booleanをfloat32へ
    is_equal_float = tf.cast(is_equal, tf.float32)
    # 平均を取る
    mean = tf.reduce_mean(is_equal_float)
    return mean


def training(dir_name):
    model = make_model()
    model.compile(optimizer='Adam', loss='mse', metrics=[metrics])
    model.fit(x_train, y_train, epochs=2000, batch_size=128,
              validation_data=(x_train, y_train),
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir="tensorboard/" + dir_name)])
    model.save("saved_model/" + dir_name)


def load_model():
    n = 1000
    model = tf.keras.models.load_model(
        "1208", custom_objects={'metrics': metrics})
    model.summary()
    res = model.predict(x_test[:n])
    print(metrics(y_test[:n], res).numpy())


# %%
training("1209")
