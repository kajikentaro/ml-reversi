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
model: tf.keras.Model = tf.keras.Sequential([
    Input(shape=(2, 8, 8,)),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dense(128),
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


# %%
# 学習
# tf.config.run_functions_eagerly(True)
model.compile(optimizer='Adam', loss='mse', metrics=[metrics])
model.fit(x_train, y_train, epochs=10000, batch_size=64,
          validation_data=(x_train, y_train))
model.save("1208")

# %%
n = 100
res = model.predict(x_train[n:n+1])
ans = y_train[n:n+1]
print(np.unravel_index(np.argmax(res), res.shape))
print(np.unravel_index(np.argmax(ans), ans.shape))

# %%
