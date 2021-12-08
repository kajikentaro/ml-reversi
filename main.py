# %%
# ライブラリインポート
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

x_test = teaching_x[:n]
y_test = teaching_y[:n]

# %%
# モデル定義
model: tf.keras.Model = tf.keras.Sequential([
    Input(shape=(2, 8, 8,)),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(64),
    Reshape((8, 8)),
    Softmax()
])
model.summary()

# %%
# 学習
model.compile(optimizer='Adam', loss='mse')
model.fit(x_train, y_train, epochs=10000, batch_size=64)
# model.predict(x_train[0:1])

# %%
n = 100
res = model.predict(x_train[n:n+1])
ans = y_train[n:n+1]
print(np.unravel_index(np.argmax(res), res.shape))
print(np.unravel_index(np.argmax(ans), ans.shape))
