# %%
# ライブラリインポート
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_math_ops import arg_max
from get_teaching_data import get_teaching_data, stack_player_audience, Board
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Softmax, Dropout
import numpy as np

teaching_x = None
teaching_y = None


# %%
# モデル定義
def make_model():
    model: tf.keras.Model = tf.keras.Sequential([
        Input(shape=(2, 8, 8,)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(96, activation='relu'),
        Dense(64),
        Reshape((8, 8)),
        Softmax()
    ])
    model.summary()
    return model


# 評価関数定義
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


def fetch_training_data():
    global teaching_x, teaching_y
    # 教師データ取得
    player_list, audience_list, ans_list = get_teaching_data()
    teaching_x = stack_player_audience(player_list, audience_list)
    teaching_y = ans_list


def training(dir_name):
    n = int(teaching_x.shape[0] / 7)
    print(n)

    x_train = teaching_x[:n]
    y_train = teaching_y[:n]

    x_test = teaching_x[n:]
    y_test = teaching_y[n:]

    # 訓練開始
    model = make_model()
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=[metrics])
    model.fit(x_train, y_train, epochs=200, batch_size=1048,
              validation_data=(x_test, y_test),
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir="tensorboard/" + dir_name)])
    model.save("saved_model/" + dir_name)


def predict(player, audience):
    predict_x = np.array([[player, audience]])
    model = tf.keras.models.load_model(
        "default_model", custom_objects={'metrics': metrics})
    predict_y = model.predict(predict_x)[0]
    board = Board()
    board.init_from_raw(player, audience)
    res = None
    for i in range(64):
        max_arg = np.unravel_index(
            np.argmax(predict_y), predict_y.shape)
        max_arg = np.array(max_arg).tolist()
        if board.reverse(max_arg[0], max_arg[1]):
            res = max_arg
            break
        predict_y[max_arg[0]][max_arg[1]] = 0
    return res, i


# %%
# fetch_training_data()
# teaching_x = np.load("teaching_x.npy")
# teaching_y = np.load("teaching_y.npy")

# %%
# training("1213-11")


# %%
# model = tf.keras.models.load_model(
# "saved_model/1210-2", custom_objects={'metrics': metrics})
# model.summary()


# %%
# np.save("teaching_x_short", teaching_x)
# np.save("teaching_y_short", teaching_y)
