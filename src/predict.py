# %%
from utils import Board
import tensorflow as tf
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(current_dir, "../"))


def predict(player, audience):
    predict_x = np.array([[player, audience]])
    model = tf.keras.models.load_model(
        "default_model", custom_objects={"metrics": metrics}
    )
    predict_y = model.predict(predict_x)[0]
    board = Board()
    board.init_from_raw(player, audience)
    res = None
    for i in range(64):
        max_arg = np.unravel_index(np.argmax(predict_y), predict_y.shape)
        max_arg = np.array(max_arg).tolist()
        if board.reverse(max_arg[0], max_arg[1]):
            res = max_arg
            break
        predict_y[max_arg[0]][max_arg[1]] = 0
    return res, i


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


def convert_to_np_input(field, color):
    player = np.zeros((8, 8))
    audience = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if field[i][j] == 0:
                continue
            elif field[i][j] == color:
                player[i][j] = 1
                continue
            else:
                audience[i][j] = 1
    return player, audience


# %%
if __name__ == "__main__":
    field = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    color = 1
    player, audience = convert_to_np_input(field, color)
    res, tries = predict(player, audience)
    print(res, tries)
