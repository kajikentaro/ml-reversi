# %%
from utils import Board
import tflite_runtime.interpreter as tflite
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(current_dir, "../"))


def predict(player, audience):
    predict_x = np.float32(np.array([[player, audience]]))

    # モデル初期化
    interpreter = tflite.Interpreter(model_path="default_model.tslite")
    interpreter.allocate_tensors()

    # 入力値を設定
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], predict_x)

    # 呼び出し
    interpreter.invoke()

    # 出力値を取得
    output_details = interpreter.get_output_details()
    predict_y = interpreter.get_tensor(output_details[0]['index'])[0]

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


# %%