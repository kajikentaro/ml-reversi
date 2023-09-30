# %%
import sys
import os
from utils import get_teaching_data, stack_player_audience
import tensorflow as tf
from keras.layers import Input, Flatten, Dense, Reshape, Softmax
import datetime
from predict import metrics

# カレントディレクトリをモジュールの検索パスに追加
# 親ディレクトリをworking directoryに変更
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.chdir(os.path.join(current_dir, "../"))


# %%
def load_training_data_to_global():
    global teaching_x, teaching_y
    player_list, audience_list, ans_list = get_teaching_data("2007-2015-short.txt")
    teaching_x = stack_player_audience(player_list, audience_list)
    teaching_y = ans_list


teaching_x = None
teaching_y = None
load_training_data_to_global()

# %%
# npy形式で保存するとtextから読むよりも高速
# np.save("teaching_x_short", teaching_x)
# np.save("teaching_y_short", teaching_y)
# teaching_x = np.load("teaching_x_short.npy")
# teaching_y = np.load("teaching_y_short.npy")


# %%
# モデル定義
def make_model():
    model: tf.keras.Model = tf.keras.Sequential(
        [
            Input(
                shape=(
                    2,
                    8,
                    8,
                )
            ),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(96, activation="relu"),
            Dense(64),
            Reshape((8, 8)),
            Softmax(),
        ]
    )
    model.summary()
    return model


def training(dir_name):
    n = int(teaching_x.shape[0] / 7)
    print(n)

    x_train = teaching_x[:n]
    y_train = teaching_y[:n]

    x_test = teaching_x[n:]
    y_test = teaching_y[n:]

    # 訓練開始
    model = make_model()
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=[metrics])
    model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=1048,
        validation_data=(x_test, y_test),
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir="tensorboard/" + dir_name)],
    )
    model.save("saved_model/" + dir_name)


# %%
# 訓練開始
nowTime = datetime.datetime.now(
    datetime.timezone(datetime.timedelta(hours=9))
).strftime("%Y%m%d%H%M%S")
training(nowTime)

# %%
# 訓練後、保存したモデルを読み込み
model = tf.keras.models.load_model(
    "saved_model/" + nowTime, custom_objects={"metrics": metrics}
)
model.summary()

# %%
# 訓練後、モデルをts-liteに変換
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/" + nowTime)
tflite_model = converter.convert()
with open("default_model.tslite", "wb") as f:
    f.write(tflite_model)
