import json
from fastapi import FastAPI, HTTPException
from typing import List
from predict_tslite import convert_to_np_input, predict
from pydantic import BaseModel
import uvicorn

app = FastAPI()


def fieldLengthNotMatch():
    raise HTTPException(status_code=421, detail="'field' must have 8*8 length.")


def fieldElementNotMatch():
    raise HTTPException(
        status_code=421, detail="Each element in 'field' must be [0-2] number."
    )


def ColorNotMatch():
    raise HTTPException(status_code=421, detail="'color' must be [1-2] number.")


class Board(BaseModel):
    field: List[List[int]]
    color: int


def execute(board: Board):
    if board.color != 1 and board.color != 2:
        ColorNotMatch()
    if len(board.field) != 8:
        fieldLengthNotMatch()
    for line in board.field:
        if len(line) != 8:
            fieldLengthNotMatch()
        for i in line:
            if i < 0 or 3 <= i:
                fieldElementNotMatch()
    player, audience = convert_to_np_input(board.field, board.color)
    res, tries = predict(player, audience)
    return res, tries


@app.post("/")
def read_root(board: Board):
    res, tries = execute(board)
    if res:
        return {"status": "success", "n": tries, "response": json.dumps(res)}
    else:
        return {"status": "NG"}


@app.post("/text")
def read_text(board: Board):
    res, tries = execute(board)
    print(str(res[0]) + " " + str(res[1]))
    if res:
        return str(res[0]) + " " + str(res[1])
    else:
        return "NG"


uvicorn.run(app, host="0.0.0.0", port=8000)
