import json
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from main import predict
import uvicorn
import numpy as np

app = FastAPI()


def fieldLengthNotMatch():
    raise HTTPException(
        status_code=421, detail="'field' must have 8*8 length.")


def fieldElementNotMatch():
    raise HTTPException(
        status_code=421, detail="Each element in 'field' must be [0-2] number.")


def ColorNotMatch():
    raise HTTPException(
        status_code=421, detail="'color' must be [1-2] number.")


def convertToMLInput(field, color):
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


class Board(BaseModel):
    field: List[List[int]]
    color: int


@app.post("/")
def read_root(board: Board):
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
    player, audience = convertToMLInput(board.field, board.color)
    res, tries = predict(player, audience)
    if res:
        return {"status": "success", "n": tries, "response": json.dumps(res)}
    else:
        return {"status": "NG"}


@app.post("/text")
def read_test(board: Board):
    res = read_root(board)
    ans = None
    if(res["status"] == "NG"):
        ans = "NG"
    else:
        ans = res["response"]
    return ans


uvicorn.run(app, host="0.0.0.0", port=8000)
