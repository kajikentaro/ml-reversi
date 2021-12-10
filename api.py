from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from main import predict
import numpy as np

app = FastAPI()


class Board(BaseModel):
    field: List[List[int]]
    color: int


def fieldLengthNotMatch():
    raise HTTPException(
        status_code=422, detail="'field' must have 8*8 length.")


def fieldElementNotMatch():
    raise HTTPException(
        status_code=422, detail="Each element in 'field' must be [0-2] number.")


def ColorNotMatch():
    raise HTTPException(
        status_code=422, detail="'color' must be [1-2] number.")


def convertToMLInput(field, color):
    player = np.zeros(8, 8)
    audience = np.zeros(8, 8)
    for i in range(8):
        for j in range(8):
            if field[i][j] == 0:
                continue
            elif field[i][j] == color:
                player[i][j] = 1
                continue
            else:
                audience[i][j] = 1
    return np.array([player, audience])


class User(BaseModel):
    name: str


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
    input_x = convertToMLInput(board.field)
    return {"response": predict(input_x).tolist()}
