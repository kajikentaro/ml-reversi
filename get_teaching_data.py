from typing import AnyStr
import numpy as np
from numpy.lib.function_base import extract


def get_teaching_data():
    each_board_list: list[Board] = []
    with open("2007-2015-short.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            pre_wthor = Board()
            pre_wthor.init_from_wthor(line[0:2])
            each_board_list.append(pre_wthor)
            for i in range(0, len(line)-1, 2):
                if(i == 0):
                    continue
                wthor = Board()
                wthor.init_from_wthor(line[i:i+2], pre_wthor)
                each_board_list.append(wthor)
                pre_wthor = wthor
        f.close()
    player_list = []
    audience_list = []
    ans_list = []
    for board in each_board_list:
        player_list.extend(board.get_upsized_player())
        audience_list.extend(board.get_upsized_audience())
        ans_list.extend(board.get_upsized_ans())
    return np.array(player_list), np.array(audience_list), np.array(ans_list)


def stack_player_audience(player_list, audience_list):
    return np.stack([player_list, audience_list], 1)


class Board:
    # Input
    player = None
    audience = None
    # Output
    ans = None
    # 学習には使わないが次データ作成に使用
    post_player = None
    post_audience = None

    def init_from_raw(self, player, audience):
        self.player = player
        self.audience = audience
        self.post_player = self.player.copy()
        self.post_audience = self.audience.copy()

    def init_from_wthor(self, new_stone, now_board=None):
        self.ans = np.zeros((8, 8))
        if now_board:
            self.player = now_board.post_audience
            self.audience = now_board.post_player
        else:
            self.player = np.array([[0]*8]*8)
            self.audience = np.array([[0]*8]*8)
            self.audience[3][3] = 1
            self.audience[4][4] = 1
            self.player[3][4] = 1
            self.player[4][3] = 1
        self.post_player = self.player.copy()
        self.post_audience = self.audience.copy()
        col = ord(new_stone[0]) - ord('a')
        row = ord(new_stone[1]) - ord('1')
        self.ans[row][col] = 1
        self.mix = [self.player, self.audience]
        if self.reverse(row, col) == False:
            self.player, self.audience = self.audience, self.player
            self.post_audience, self.post_player = self.post_player, self.post_audience
            self.reverse(row, col)

    def print(self):
        print(np.array(self.player))
        print(np.array(self.audience))

    def upsize(self, input):
        res_array = []
        res_array.append(np.rot90(input, k=0))
        #res_array.append(np.rot90(input, k=1))
        res_array.append(np.rot90(input, k=2))
        #res_array.append(np.rot90(input, k=3))
        #res_array.append(np.flipud(np.rot90(input, k=0)))
        res_array.append(np.flipud(np.rot90(input, k=1)))
        #res_array.append(np.flipud(np.rot90(input, k=2)))
        res_array.append(np.flipud(np.rot90(input, k=3)))
        #res_array.append(np.fliplr(np.rot90(input, k=0)))
        #res_array.append(np.fliplr(np.rot90(input, k=1)))
        #res_array.append(np.fliplr(np.rot90(input, k=2)))
        #res_array.append(np.fliplr(np.rot90(input, k=3)))
        return res_array

    def get_upsized_player(self):
        return self.upsize(self.player)

    def get_upsized_audience(self):
        return self.upsize(self.audience)

    def get_upsized_ans(self):
        return self.upsize(self.ans)

    def reverse(self, row, col):
        done_reverse = False
        if self.post_player[row][col] + self.post_audience[row][col] >= 1:
            return False
        dirs = [(0, 1), (1, 0), (-1, 0), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in dirs:
            may_reverse = []
            do_reverse = False
            row_t = row
            col_t = col
            while True:
                row_t += dir[0]
                col_t += dir[1]
                if(row_t >= 8 or col_t >= 8):
                    break
                if(row_t < 0 or col_t < 0):
                    break
                if(self.post_player[row_t][col_t] == 1):
                    do_reverse = True
                    break
                if(self.post_audience[row_t][col_t] == 1):
                    may_reverse.append((row_t, col_t))
                    continue
                break
            if(do_reverse and len(may_reverse) >= 1):
                self.post_player[row][col] = 1
                done_reverse = True
                for p in may_reverse:
                    self.post_player[p[0]][p[1]] = 1
                    self.post_audience[p[0]][p[1]] = 0
        return done_reverse
