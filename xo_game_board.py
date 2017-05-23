from termcolor import colored


C = 3


class Board():
    def __init__(self, size):
        self.size = size
        self.board = [[0] * self.size for i in range(self.size)]
        self.gameover = False
        self.winner = 0

    def __str__(self):
        s = ''
        i = 1
        for line in self.board:
            for square in line:
                s += '{0:^3}'.format(square)
                if i % self.size == 0:
                    s += '\n'
                i += 1
        return s

    def is_gameover(self, move, piece):
        self.is_win(move, piece)
        if len(self.get_empty_squares()) == 0:
            self.gameover = True
        return self.gameover

    def get_empty_squares(self):
        empty_squares = []
        for i in range(self.size):
            for j in range(self.size):
                if not self.board[i][j]:
                    empty_squares += [i * self.size + j]
        return empty_squares

    def is_move_OK(self, pos):
        return 0 <= pos < self.size ** 2 and not self.board[pos // self.size][pos % self.size]

    def make_move(self, pos, piece):
        self.board[pos // self.size][pos % self.size] = piece

    def is_win(self, move, piece):
        global C

        flag = False
        r, s = move // self.size, move % self.size
        matrix = self.board

        h = matrix[r][max(s - (C-1), 0):s] + [matrix[r][s]] + matrix[r][s + 1:min(s + (C-1), self.size - 1) + 1]
        v = [matrix[a][s] for a in range(max(r - (C-1), 0), r)] + [matrix[r][s]] +\
            [matrix[a][s] for a in range(r + 1, min(r + (C-1), self.size - 1) + 1)]
        d1 = [matrix[r - a][s - a] for a in range(min(r, s, (C-1)) + 1 -1, 1-1, -1)] +\
             [matrix[r][s]] + [matrix[r + a][s + a] for a in range(1, min(self.size - r - 1, self.size - s - 1, (C-1)) + 1)]
        d2 = [matrix[r - a][s + a] for a in range(min(r, self.size - s - 1, (C-1)) + 1 -1, 1-1, -1)] +\
             [matrix[r][s]] + [matrix[r + a][s - a] for a in range(1, min(self.size - r - 1, s, (C-1)) + 1)]
        lines = [h, v, d1, d2]                              # all possible loss-positions
        # print('lines:', lines)

        player = ['.', 'X', 'O']            # converting -1, 1, 0 to O, X, .
        if piece == 1 or piece == -1:
            five_pieces = player[piece] * C
        else:
            five_pieces = '#' * C

        for line in lines:
            s = ''
            for elem in line:
                s += player[elem]
            if five_pieces in s:
                flag = True
                break

        if flag:
            self.gameover = True
            self.winner = piece
        return flag

    def show(self):
        s = ''
        i = 0
        # piece = ['.', 'X', 'O']
        colors = ['white', 'red', 'blue']
        for line in self.board:
            for square in line:
                piece = [str(i), 'X', 'O']
                s += colored(str('{0:^3}'.format(piece[square])), colors[square])
                if (i+1) % self.size == 0:
                    s += '\n'
                i += 1
        print(s)
