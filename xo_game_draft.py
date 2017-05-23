#Tic-Tac-Toe Minimax Program by Peter

#Tutorial found at: http://giocc.com/concise-implementation-of-minimax-through-higher-order-functions.html
from operator import itemgetter

class GameState:
    def __init__(self,board):
        self.board = board
        self.winning_combos = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    def get_winner(self):
        '''returns None if the game is still going, otherwise scores from computer's point of view (1=win, 0=tie, -1=win)'''
        if self.board.count('_') == 0:
            return 0
        for combo in self.winning_combos:
            if (self.board[combo[0]] == 'X' and self.board[combo[1]] == 'X' and self.board[combo[2]] == 'X'):
                return 1
            elif self.board[combo[0]] == 'O' and self.board[combo[1]] == 'O' and self.board[combo[2]] == 'O':
                return 0
        return None
    def get_possible_moves(self):
        '''returns all possible squares to place a character'''
        return [index for index, square in enumerate(self.board) if square == '_']
    def get_next_state(self, move, our_turn):
        '''returns the gamestate with the move filled in'''
        copy = self.board[:]
        copy[move] = 'X' if our_turn else 'O'
        return GameState(copy)


def play(game_state, our_turn):
    '''if the game is over returns (None, score), otherwise recurses to find the best move and returns it and the score.'''
    score = game_state.get_winner()
    if score != None:
        return None, score
    moves = ((move, play(game_state.get_next_state(move, our_turn), not our_turn)[1]) for move in game_state.get_possible_moves())
    return (max if our_turn else min)(moves, key=itemgetter(1))

def pretty_print(board):
    '''prints a list by 3 chars, joined by spaces'''
    print(' '.join(board[:3]))
    print(' '.join(board[3:6]))
    print(' '.join(board[6:9]))


#Interpreting and printing board
start_game_state = GameState(['_','_','_',
                              '_','O','_',
                              '_','_','_'
                            ])
pretty_print(start_game_state.board)
#Finding best possible move and score
move, score = play(start_game_state, True)
#Displaying move and outcome
if score == 0:
    word = 'TIE'
elif score == 1:
    word = 'WIN'
else:
    word = 'LOSS, who rigged the board?!?'
print('X should go at index #',move, 'Which will always result in a ' + word)
start_game_state.board[move] = 'X'
pretty_print(start_game_state.board)
