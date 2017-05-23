from copy import deepcopy
import random
from board import *
import neural_network as nn


N = 3


''' MAIN PART '''

def save_player(player):
    f = open('bot_player.txt', 'w')
    f.write(str(player))
    f.close()


def get_player():
    f = open('bot_player.txt', 'r')
    player = int(f.readline())
    return player


def get_enemy(player):
    if player == -1:
        return 1
    return -1


def bot_move(position, player, t):
    save_player(player)
    if t == None:
        players = [None, 'X', 'O']
        ans = str(input("I'm '" + str(players[player]) + "'. Which Bot do u prefer? [mm/ab/mc/r] "))
        if ans == 'mm':
            move = mm_bot(position, player)
        elif ans == 'ab':
            move = ab_bot(position, player)
        elif ans == 'mc':
            move = mc_bot(position, player)
        elif ans == 'r':
            move = r_bot(position)
        print("My move is...")
    else:
        if t == 'mm':
            move = mm_bot(position, player)
        elif t == 'ab':
            move = ab_bot(position, player)
        elif t == 'mc':
            move = mc_bot(position, player)
        elif t == 'r':
            move = r_bot(position)
    return move


'''RANDOM BOT'''

def r_bot(position):
    empty_squares = position.get_empty_squares()
    move = empty_squares[random.randrange(len(empty_squares))]
    return move


''' MINIMAX ALGORYTM '''

# minimax bot pack for 'O'-player
# def minimax(position, depth):
#     moves = position.get_empty_squares()
#     best_move = moves[0]
#     best_score = float('inf')
#     for move in moves:
#         clone = deepcopy(position)
#         clone.make_move(move, -1)
#         score = max_play(clone, depth-1, move, -1)
#         if score < best_score:
#             best_move = move
#             best_score = score
#     return best_move
#
#
# def min_play(position, depth, lastmove, lastpiece):
#   if position.is_gameover(lastmove, lastpiece) or depth == 0:
#       return evaluate(position, depth)
#   moves = position.get_empty_squares()
#   best_score = float('inf')
#   for move in moves:
#     clone = deepcopy(position)
#     clone.make_move(move, -1)
#     score = max_play(clone, depth-1, move, -1)
#     if score < best_score:
#       best_move = move
#       best_score = score
#   return best_score
#
#
# def max_play(position, depth, lastmove, lastpiece):
#   if position.is_gameover(lastmove, lastpiece) or depth == 0:
#       return evaluate(position, depth)
#   moves = position.get_empty_squares()
#   best_score = float('-inf')
#   for move in moves:
#     clone = deepcopy(position)
#     clone.make_move(move, 1)
#     score = min_play(clone, depth-1, move, 1)
#     if score > best_score:
#       best_move = move
#       best_score = score
#   return best_score
#
#
# def evaluate(pos, dep):
#     if pos.winner == 1:
#         return 10 * (dep+1)
#     elif pos.winner == -1:
#         return -10 * (dep+1)
#     return 0

def minimax(position, depth, player):
    moves = position.get_empty_squares()
    best_move = moves[0]
    best_score = float('-inf')
    for move in moves:
        clone = deepcopy(position)
        clone.make_move(move, player)
        score = min_play(clone, depth-1, move, player)
        if score > best_score:
            best_move = move
            best_score = score
    return best_move


def min_play(position, depth, lastmove, player):
  if position.is_gameover(lastmove, player) or depth == 0:
      return evaluate(position, depth)
  moves = position.get_empty_squares()
  player = get_enemy(player)
  best_score = float('inf')
  for move in moves:
    clone = deepcopy(position)
    clone.make_move(move, player)
    score = max_play(clone, depth-1, move, player)
    if score < best_score:
      best_move = move
      best_score = score
  return best_score


def max_play(position, depth, lastmove, player):
  if position.is_gameover(lastmove, player) or depth == 0:
      return evaluate(position, depth)
  moves = position.get_empty_squares()
  player = get_enemy(player)
  best_score = float('-inf')
  for move in moves:
    clone = deepcopy(position)
    clone.make_move(move, player)
    score = min_play(clone, depth-1, move, player)
    if score > best_score:
      best_move = move
      best_score = score
  return best_score


def evaluate(pos, dep):
    cur_player = get_player()
    if pos.winner == cur_player:
        return 10 * (dep+1)
    elif pos.winner == cur_player * (-1):
        return -10 * (dep+1)
    return 0


def mm_bot(position, player):
    return minimax(position, 6, player)


''' ALPHA BETA PRUNING '''

def alphabeta(position, lastmove, player, alpha, beta):         # alpha = 'X', beta = 'O'
    if position.is_gameover(lastmove, get_enemy(player)):
        # return position.winner * (-1)
        cur_player = get_player()
        if cur_player == -1:
            return position.winner * (-1)
        elif cur_player == 1:
            return position.winner
        return 0
    for move in position.get_empty_squares():
        clone = deepcopy(position)
        clone.make_move(move, player)
        val = alphabeta(clone, move, get_enemy(player), alpha, beta)
        if player == get_player():
            if val > alpha:
                alpha = val
            if alpha >= beta:
                return beta
        else:
            if val < beta:
                beta = val
            if beta <= alpha:
                return alpha
    if player == get_player():
        return alpha
    else:
        return beta


def ab_bot(position, player):
    # player = -1
    a = -2
    choices = []
    if len(position.get_empty_squares()) == 9:
        return 4
    players = [None, 'O', 'X']
    for move in position.get_empty_squares():
        clone = deepcopy(position)
        clone.make_move(move, player)
        val = alphabeta(clone, move, get_enemy(player), -2, 2)
        print("move", move, "causes to", players[val], "wins!")
        if val > a:
            a = val
            choices = [move]
        elif val == a:
            choices.append(move)
    return random.choice(choices)


''' MONTE CARLO SELECTION '''

NTRIALS = 1000
SCORE_CURRENT = 1.0
SCORE_OTHER = 2.0

def mc_trial(position, player):
    empty_squares = position.get_empty_squares()
    move = empty_squares[random.randrange(len(empty_squares))]
    while not position.is_gameover(move, get_enemy(player)):
        empty_squares = position.get_empty_squares()
        move = empty_squares[random.randrange(len(empty_squares))]
        position.make_move(move, player)
        player = get_enemy(player)


def mc_update_scores(scores, position, player):
    winner = position.winner
    if winner == 0:
        return
    coef = 1
    if player != winner:
        coef = -1
    for row in range(N):
        for col in range(N):
            if position.board[row][col] == player:
                scores[row][col] += coef * SCORE_CURRENT
            elif position.board[row][col] != 0:
                scores[row][col] -= coef * SCORE_OTHER


def get_best_move(position, scores):
    best_square = position.get_empty_squares()[0]
    r, s = best_square // N, best_square % N
    best_score = scores[r][s]
    for square in position.get_empty_squares():
        r, s = square // N, square % N
        if scores[r][s] > best_score:
            best_square = square
            best_score = scores[r][s]
    out = []
    for square in position.get_empty_squares():
        r, s = square // N, square % N
        if scores[r][s] == best_score:
            out.append(square)
    return out[random.randrange(len(out))]


def mc_move(position, player, trials):
    scores = [[0] * N for i in range(N)]
    num = 0
    while num < trials:
        clone = deepcopy(position)
        mc_trial(clone, player)
        mc_update_scores(scores, clone, player)
        num += 1
    print(scores)
    return get_best_move(position, scores)


# mc_bot_pack v 2.0
# NTRIALS = 1000   # Number of trials to run
# MCMATCH = 2.0  # Score for squares played by the machine player
# MCOTHER = 1.0  # Score for squares played by the other player
#
# EMPTY = 1
# PLAYERX = 2
# PLAYERO = 3
# DRAW = 4
#
# def mc_trial(position, player):
#     empty_squares = position.get_empty_squares()
#     if len(empty_squares):
#         move = empty_squares[random.randrange(len(empty_squares))]
#         while not position.is_gameover(move, get_enemy(player)):
#             empty_squares = position.get_empty_squares()
#             move = empty_squares[random.randrange(len(empty_squares))]
#             position.make_move(move, player)
#             player = get_enemy(player)
#
#
# def adding_scores(position, scores, player, r, s, win):
#     if win:
#         if (position.board[r][s] == 0):
#             scores[r][s] += 0.0
#         elif (position.board[r][s] == player):
#             scores[r][s] += (MCMATCH)
#         else:
#             scores[r][s] += (-MCOTHER)
#     else:
#         if (position.board[r][s] == 0):
#             scores[r][s] += 0.0
#         elif (position.board[r][s] == player):
#             scores[r][s] += (-MCMATCH)
#         else:
#             scores[r][s] += (MCOTHER)
#
# def mc_update_scores(scores, position, player):
#     winner = position.winner
#     if player == winner:
#         for r in range(N):
#             for s in range(N):
#                 adding_scores(position, scores, player, r, s, True)
#     elif winner == 0:
#         pass
#     else:
#         for r in range(N):
#             for s in range(N):
#                 adding_scores(position, scores, player, r, s, False)
#
#
# def max_score_in_2d_list(scores, given_list):
#     max_value = scores[given_list[0] // N][given_list[0] % N]
#     for empty in given_list:
#         if (scores[empty // N][empty % N] >= max_value):
#             max_value = scores[empty // N][empty % N]
#     return max_value
#
#
# def get_best_move(position, scores):
#     if (position.get_empty_squares() == []):
#         return
#     empty_squares = position.get_empty_squares()
#     max_list = []
#     max_value = max_score_in_2d_list(scores, empty_squares)
#     for empty in empty_squares:
#         if (scores[empty // N][empty % N] == max_value):
#             max_list.append(empty)
#     empty = random.choice(max_list)
#     return empty
#
#
# def mc_move(position, player, trials):
#     scores = []
#     scores = [[0] * N for i in range(N)]
#
#     for _dummy_trails in range(trials):
#         clone = deepcopy(position)
#         mc_trial(clone, player)
#         mc_update_scores(scores, clone, player)
#     # print scores,get_best_move(board,scores)
#     print(scores)
#     return get_best_move(position, scores)

def mc_bot(position, player):
    clone = deepcopy(position)
    return mc_move(clone, player, NTRIALS)


''' HEURISTIC Func '''

def h_bot(position):
    pass


''' ITERATIVE DEEPING '''

def id_bot(position):
    pass


# def nn_bot(pos):
#     neural_network = nn.NeuralNetwork(N ** 2 - 1, N ** 2, 1)
#     nn.train_neural_network(neural_network)
#     print(int(neural_network.feed_forward(pos)[0] * N ** 2))

# Simple way of teaching it is too long
# Literature of how to do it correctly:
# https://pdfs.semanticscholar.org/6251/1fb1c8e0d3bbdc445fb097ac4fc9b1e21a2f.pdf (2011)
# https://www.researchgate.net/publication/312325842_Move_prediction_in_Gomoku_using_deep_learning (2016)

