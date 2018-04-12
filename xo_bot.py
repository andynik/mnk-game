import random
from copy import deepcopy
import time


# files for time measurements
fout_mm = open("mm_time.txt", 'w')
fout_ab = open("ab_time.txt", 'w')
fout_mc = open("mc_time.txt", 'w')


''' MAIN PART '''

def save_player(player):
    f = open('cur_player.txt', 'w')
    f.write(str(player))
    f.close()


def get_player():
    f = open('cur_player.txt', 'r')
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
            start = time.time()
            move = mm_bot(position, player)
            end = time.time()
            fout_mm.write(str(end - start) + '\n')
        elif t == 'ab':
            start = time.time()
            move = ab_bot(position, player)
            end = time.time()
            fout_ab.write(str(end - start) + '\n')
            # move = ab_bot(position, player)
        elif t == 'mc':
            start = time.time()
            move = mc_bot(position, player)
            end = time.time()
            fout_mc.write(str(end - start) + '\n')
            # move = mc_bot(position, player)
        elif t == 'r':
            move = r_bot(position)
    return move


'''RANDOM BOT'''

def r_bot(position):
    empty_squares = position.get_empty_squares()
    move = empty_squares[random.randrange(len(empty_squares))]
    return move


''' MINIMAX ALGORYTM '''

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

def alphabeta(position, lastmove, player, alpha, beta, depth):         # alpha = get_player(), beta = enemy
    if position.is_gameover(lastmove, get_enemy(player)) or depth == 0:
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
        val = alphabeta(clone, move, get_enemy(player), alpha, beta, depth-1)
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
    if len(position.get_empty_squares()) == position.size ** 2: # best 1st move
        return position.size ** 2 // 2 + 1
    players = [None, 'O', 'X']
    for move in position.get_empty_squares():
        clone = deepcopy(position)
        clone.make_move(move, player)
        val = alphabeta(clone, move, get_enemy(player), -2, 2, 4)
        print("move", move, "causes to", players[val], "wins!")
        if val > a:
            a = val
            choices = [move]
        elif val == a:
            choices.append(move)
    return random.choice(choices)


''' MONTE CARLO SELECTION '''

NTRIALS = 3000
SCORE_CURRENT = 1.0
SCORE_OTHER = 2.0
DEP = 4

def mc_trial(position, player):
    empty_squares = position.get_empty_squares()
    move = empty_squares[random.randrange(len(empty_squares))]
    depth = DEP
    while not position.is_gameover(move, get_enemy(player)) and depth != 0:
        empty_squares = position.get_empty_squares()
        move = empty_squares[random.randrange(len(empty_squares))]
        position.make_move(move, player)
        player = get_enemy(player)
        depth -= 1


def mc_update_scores(scores, position, player):
    winner = position.winner
    if winner == 0:
        return
    coef = 1
    if player != winner:
        coef = -1
    for row in range(position.size):
        for col in range(position.size):
            if position.board[row][col] == player:
                scores[row][col] += coef * SCORE_CURRENT
            elif position.board[row][col] != 0:
                scores[row][col] -= coef * SCORE_OTHER


def get_best_move(position, scores):
    best_square = position.get_empty_squares()[0]
    r, s = best_square // position.size, best_square % position.size
    best_score = scores[r][s]
    for square in position.get_empty_squares():
        r, s = square // position.size, square % position.size
        if scores[r][s] > best_score:
            best_square = square
            best_score = scores[r][s]
    out = []
    for square in position.get_empty_squares():
        r, s = square // position.size, square % position.size
        if scores[r][s] == best_score:
            out.append(square)
    return out[random.randrange(len(out))]


def mc_move(position, player, trials):
    scores = [[0] * position.size for i in range(position.size)]
    num = 0
    while num < trials:
        clone = deepcopy(position)
        mc_trial(clone, player)
        mc_update_scores(scores, clone, player)
        num += 1
    print(scores)
    return get_best_move(position, scores)


def mc_bot(position, player):
    clone = deepcopy(position)
    return mc_move(clone, player, NTRIALS)
