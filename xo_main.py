from board import *
import xo_bot as xob


N = 3


def new_game(par, bot1, bot2):
    b = Board(N)
    # b.make_move(0, 1)
    # b.make_move(4, -1)
    # b.make_move(2, 1)
    # b.make_move(1,-1)
    # b.make_move(7, 1)
    # b.make_move(3, -1)
    b.show()

    if par == 'PvsB':
        playerX = 'player'
        playerO = 'bot'
    elif par == 'BvsB':
        playerX = 'bot'
        playerO = 'bot'
    else:
        playerX = 'player'
        playerO = 'player'

    player = [None, 'X', 'O']
    move, movepiece, movenum = 0, None, 0
    while not b.is_gameover(move, movepiece) and movenum != N**2:
        if movenum % 2 == 0:
            movepiece = 1
        else:
            movepiece = -1

        if movepiece == -1 and playerO == 'bot':
            # ans = str(input("It's my turn! Which bot do u prefer? [mm/ab/mc] "))
            move = xob.bot_move(b, movepiece, bot2)
            print(move)
        elif movepiece == 1 and playerX == 'bot':
            move = xob.bot_move(b, movepiece, bot1)
            print(move)
        else:
            move = int(input("It's '" + str(player[movepiece]) + "' move: "))

        while not b.is_move_OK(move):
            move = int(input(colored('Invalid move!', 'red') + ' Please try again: '))
        b.make_move(move, movepiece)
        b.show()
        movenum += 1
    else:
        if b.gameover:
            print("'" + str(player[b.winner]) + "' wins!")
    return b.winner


def print_res_of_bot_fights(res_of_fights, N):
    print("* * *\n"
          "Summarising results:")
    for elem in res_of_fights:
        print("Algorythm type '" + str(elem) + "' wins "
              + str(res_of_fights[elem]) + " times from " + str(N) + " games.")


def main():
    ans = str(input("Wanna to play? [y/n] "))
    if ans == 'y':
        ans = str(input("With me? [y/n] "))
        if ans == 'y':
            ans = str(input("Opponent is you? [y/n] "))
            if ans == 'y':
                print("Hooray! Let's play!")
                new_game('PvsB', None, None)
            else:
                print("You will see how I get him down!")
                ans = int(input("How many times? (input even-num) "))
                bot1, bot2 = 'mc', 'ab'
                res_of_fights = dict()
                res_of_fights[bot1] = 0
                res_of_fights[bot2] = 0
                for i in range(ans):
                    if i % 2 == 0:
                        res = new_game('BvsB', bot1, bot2)
                        if res == 1:
                            res_of_fights[bot1] += 1
                        elif res == -1:
                            res_of_fights[bot2] += 1
                    else:
                        res = new_game('BvsB', bot2, bot1)
                        if res == 1:
                            res_of_fights[bot2] += 1
                        elif res == -1:
                            res_of_fights[bot1] += 1
                print_res_of_bot_fights(res_of_fights, ans)

        else:
            print("Allright then. Play with yourself")
            new_game('PvsP', None, None)
    else:
        print(':(')


main()
