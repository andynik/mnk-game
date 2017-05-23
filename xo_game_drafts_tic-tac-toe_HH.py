# Start small to go big

N = 3

def printField(f):
    print(' *    1   2   3', end='')
    rnum = 0
    for elem in f:
        if rnum % 3 == 0:
            print('\n', rnum // N, end='  ')
        print('{0:3}'.format(elem), end=' ')
        rnum += 1
    print()


def ifVictory(f, turn):
    flag = False

    lines = []
    h1, h2, h3 = f[0:3], f[3:6], f[6:]
    v1, v2, v3 = [[f[i], f[i+N], f[i+2*N]] for i in range(N)]
    d1 = [f[0], f[4], f[8]]
    d2 = [f[2], f[4], f[6]]
    lines = [h1, h2, h3, v1, v2, v3, d1, d2]

    if turn == 'X':
        num = 1
    else:
        num = 0

    if [num] * 3 in lines:
        flag = True

    return flag



print("'X' - 1st\n"
      "'O' - 2nd")
field = [0.5 for j in range(N**2)]
printField(field)
movenum = 1
turn = 'X'
while not ifVictory(field, turn) and movenum < 9:
    if movenum % 2 == 1:
        turn = 'X'
    else:
        turn = 'O'
    print("It's '" + turn + "' move. Input coords:", end=' ')
    r, s = map(int, input().split())
    field[r*N+s] = movenum % 2
    movenum += 1
    printField(field)

winflag = ifVictory(field, turn)
if winflag:
    print("'" + turn + "' wins!")
else:
    print("It's a draw.")
