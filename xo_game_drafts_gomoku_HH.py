N = 9

def printField(f):
    s = '*  ' + ' '.join([str("%02d" % i) for i in range(N)])
    print(s)
    rnum = 0
    for elem in f:
        print(str("%02d" % rnum) + ' ' + ' '.join(map(str, elem)))
        rnum += 1

def IsWin(f, symb, r, s):
    flag = False

    print(N)
    h = f[r][max(s-4, 0):s] + [f[r][s]] + f[r][s+1:min(s+4, N-1)+1]
    v = [f[a][s] for a in range(max(r-4, 0), r)] + [f[r][s]] + [f[a][s] for a in range(r+1, min(r+4, N-1)+1)]
    d1 = [f[r-a][s-a] for a in range(1, min(r, s, 4)+1)] + [f[r][s]] + [f[r+a][s+a] for a in range(1, min(N-r-1, N-s-1, 4)+1)]
    d2 = [f[r-a][s+a] for a in range(1, min(r, N-s-1, 4)+1)] + [f[r][s]] + [f[r+a][s-a] for a in range(1, min(N-r-1, s, 4)+1)]

    lines = [h, v, d1, d2]
    print(lines)

    fiveinrow = [symb] * 5
    for line in lines:
        if ''.join(map(str, fiveinrow)) in ''.join(map(str, line)):
            flag = True

    return flag


field = [[" ."] * N for j in range(N)]

print("'x' - first, 'o' - second")
movenum = 0
winflag = False
moveto = ' x'
printField(field)
r, s = 0, 0
while not IsWin(field, moveto, r, s) and movenum < N ** 2:
    if movenum % 2 == 0:
        moveto = ' x'
    else:
        moveto = ' o'
    print("It's", moveto, "move. Input coords:")
    r, s = map(int, input().split())
    field[r][s] = moveto
    movenum += 1
    printField(field)

winflag = IsWin(field, moveto, r, s)
if winflag:
    print(moveto, "wins!")
else:
    print("It's a draw.")
