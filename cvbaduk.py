import cv2 
import numpy
import sys
import imutils
from sklearn.feature_extraction import image
from skimage import util

EMPTY = 0
WHITE = 1 << 0
BLACK = 1 << 2
DAME = WHITE | BLACK

class Stone:
    color = 0
    canPaint = False

    def __init__(self, state):
        self.color = state
        if state == EMPTY:
            self.canPaint = True

    def __str__(self):
        if self.color == EMPTY:
            return "E"
        elif self.color == WHITE:
            return "W"
        elif self.color == BLACK:
            return "B"
        elif self.color == DAME:
            return "D"
    
    def paint(self, influence):
        if not self.canPaint:
            return
        self.color |= influence.color

imgfile = "/Users/exm/go/src/github.com/millere/cvbaduk/Game01/1an-dsp.jpeg"

def toSquare(img, box, l):
    # corners = numpy.array([[c[0][0], c[0][1]] for c in box])
    #todo maybe need to order these ^?
    pts = numpy.array(box.reshape(4,2), dtype="float32")

    target = numpy.array([(0,0), (0, l), (l, l), (l, 0)], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, target)
    return cv2.warpPerspective(img, M, (l,l))

def toTiles(board):
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    tileSize = len(gray)/19
    return util.view_as_blocks(gray, block_shape=(tileSize, tileSize))

def identifyStone(stone):
    # stone should be 53x53
    # maybe a straight avg is good enough?
    avg = numpy.average(stone[10:43, 10:43])
    ident = EMPTY
    if avg > 180:
        ident = WHITE
    elif avg < 100:
        ident = BLACK
    return Stone(ident)

def paint_board(board):
    for i in range(0,19):
        for j in range(0, 19):
            if i > 0:
                board[i][j].paint(board[i-1][j])
            if j > 0:
                board[i][j].paint(board[i][j-1])
            if i < 18:
                board[i][j].paint(board[i+1][j])
            if j < 18:
                board[i][j].paint(board[i][j+1])

def board_painted(board):
    for row in board:
        for stone in row:
            if stone.color == EMPTY:
                return False
    return True

def show_automaton_state(boardimg, gamestate):
    b = boardimg.copy()
    for (j, row) in enumerate(gamestate):
        for (i, stone) in enumerate(row):
            p1x = 53 * i + 13
            p1y = 53 * j + 13
            p2x = 53 * i + 39
            p2y = 53 * j + 39
            color = None
            if stone.color == WHITE:
                color = (255,255,255)
            elif stone.color == BLACK:
                color = (0,0,0)
            elif stone.color == DAME:
                color = (0,0,255)
            if color is not None:
                cv2.rectangle(b, (p1x,p1y), (p2x,p2y), color, 3)
        
    cv2.imshow(winname, b)
    cv2.waitKey(0)

def score(game, komi):
    white = komi
    black = 0
    dame = 0
    show_automaton_state(board, game)
    while not board_painted(game):
        paint_board(game)
        show_automaton_state(board, game)
    for (i, row) in enumerate(game):
        for (j, point) in enumerate(row):
            if point.color == WHITE:
                white += 1
            elif point.color == BLACK:
                black += 1
            elif point.color == DAME:
                dame += 1

    return (black, white, dame)
    # area scoring :D


komi = 7.5
if len(sys.argv) == 3:
    komi = int(sys.argv[2])

if len(sys.argv) > 1:
    imgfile = sys.argv[1]

def donothing(img):
    pass

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

winname = "cvbaduk"

gray = cv2.bilateralFilter(gray, 10, 75, 75)
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname, 1000, 1000)
cv2.createTrackbar("Threshold", winname, 191, 255, donothing)
cnts = None
while True:
    _, thresh = cv2.threshold(
        gray, cv2.getTrackbarPos("Threshold", winname), 255,
        cv2.THRESH_BINARY_INV)
    itemp = img.copy()

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cv2.drawContours(itemp, cnts, -1, (0, 255, 0), 4)
    cv2.imshow(winname, itemp)
    k = cv2.waitKey(1) & 0xFF
    if k == 13 or k == 27: # enter or esc
        break

candidate = max(cnts, key=lambda c: cv2.arcLength(c, True))
epsilon = 0.1*cv2.arcLength(candidate, True)
approx = cv2.approxPolyDP(candidate, epsilon, True)
if len(approx) != 4:
    print "Could not find board (polygon is", len(approx), "sided)"
    sys.exit(0)
cv2.drawContours(img, [approx], -1, (255, 0, 0), 4)
board = toSquare(img, approx, 1007) # 1007 is nicely divisible by 19
cv2.imshow(winname, board)
cv2.waitKey(0)

stones = toTiles(board)

gameState = [[identifyStone(s) for s in row] for row in stones]
for row in stones:
    for stone in row:
        cv2.putText(stone, 
        identifyStone(stone).__str__(), (2, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0))
        cv2.imshow(winname, stone)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    else:
        continue
    break
            
(blackScore, whiteScore, damePoints) = score(gameState, komi)
print "Black: ", blackScore
print "White: ", whiteScore
if blackScore - whiteScore > 0:
    print "Result: B +", blackScore-whiteScore
else:
    print "Result: W +", whiteScore-blackScore
cv2.destroyAllWindows()