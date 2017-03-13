import cv2
import numpy as np
from glob import glob
import os

pieces = [('white_rook', 0.60), ('white_knight', 0.6), ('white_bishop', 0.85), ('white_king', 0.6) , ('white_queen', 0.87) , ('white_pawn', 0.55) , ('black_rook', 0.6) , ('black_knight', 0.8), ('black_bishop', 0.9), ('black_king', 0.8), ('black_queen', 0.87), ('black_pawn', 0.85)]

FEN_TABLE = {"white_rook":"R", "white_knight":"N", "white_bishop":"B", "white_king":"K", "white_queen":"Q", "white_pawn":"P", "black_rook":"r", "black_knight":"n", "black_bishop":"b", "black_king":"k", "black_queen":"q", "black_pawn":"p"}

def get_fen(img):
    fen = ""
    fen_ending = " w - - 0 1"
    board = [x[:] for x in [[""] * 8] * 8]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for piece, threshold in pieces:
        template = cv2.imread('../data/templates/{}.png'.format(piece),0)
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            center = (pt[0] + template.shape[0] / 2, pt[1] + template.shape[1] / 2)
            square_shape = tuple(map((1/8.0).__mul__, img_gray.shape[::-1]))
            letter = int(center[0]/square_shape[0])
            number = int(center[1]/square_shape[0])
            board[number][letter] = FEN_TABLE[piece]

    for number in range(8):
        empty = 0
        for letter in range(8):
            if board[number][letter] == "":
                empty += 1
            elif empty > 0:
                fen += str(empty)
                empty = 0
            fen += board[number][letter]
        if empty > 0:
            fen += str(empty)
        fen += "/"
    fen += fen_ending
    return fen

images = glob("../downloads/images/digital_board*.png")
print(images[2])
print(len(images))
for image in images:
    img_rgb = cv2.imread(image)
    fen = get_fen(img_rgb)
    fen_path = "../data/fen_files/{}.fen".format(os.path.basename(image)[:-4])
    fen_file = open(fen_path, "w")
    fen_file.write(fen)
    fen_file.close()

