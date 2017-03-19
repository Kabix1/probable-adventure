import re
import cv2
import numpy as np
from glob import glob
import os


class piece:
    def __init__(self, name, threshold, FEN, pos, template_path="../data/templates/"):
        self.name = name
        self.threshold = threshold
        self.FEN = FEN
        self.pos = pos
        self.template_path = template_path
        self.update_templates()

    def update_templates(self):
        self.templates = glob("{}{}*.png".format(self.template_path, self.name))


class analyser:
    """Analyses png files and outputs fen files"""

    pieces = []

    thresholds = [
        ('white_rook', 0.6), ('white_knight', 0.6), ('white_bishop', 0.85),
        ('white_king', 0.6), ('white_queen', 0.87), ('white_pawn', 0.55),
        ('black_rook', 0.6) , ('black_knight', 0.8), ('black_bishop', 0.9),
        ('black_king', 0.8), ('black_queen', 0.87), ('black_pawn', 0.85)
        ]
 
    FEN_TABLE = {
        "white_rook":"R", "white_knight":"N", "white_bishop":"B",
        "white_king":"K", "white_queen":"Q", "white_pawn":"P",
        "black_rook":"r", "black_knight":"n", "black_bishop":"b",
        "black_king":"k", "black_queen":"q", "black_pawn":"p"
        }


    def __init__(self, source_path="../downloads/images/",
                 source_pattern="digital_board*",
                 template_path="../data/templates/",
                 FEN_path="../data/FEN_files/"):
        """initialise path variables"""
        self.source_path = source_path
        self.source_pattern = source_pattern
        self.template_path = template_path
        self.FEN_path = FEN_path

        self.pieces.append(piece("white_rook", 0.8, "R", [(0,0), (7,0)]))
        self.pieces.append(piece("white_knight", 0.6, "N", [(1,0), (6,0)]))
        self.pieces.append(piece("white_bishop", 0.57, "B", [(2,0), (5,0)]))
        self.pieces.append(piece("white_king", 0.8, "K", [(4,0)]))
        self.pieces.append(piece("white_queen", 0.8, "Q", [(3,0)]))
        self.pieces.append(piece("white_pawn", 0.65, "P", [(0,1), (1,1)]))

        self.pieces.append(piece("black_rook", 0.8, "r", [(0,7), (7,7)]))
        self.pieces.append(piece("black_knight", 0.8, "n", [(1,7), (6,7)]))
        self.pieces.append(piece("black_bishop", 0.9, "b", [(2,7), (5,7)]))
        self.pieces.append(piece("black_king", 0.8, "k", [(4,7)]))
        self.pieces.append(piece("black_queen", 0.95, "q", [(3,7)]))
        self.pieces.append(piece("black_pawn", 0.85, "p", [(0,6), (1,6)]))


    def set_thresholds(self, thresholds):
        """change thresholds for templates [... ('piece', threshold) ...]"""
        self.thresholds = thresholds


    def _get_template(self, board, pos):
        sq = board.shape[0] / 8
        X = (round(sq * pos[0]), round(sq*(pos[0] + 1)))
        Y = ((7-pos[1]) * sq, (7-pos[1] + 1)*sq)
        return board[Y[0]:Y[1], X[0]:X[1], :]

    def make_templates(self):
        img = cv2.imread("{}board.png".format(self.template_path))
        for piece in self.pieces:
            for pos, i in list(zip(piece.pos, range(1, len(piece.pos) + 1))):
                template = self._get_template(img, pos)
                cv2.imwrite("{}{}{}.png".format(self.template_path,
                                                piece.name, i), template)
            piece.update_templates()


    def get_position(self, point, square):
        x = point[0] + square[0] / 2
        y = point[1] + square[1] / 2
        return (int(x / square[0]), int(y / square[1]))

    def image_to_FEN(self, img):
        """takes img and returns fen file matching the position"""
        fen = ""
        FEN_ending = " w - - 0 1"
        board = [x[:] for x in [[""] * 8] * 8]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for piece in self.pieces:
            for template in piece.templates:
                temp_img = cv2.imread(template, 0)
                res = cv2.matchTemplate(img_gray, temp_img, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= piece.threshold)
                for point in zip(*loc[::-1]):
                    pos = self.get_position(point, temp_img.shape)
                    if 0 <= pos[0] <= 7 and 0 <= pos[1] <= 7:
                        board[pos[1]][pos[0]] = piece.FEN
                        # print("found piece {} at {}".format(piece.name, pos))

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
        fen += FEN_ending
        print(fen)
        return fen


    def test_FEN(self):
        FEN_strings = {}
        FEN_files = glob("{}{}.fen".format(self.FEN_path, self.source_pattern))
        for FEN in FEN_files:
            sec = int(re.match(".*digital_board_(\d+).*", FEN).groups()[0])
            string = open(FEN, 'r').read()
            FEN_strings[sec] = string

        old_characters = {}
        keys = sorted(FEN_strings)
        for key in keys[19:]:
            characters = {}
            for char in FEN_strings[key]:
                if char.isalpha():
                    if char in characters:
                        characters[char] = characters[char] + 1
                    else:
                        characters[char] = 1
            if old_characters != {}:
                for char in old_characters.keys():
                    if char in characters and char in old_characters:
                        if characters[char] > old_characters[char]:
                            print("Error in digital_board_{}s, threshold of {} too low".format(key, char))
            old_characters = characters
        if FEN_strings[keys[19]] != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBqKBNR/ w - - 0 1":
            print("Threshold too high!")



    def run(self):
        """runs self.image_to_FEN for all png-files
        matching source_pattern in source_path"""
        images = glob("{}{}.png".format(self.source_path, self.source_pattern))
        print(len(images))
        for image in images:
            img_rgb = cv2.imread(image)
            fen = self.image_to_FEN(img_rgb)
            FEN_path = "{}{}.fen".format(self.FEN_path, os.path.basename(image)[:-4])
            FEN_file = open(FEN_path, "w")
            FEN_file.write(fen)
            FEN_file.close()

test = analyser()
test.run()
# test.make_templates()
