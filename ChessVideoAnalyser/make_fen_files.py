import re
import cv2
import numpy as np
from glob import glob
import os



class analyser:
    """Analyses png files and outputs fen files"""

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


    def set_thresholds(self, thresholds):
        """change thresholds for templates [... ('piece', threshold) ...]"""
        self.thresholds = thresholds


    def make_templates(self):
        img = cv2.imread("{}board.png".format(self.template_path))
        sq = img.shape[0] / 8
        for i in range(8):
            X = (int(sq*i), int(sq*(i+1)))
            Y = (0, int(sq))
            print(X, Y)
            template = img[Y[0]:Y[1],X[0]:X[1],:]
            print(template)
            cv2.imwrite("template{}.png".format(i), template)


    def image_to_FEN(self, img):
        """takes img and returns fen file matching the position"""
        fen = ""
        FEN_ending = " w - - 0 1"
        board = [x[:] for x in [[""] * 8] * 8]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for piece, threshold in self.thresholds:
            template = cv2.imread('{}{}.png'.format(self.template_path, piece),0)
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
        fen += FEN_ending
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
        print(images[2])
        print(len(images))
        for image in images:
            img_rgb = cv2.imread(image)
            fen = image_to_FEN(img_rgb)
            FEN_path = "{}{}.fen".format(self.output_path, os.path.basename(image)[:-4])
            FEN_file = open(FEN_path, "w")
            FEN_file.write(fen)
            FEN_file.close()

test = analyser()
test.make_templates()