import pafy
import imageio
from os import path
from os import makedirs
# import pylab
# import numpy as np

download_folder = "../downloads"

def ensure_dir(file_path):
    directory = path.dirname(file_path)
    if not path.exists(directory):
        makedirs(directory)

def download(url="https://www.youtube.com/watch?v=7bubb3Wn-48", filename="../downloads/videos/chess1.mp4"):
    # url = "https://www.youtube.com/watch?v=7bubb3Wn-48"
    video = pafy.new(url)
    print("downloading {} ".format(video.title))
    streams = video.videostreams
    ensure_dir(filename)
    streams[1].download(filename, quiet=False)


def create_images(filename="../downloads/videos/chess1.mp4"):
    ensure_dir(filename)
    ensure_dir("../downloads/images/test.png")
    vid = imageio.get_reader(filename, "ffmpeg")
    X_digital, Y_digital = (170, 248), (61, 139)
    X_stream, Y_stream = (6, 164), (24, 127)
    fps = 30
    for second in range(int(vid.get_length()/30)):
        image = vid.get_data(second*fps)
        digital_board = image[Y_digital[0]:Y_digital[1], X_digital[0]:X_digital[1], :]
        imageio.imwrite("../downloads/images/digital_board_{}s.png".format(second), digital_board)
        stream_board = image[Y_stream[0]:Y_stream[1], X_stream[0]:X_stream[1], :]
        imageio.imwrite("../downloads/images/stream_board_{}s.png".format(second), stream_board)
