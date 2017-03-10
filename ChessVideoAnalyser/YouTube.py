import pafy
import imageio
# import pylab
# import numpy as np

def download(url="https://www.youtube.com/watch?v=7bubb3Wn-48", filename="chess1.mp4"):
    # url = "https://www.youtube.com/watch?v=7bubb3Wn-48"
    video = pafy.new(url)
    print("downloading {} ".format(video.title))
    streams = video.videostreams
    streams[1].download(filename, quiet=False)


def create_images(filename="chess1.mp4"):
    vid = imageio.get_reader(filename, "ffmpeg")
    X_digital, Y_digital = (170, 248), (61, 139)
    X_stream, Y_stream = (6, 164), (24, 127)
    fps = 30
    for second in range(int(vid.get_length()/30)):
        image = vid.get_data(second*fps)
        digital_board = image[Y_digital[0]:Y_digital[1], X_digital[0]:X_digital[1], :]
        imageio.imwrite("images/digital_board_{}s.png".format(second), digital_board)
        stream_board = image[Y_stream[0]:Y_stream[1], X_stream[0]:X_stream[1], :]
        imageio.imwrite("images/stream_board_{}s.png".format(second), stream_board)
