# import pafy
import imageio
import pylab
import numpy as np

url = "https://www.youtube.com/watch?v=7bubb3Wn-48"
filename = "chess1.mp4"
# video = pafy.new(url)
# print(video.title)
# streams = video.videostreams
# for s in streams:
    # print(s)
# streams[1].download(filename, quiet=False)
vid = imageio.get_reader(filename, "ffmpeg")
board_y = np.zeros(144, dtype=bool)

X = (170, 248)
Y = (61, 139)
fps = 30

for second in range(int(vid.get_length()/30)):
    image = vid.get_data(second*fps)
    board_image = image[Y[0]:Y[1], X[0]:X[1], :]
    imageio.imwrite("images/board_{}s.jpg".format(second), board_image)
