import imageio

HEIGHT = 26
WIDTH = 40
FACE = 1
CAR = -1

def read_file(fname):
    image = imageio.imread(fname)
    assert image.shape == (HEIGHT, WIDTH, 3)
    return image[:,:,0].cumsum(axis=0).cumsum(axis=1).view('int64')