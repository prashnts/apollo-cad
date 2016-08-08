import skimage.color
import skimage.exposure
import skimage.filters


BIN_COUNT = 10
BLOCK_SIZE = 35
KERNEL_SIZE = 50


def _to_grayscale(img):
  return skimage.color.rgb2gray(im)

def equalize(img, kernel=KERNEL_SIZE, bins=BIN_COUNT, block=BLOCK_SIZE):
  gimg = _to_grayscale(img)
  adapted_histogram = (skimage.exposure
      .equalize_adapthist(gimg, kernel_size=kernel, nbins=bins))
  return (skimage.filters
      .threshold_adaptive(adapted_histogram, block_size=block))
