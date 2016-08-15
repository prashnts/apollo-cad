import skimage.color
import skimage.exposure
import skimage.filters
import skimage.util


BIN_COUNT = 5
KERNEL_SIZE = 200
CLIP_LIMIT = 0.99

def _to_grayscale(img):
  return skimage.color.rgb2gray(img)

def equalize(img, kernel=KERNEL_SIZE, bins=BIN_COUNT, clip=CLIP_LIMIT):
  eqimg = (skimage.exposure
      .equalize_adapthist(img, kernel_size=kernel,
          nbins=bins, clip_limit=CLIP_LIMIT))
  return skimage.color.rgb2gray(eqimg)
