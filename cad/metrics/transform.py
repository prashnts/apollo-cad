import numpy as np
import skimage.transform

def compute_transformation_matrix(vertices, length, breadth):
  predicate = lambda x: np.linalg.norm(x, ord=2)
  source = [
    [0, 0],
    [0, length],
    [breadth, length],
    [breadth, 0]]
  vertices = list(vertices)
  source.sort(key=predicate)
  vertices.sort(key=predicate)
  src, dst = map(np.array, [source, vertices])

  tf = skimage.transform.ProjectiveTransform()
  tf.estimate(src, dst)
  return tf


def apply_transform(img, tf, length, breadth):
  shape = (int(length), int(breadth))
  return skimage.transform.warp(img, tf, output_shape=shape)

