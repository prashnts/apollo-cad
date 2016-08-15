import numpy as np
import skimage.feature
import skimage.transform
import scipy

from pydash import py_


THETA_INCREMENT = 0.5
THETA_SPAN = 10

THETA_H = np.deg2rad(np.arange(0 - THETA_SPAN, THETA_SPAN, THETA_INCREMENT))
THETA_V = np.deg2rad(np.arange(90 - THETA_SPAN, 90 + THETA_SPAN, THETA_INCREMENT))

SIGMA = 1
THRESHOLD = 100
MIN_LENGTH = 100
MIN_GAP = 3

def _filter_lines(img,
      axis=0,
      sigma=SIGMA,
      threshold=THRESHOLD,
      length=MIN_LENGTH,
      gap=MIN_GAP,
      theta=None,
      **kwa):
  if not theta:
    theta = THETA_V if axis else THETA_H
  edges = skimage.feature.canny(img, sigma)
  return skimage.transform.probabilistic_hough_line(edges,
      threshold=threshold,
      line_length=length,
      line_gap=gap,
      theta=theta)


def group_adaptive(points, axis=0):
  """Group points that are possibly the gridlines.

  The method adaptively estimates the groups each points may belong to. An
  overview is below:
  - Sort the points in increasing order of primary axis.
  - Calculate the increments in secondary axis.
  - Calculate incremental threshold as the 90th percentile of increments.
  - Split the ``points`` array wherever the increment is greater than the
    threshold.

  Args:
      points (List of (x, y) tuples): Points that describe line segments.
      axis (int:`{0, 1}`, optional): Specify inference axis.

  Returns:
      List of lists of points, each grouped together.
  """
  points = py_(points).sort(key=lambda x: x[axis])
  principle = (points
      .clone(is_deep=True)
      .map_(lambda x: x[axis])
      .value())
  n = len(principle)
  increments = [0] + [abs(principle[i + 1] - principle[i]) for i in range(n - 1)]
  di = py_.uniq(increments)
  threshold = np.percentile(increments, 90)
  result = []
  group = []

  for pair in points.zip(increments).value():
    if pair[1] >= threshold:
      group.sort(key=lambda x: x[1 if not axis else 0])
      result.append(group.copy())
      del group[:]
    group.append(pair[0])
  result.append(group)
  return result


def approximate_line_span(points, axis=0):
  naxis = int(not axis)
  coors = py_(points).sort(key=lambda n: n[axis]).unzip().value()

  c_a = [np.percentile(coors[axis], p) for p in (10, 90)]
  c_b = [fn(coors[naxis]) for fn in (min, max)]
  # Slope of `c_b` must be same as `coors[naxis]`. We will reverse c_b if
  # coors[naxis] is a decreasing set (_WITHOUT_ reordering).
  dcn = (np.diff(coors[naxis]) > 0).sum() < (len(coors[naxis]) / 2)
  if dcn:
    c_b.reverse()
  spans = [c_a, c_b]
  if axis:
    spans.reverse()
  return list(zip(*spans))


def point_of_intersection(l, m):
  def slope_intercept(k):
    (x1, y1), (x2, y2) = k
    m = (y2 - y1) / (x2 - x1)
    i = y1 - (m * x1)
    return m, i
  a, c = slope_intercept(l)
  b, d = slope_intercept(m)
  x = (d - c) / (a - b)
  y = (a * x) + c
  return x, y


def find_convex_hull_rect(points):
  points = np.array(points)
  hull = scipy.spatial.ConvexHull(points)
  v = hull.vertices
  vertices = [points[i] for i in v]
  p = [[v[i], v[(i + 1) % len(v)]] for i in range(len(v))]
  vertex_pairs = [points[i] for i in p]
  edges = [np.linalg.norm(np.diff(pt), ord=2) for pt in vertex_pairs]
  length, breadth = max(edges[0], edges[3]), max(edges[1], edges[2])
  return vertex_pairs, length, breadth


def infer_lines(img, axis=0, **kwa):
  """Infer line span from broken segments.

  Args:
      img (ndarray:`float64`): Grayscaled image matrix.
      axis (int:`{0, 1}`, optional): Specify inference axis.
      **kwa: Passed to ``_filter_lines`` routine.

  Returns:
      Combined point-pairs that span (approximately) through the observed
      line segments. Example::

          [((1, 1), (10, 1)),
           ((2, 2), (9,  2))]

  """
  segments = _filter_lines(img, axis, **kwa)
  return (py_(segments)
      .flatten()
      .thru(lambda x: group_adaptive(x, axis))
      .map_(lambda x: approximate_line_span(x, axis))
      .value())


def infer_grid(img, **kwa):
  vt = infer_lines(img, axis=0, **kwa)
  hz = infer_lines(img, axis=1, **kwa)
  return vt, hz
