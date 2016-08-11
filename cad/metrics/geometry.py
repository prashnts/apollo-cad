import numpy as np
import skimage.feature
import skimage.transform

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
  points = py_(points).sort(key=lambda x: x[axis])
  principle = (points
      .clone(is_deep=True)
      .map_(lambda x: x[axis])
      .value())
  n = len(principle)
  increments = [0] + [principle[i + 1] - principle[i] for i in range(n - 1)]
  di = py_.uniq(increments)
  threshold = sum([(increments.count(x) * x) for x in di]) / n
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
