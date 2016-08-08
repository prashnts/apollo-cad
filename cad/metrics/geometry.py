import numpy as np
import skimage.feature
import skimage.transform

from pydash import py_


THETA_INCREMENT = 0.05
THETA_SPAN = 5

THETA_H = np.deg2rad(np.arange(0 - THETA_SPAN, THETA_SPAN, THETA_INCREMENT))
THETA_V = np.deg2rad(np.arange(90 - THETA_SPAN, 90 + THETA_SPAN, THETA_INCREMENT))

SIGMA = 1
THRESHOLD = 100
MIN_LENGTH = 100
MIN_GAP = 3

def infer_lines(img,
      axis=0,
      sigma=SIGMA,
      threshold=THRESHOLD,
      length=MIN_LENGTH,
      gap=MIN_GAP,
      theta=None):
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
  x, y = py_(points).unzip().value()
  if axis == 0:
    return (np.percentile(x, 10), min(y)), (np.percentile(x, 90), max(y))
  else:
    return (min(x), np.percentile(y, 10)), (max(x), np.percentile(y, 90))

(py_(lines)
    .flatten()
    .thru(lambda x: grouped_points(x, axis=0))
    .map_(lambda x: approximate_line_span(x, axis=0))
    .value())
