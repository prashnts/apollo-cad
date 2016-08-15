import numpy.testing as nt

from verify import expect

from ..metrics import geometry


def test_group_adaptive():
  pts = [
    (0, 1),
    (8, 3),
    (9, 4),
    (2, 3),
    (1, 4),
    (1, 2),
    (8, 5),
    (7, 2),
    (2, 5),
    (8, 1),
  ]
  grouped = [
    [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)],
    [(7, 2), (8, 1), (8, 3), (8, 5), (9, 4)]
  ]
  groups = geometry.group_adaptive(pts, axis=0)
  (expect(groups)
      .to_be_list()
      .to_have_length(min=2, max=2))
  for x, y in zip(grouped, groups):
    expect(set(x) == set(y)).is_true()


def test_point_of_intersection():
  pt1 = [
    ((0, 0), (1, 1)),
    ((2, 2), (0, -1))]
  pt2 = [
    ((0, 0), (1, 1)),
    ((0, 1), (0, -1))]
  pt3 = [
    ((1, 1), (1, -1)),
    ((0, 0), (0, -1))]
  pt4 = [
    ((0, 0), (0, -1)),
    ((0, 0), (0, -1))]

  assert geometry.point_of_intersection(*pt1) == (2, 2)
  assert geometry.point_of_intersection(*pt2) == (0, 0)
  assert geometry.point_of_intersection(*pt3) == None
  assert geometry.point_of_intersection(*pt4) == None


def test_infer_lines_span():
  segs = [
    ((1273, 357),  (1273, 139)),
    ((1273, 1784), (1273, 383)),
    ((1295, 1812), (1295, 1)),
    ((149, 1115),  (149, 717)),
    ((148, 1442),  (148, 1160)),
    ((146, 854),   (146, 719)),
    ((357, 1114),  (357, 825)),
    ((352, 1431),  (352, 1317)),
    ((356, 1341),  (356, 1228)),
    ((353, 1189),  (353, 1010)),
    ((778, 1144),  (778, 1030))]
  spans = [
    ((146.0,  717),  (149.0,  1442)),
    ((352.0,  825),  (357.0,  1431)),
    ((778.0,  1030), (778.0,  1144)),
    ((1273.0, 1812), (1295.0, 1))]
  spans_found = geometry.infer_lines_span(segs, axis=0)
  assert set(spans) == set(spans_found)
