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

  assert geometry.point_of_intersection(*pt1) == (2, 2)
  assert geometry.point_of_intersection(*pt2) == (0, 0)
  assert geometry.point_of_intersection(*pt3) == None
