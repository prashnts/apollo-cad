import numpy.testing as nt
import numpy as np

from verify import expect

from ..metrics import statistics

def test_scan_axis():
  img = np.array([
    [ 0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  1,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  1,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  1,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
  ], dtype='float64')
  scn = statistics.ScanLine(img)
  p0 = scn._profile()
  (expect(p0)
    .to_be_list()
    .to_have_length(min=2, max=2))
