import skimage.transform

class ScanLine(object):
  def __init__(self, img):
    self.img = img

  def _scan(self, img, axis=0):
    _a = 0 if axis else 1
    return img.mean(axis=_a)

  def _profile(self, theta=0, cval=1):
    img = skimage.transform.rotate(self.img, theta, cval=cval)
    return [self._scan(img=img, axis=x) for x in [0, 1]]
