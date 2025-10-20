import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),      
        (Rotate, 0, 30),
        (Sharpness,0.05, 0.95),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (Solarize, 0, 256),  
        (TranslateX, 0., 0.3),
        (TranslateY, 0., 0.3)
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = random.uniform(0,1)*(maxval - minval) + minval
            img = op(img, val)
        img = Cutout(img, random.uniform(0,1)*0.5)
        return img

# --- Add these imports at top if not already present ---
import random
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

# --- Grayscale-friendly RandAugment (no RGB roundtrips) ---
class GrayRandAugment:
    """
    RandAugment variant tailored for single-channel (grayscale) images.

    Key ideas:
    - Keep ops that are meaningful on grayscale and inexpensive.
    - Avoid color-only ops (Color, Posterize with 3ch tuning, etc.).
    - Cutout uses scalar fill compatible with 'L' mode.
    - All ops work on PIL images in 'L' mode, and we never convert to RGB.

    Parameters
    ----------
    n : int
        Number of ops to apply sequentially.
    m : int
        Magnitude index in [0, 30]. Maps to op-specific ranges.
    magnitude_std : float
        If >0, sample magnitude from Normal(m, magnitude_std) and clip to [0, 30].
    cutout_fill : int
        Fill value for Cutout (0~255). 0 = black, 128 = gray, 255 = white.
    """

    def __init__(self, n=2, m=10, magnitude_std=0.0, cutout_fill=0):
        self.n = n
        self.m = m
        self.magnitude_std = magnitude_std
        self.cutout_fill = int(cutout_fill)
        self._ops = [
            self._shear_x, self._shear_y,
            self._translate_x, self._translate_y,
            self._rotate,
            self._invert,
            self._equalize,
            self._solarize,
            self._contrast,   # works on 'L' via ImageEnhance
            self._sharpness,  # works on 'L'
            self._brightness, # works on 'L'
            self._cutout,
        ]

    # -------------------------
    # Magnitude helpers
    # -------------------------
    @staticmethod
    def _float_param(m, maxval):  # m in [0, 30]
        return float(m) / 30.0 * maxval

    @staticmethod
    def _int_param(m, maxval):
        return int(GrayRandAugment._float_param(m, maxval))

    def _sample_m(self):
        if self.magnitude_std > 0:
            mag = random.gauss(self.m, self.magnitude_std)
            mag = max(0, min(30, mag))
            return mag
        return self.m

    # -------------------------
    # Ops (grayscale safe)
    # -------------------------
    def _shear_x(self, img, m):
        # Shear range [-0.3, 0.3]
        v = self._float_param(m, 0.3)
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0), resample=Image.BILINEAR)

    def _shear_y(self, img, m):
        v = self._float_param(m, 0.3)
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0), resample=Image.BILINEAR)

    def _translate_x(self, img, m):
        # Translate in pixels up to ±(img.width * 0.45)
        max_shift = int(img.size[0] * 0.45)
        v = self._int_param(m, max_shift)
        if random.random() < 0.5:
            v = -v
        # Use affine transform instead of ImageOps.offset for broad Pillow compatibility
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0),
                                resample=Image.BILINEAR, fillcolor=0)
        # try:
        #     return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0),
        #                         resample=Image.BILINEAR, fillcolor=0)
        # except TypeError:
        #     # Fallback for very old Pillow without 'fillcolor' param
        #     bg = Image.new(img.mode, img.size, color=0)
        #     bg.paste(img, (v, 0))
        #     return bg.crop((0, 0, img.size[0], img.size[1]))

    def _translate_y(self, img, m):
        max_shift = int(img.size[1] * 0.45)
        v = self._int_param(m, max_shift)
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v),
                                resample=Image.BILINEAR, fillcolor=0)
        # try:
        #     return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v),
        #                         resample=Image.BILINEAR, fillcolor=0)
        # except TypeError:
        #     bg = Image.new(img.mode, img.size, color=0)
        #     bg.paste(img, (0, v))
        #     return bg.crop((0, 0, img.size[0], img.size[1]))

    def _rotate(self, img, m):
        # Angle in degrees up to ±30
        v = self._float_param(m, 30.)
        if random.random() < 0.5:
            v = -v
        # Use fill=0 to keep background black for EMNIST-like digits
        return img.rotate(v, resample=Image.BILINEAR, fillcolor=0)

    def _invert(self, img, m):
        # Invert works on 'L'
        return ImageOps.invert(img)

    def _equalize(self, img, m):
        return ImageOps.equalize(img)

    def _solarize(self, img, m):
        # Threshold in [0, 256)
        v = self._int_param(m, 256)
        return ImageOps.solarize(img, threshold=v)

    def _contrast(self, img, m):
        # Factor in [0.1, 1.9]
        v = 0.1 + self._float_param(m, 1.8)
        return ImageEnhance.Contrast(img).enhance(v)

    def _sharpness(self, img, m):
        v = 0.1 + self._float_param(m, 1.8)
        return ImageEnhance.Sharpness(img).enhance(v)

    def _brightness(self, img, m):
        v = 0.1 + self._float_param(m, 1.8)
        return ImageEnhance.Brightness(img).enhance(v)

    def _cutout(self, img, m):
        """
        Cutout with a single gray value on 'L' images.
        Box size scales with magnitude. The box is square.
        """
        # Box size up to 40% of the shortest side
        max_len = int(min(img.size) * 0.4)
        box_len = self._int_param(m, max_len)
        if box_len < 1:
            return img

        w, h = img.size
        x0 = random.randint(0, max(0, w - box_len))
        y0 = random.randint(0, max(0, h - box_len))
        x1, y1 = x0 + box_len, y0 + box_len

        out = img.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle([x0, y0, x1, y1], fill=self.cutout_fill)
        return out

    # -------------------------
    # Callable
    # -------------------------
    def __call__(self, img):
        """
        Apply 'n' randomly selected ops with (possibly jittered) magnitude.
        Assumes 'img' is a PIL.Image in mode 'L'. If not, it will be converted.
        """
        if img.mode != "L":
            img = img.convert("L")

        ops = random.sample(self._ops, k=min(self.n, len(self._ops)))
        for op in ops:
            mag = self._sample_m()
            img = op(img, mag)
        return img
