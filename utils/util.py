import math
import random
from os import environ
from platform import system

import cv2
import numpy
import torch
from PIL import Image, ImageEnhance, ImageOps

max_value = 10.0


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def plot_lr(args, optimizer, scheduler):
    import copy
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        y.append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch + 1, optimizer)

    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('epoch')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def weight_decay(model, decay):
    p1 = []
    p2 = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias"):
            p1.append(param)  # bias (no decay)
        else:
            p2.append(param)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


@torch.no_grad()
def accuracy(outputs, targets, top_k):
    results = []
    outputs = outputs.topk(max(top_k), 1, True, True)[1].t()
    outputs = outputs.eq(targets.view(1, -1).expand_as(outputs))

    for k in top_k:
        correct = outputs[:k].reshape(-1)
        correct = correct.float().sum(0, keepdim=True)
        results.append(correct.mul_(100.0 / targets.size(0)))
    return results


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def rotate(image, m):
    m = (m / max_value) * 30.0

    if random.random() > 0.5:
        m *= -1

    return image.rotate(m, resample=resample())


def shear_x(image, m):
    m = (m / max_value) * 0.30

    if random.random() > 0.5:
        m *= -1

    return image.transform(image.size, Image.AFFINE, (1, m, 0, 0, 1, 0), resample=resample())


def shear_y(image, m):
    m = (m / max_value) * 0.30

    if random.random() > 0.5:
        m *= -1

    return image.transform(image.size, Image.AFFINE, (1, 0, 0, m, 1, 0), resample=resample())


def translate_x(image, m):
    m = (m / max_value) * 0.30

    if random.random() > 0.5:
        m *= -1

    pixels = m * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())


def translate_y(image, m):
    m = (m / max_value) * 0.30

    if random.random() > 0.5:
        m *= -1

    pixels = m * image.size[1]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def equalize(image, _):
    return ImageOps.equalize(image)


def normalize(image, _):
    return ImageOps.autocontrast(image)


def brightness(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Brightness(image).enhance(m)


def color(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Color(image).enhance(m)


def contrast(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Contrast(image).enhance(m)


def sharpness(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Sharpness(image).enhance(m)


def solar(image, m):
    if random.random() > 0.5:
        m = min(256, int((m / max_value) * 256))
    else:
        m = min(128, int((m / max_value) * 110))
    return ImageOps.solarize(image, m)


def poster(image, m):
    if random.random() > 0.5:
        m = int((m / max_value) * 4)
    else:
        m = int((m / max_value) * 4) + 4

    if m >= 8:
        return image
    return ImageOps.posterize(image, m)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        size = self.size
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([size, size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        if (size[0] / size[1]) < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (size[0] / size[1]) > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class RandomAugment:
    def __init__(self, mean=9.0, sigma=0.5, n=2, p=0.5):
        self.p = p
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, sharpness, solar, poster)

    def __call__(self, image):
        if random.random() > self.p:
            return image

        for transform in numpy.random.choice(self.transform, self.n):
            m = numpy.random.normal(self.mean, self.sigma)
            m = min(max_value, max(0.0, m))

            image = transform(image, m)
        return image


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num
