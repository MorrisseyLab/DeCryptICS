
from MiscFunctions import plot_img
from read_svs_class import svs_file_w_labels
import albumentations as A
import cv2

file_test = '/home/doran/Work/images/Leeds_May2019/curated_cryptdata/train/KM13/KM13K_444695.svs'
svs_i = svs_file_w_labels(file_test, 1024, 1.1)
img, mask = svs_i.fetch_clone(0, 0)

composition = A.Compose([
    A.HorizontalFlip(), A.VerticalFlip(), A.Rotate(border_mode = cv2.BORDER_CONSTANT),
    A.OneOf(
    [
          A.ElasticTransform(alpha = 1000, sigma = 30,
                           alpha_affine = 30, border_mode = cv2.BORDER_CONSTANT, p=1),
          A.GridDistortion(border_mode = cv2.BORDER_CONSTANT, p = 1),
          A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5,
                                border_mode = cv2.BORDER_CONSTANT, p = 1),
    ],  p=0.7),
    A.CLAHE(p=0.5),
    A.OneOf(
    [
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=1),
        A.RandomBrightnessContrast(p=1),
    ], p=0.5),
    A.JpegCompression(p=0.2),
    A.OneOf(
    [
          A.MedianBlur(p=1),
          A.Blur(p=1),
    ],  p=0.3)
   ],  p = 1)

composition = A.Compose([
    A.OneOf(
    [
          A.JpegCompression(p=1),
          A.MedianBlur(p=1),
          A.Blur(p=1),
    ],  p=0.8)
   ],  p = 1)
   
composition = A.Compose([
    A.Posterize(p=0.5, num_bits=4),
   ],  p = 1)
   
imglist = []
for i in range(4):
    transformed = composition(image=img, mask=mask[:,:,0:3])
    imglist += [transformed['image'], transformed['mask']]
plot_img(tuple(imglist), nrow = 2)
