import cv2
import numpy as np

def plot_img(list_to_plot, nrow = 1, nameWindow = 'Plots', NewWindow = True, hold_plot = True):
    num_images = len(list_to_plot)
    num_cols   = int(num_images/nrow)
    if num_images%nrow != 0:
        raise(UserWarning, "If more than one row make sure there are enough images!")
    if NewWindow:
        screen_res = 1600, 1000
        cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nameWindow, screen_res[0], screen_res[1])
    if isinstance(list_to_plot, tuple) == 0: 
        vis = list_to_plot
    else:
        last_val = num_cols 
        vis      = np.concatenate(list_to_plot[0:last_val], axis=1)
        for row_i in range(1, nrow):
            first_val = last_val
            last_val  = first_val + num_cols
#            print (first_val,last_val)
            vis_aux   = np.concatenate(list_to_plot[first_val:last_val], axis=1)
            vis       = np.concatenate((vis, vis_aux), axis=0)        
    cv2.imshow(nameWindow, vis)
    if(hold_plot):
        0xFF & cv2.waitKey(0)
        cv2.destroyWindow(nameWindow)

# Method to set grayscale mask values to either 0 or 255
def fix_mask(mask):
    inds = np.where(np.isnan(mask))
    for i in range(inds[0].shape[0]):
      mask[inds[0][i],inds[1][i], inds[2][i]] = 0.0
    inds = np.where(np.isinf(mask))
    for i in range(inds[0].shape[0]):
      mask[inds[0][i],inds[1][i], inds[2][i]] = 0.0
    mask[mask < 100] = 0.0
    mask[mask >= 100] = 255.0

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.choice([90.,-90.,180.]) #np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180. * np.math.pi) * sx
        ss = np.math.sin(angle / 180. * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask
