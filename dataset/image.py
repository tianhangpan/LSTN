from PIL import Image
import numpy as np
import h5py
import cv2


class LoadData:
    @staticmethod
    def test_data(img_path, mask):
        img_path = str(img_path)
        img = Image.open(img_path).convert('RGB')

        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])

        img = img.resize((640, 360))
        h, w = target.shape
        target = cv2.resize(target, (640, 360)) * (h / 360) * (w / 640)
        if not isinstance(mask, type(None)):
            mask = cv2.resize(mask, (640, 360))

        return img, target, mask
