#!/usr/bin/python
# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'

# Note: because initialization is slow and downloading of weights also slow, please change line in module vot.tracker.trax.py
# in function `trax_python_adapter` line
# return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)
# to
# return TraxTrackerRuntime(tracker, command, log=log, timeout=30000, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)
# to find location use "python -m site"

# Link 1: https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/playground/ImageBind_SAM
# Link 2: https://github.com/facebookresearch/segment-anything
# Link 3: https://github.com/facebookresearch/dinov2

if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import vot
import numpy as np
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
import gzip
from skimage.measure import regionprops
import contextlib
from segment_anything import build_sam, SamAutomaticMaskGenerator
import pickle
import cv2
import requests
import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, normalize
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


LIMIT_PROCESSED_FRAMES = 500000


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def get_embedding_from_image(img1, dinov2_model, transforms, device):
    img1 = Image.fromarray(img1)
    img1 = transforms(img1).to(device).unsqueeze(0)
    with torch.no_grad():
        emb = dinov2_model(img1).cpu().numpy()
    return emb


def save_debug_info(txt):
    root_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    out = open(root_path + "debug.txt", "a")
    out.write(txt)
    out.close()


def save_debug_image(img, suffix):
    root_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    cv2.imwrite(root_path + 'debug_{}.jpg'.format(suffix), img)



class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def make_classification_eval_transform(
        resize_size: int = 224,
        interpolation=transforms.InterpolationMode.BICUBIC,
        crop_size: int = 224,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return transforms.Compose(transforms_list)


mask_generator = None
dinov2_model = None
dinov2_transforms = None


class ZFTracker(object):

    def __init__(self, image, masks_list):
        global mask_generator
        global dinov2_model, dinov2_transforms

        tracker_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Segment Anything
        if mask_generator is None:
            weights_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            local_path = tracker_dir + "sam_vit_h_4b8939.pth"
            if not os.path.isfile(local_path):
                torch.hub.download_url_to_file(weights_path, local_path, progress=False)
            mask_generator = SamAutomaticMaskGenerator(
                build_sam(checkpoint=local_path).to(device),
                points_per_side=16,
            )

        # Open CLIP
        if dinov2_model is None:
            if 0:
                weights_path = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
                local_path = tracker_dir + "dinov2_vitl14_pretrain.pth"
                if not os.path.isfile(local_path):
                    torch.hub.download_url_to_file(weights_path, local_path, progress=False)
            dinov2_model = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitl14',
                verbose=False,
            )
            dinov2_model.eval()
            dinov2_model.to(device)
            dinov2_transforms = make_classification_eval_transform()

        self.image = image.copy()

        # Track all objects
        self.emb_template = []
        self.template_masked = []
        for object_id, mask in enumerate(masks_list):

            # save_debug_image(255 * mask, "_mask_orig1_{}_{}".format(0, object_id))

            # Fix for strange masks
            mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_full[:mask.shape[0], :mask.shape[1]] = mask
            mask = mask_full

            # save_debug_image(255 * mask, "_mask_orig2_{}_{}".format(0, object_id))

            bboxes = regionprops(255 * mask)
            if len(bboxes) > 1:
                print('Many masks found!')
                exit()
            prop = bboxes[0]

            # Left only mask
            image_masked = image.copy()
            image_masked[mask == 0] = 0
            self.mask = mask.copy()
            self.template_masked.append(
                image_masked[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]].copy()
            )
            # save_debug_image(self.template_masked[-1], "_mask_zf_t2_{}_{}".format(0, object_id))
            self.emb_template.append(
                get_embedding_from_image(
                    self.template_masked[-1],
                    dinov2_model,
                    dinov2_transforms,
                    self.device,
                )
            )
            self.current_image = 0

    def track(self, image):

        if self.current_image < LIMIT_PROCESSED_FRAMES:
            masks = mask_generator.generate(image)

            embeddings = []
            for mask in masks:
                image_masked = image.copy()
                image_masked[mask["segmentation"].astype(np.uint8) == 0] = 0
                x1, y1, x2, y2 = convert_box_xywh_to_xyxy(mask["bbox"])
                try:
                    img1 = image_masked[y1:y2, x1:x2]
                except Exception as e:
                    save_debug_info("{} {} {} {} {} {}\n".format(self.current_image, x1, y1, x2, y2, mask["bbox"]))
                    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
                emb = get_embedding_from_image(
                    img1,
                    dinov2_model,
                    dinov2_transforms,
                    self.device,
                )
                embeddings.append(emb)

            X = np.concatenate(embeddings, axis=0)
            Y = np.concatenate(self.emb_template, axis=0)
            distances = cosine_similarity(X, Y)

            best_indexes = np.argmax(distances, axis=0)
            mask_list = []
            for object_id, best_index in enumerate(best_indexes):
                print('Object: {} Masks found: {} Best index: {} Max value: {:.6f}'.format(
                    object_id,
                    len(masks),
                    best_index,
                    distances[best_index, object_id])
                )
                mask_chosen = masks[best_index]["segmentation"].astype(np.uint8).copy()
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[...] = mask_chosen

                # save_debug_info("{} {} {} {} {} {}\n".format(self.current_image, object_id, mask.dtype, mask.shape, mask.min(), mask.max()))
                mask_list.append(mask)
        else:
            mask_list = []
            for i in range(len(self.emb_template)):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask_list.append(mask)

        self.current_image += 1
        return mask_list


handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()
imagefile = handle.frame()
image = cv2.imread(imagefile)
print('Image path: {} Objects: {}'.format(imagefile, len(objects)))
tracker = ZFTracker(image, objects)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    mask_list = tracker.track(image)
    handle.report(mask_list)
