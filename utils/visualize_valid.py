# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import math
import sys
import os
import glob
import numpy as np
import gzip
import io
import argparse
import pickle
import cv2
from typing import List, Union, TextIO


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_rgb(im, name='image'):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))


def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)

    return mask, (tl_x, tl_y)


def create_video(image_list, out_file, fps, codec):
    height, width = image_list[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = -1
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)

    for im in image_list:
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=2)
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def parse_region(string: str) -> "Region":
    """
        Parse input string to the appropriate region format and return Region object

    Args:
        string (str): comma separated list of values

    Returns:
        Region: resulting region
    """
    from vot.region import Special
    from vot.region.shapes import Rectangle, Polygon, Mask

    if string[0] == 'm':
        # input is a mask - decode it
        m_, offset_ = create_mask_from_string(string[1:].split(','))
        return Mask(m_, offset=offset_)
    else:
        # input is not a mask - check if special, rectangle or polygon
        tokens = [float(t) for t in string.split(',')]
        if len(tokens) == 1:
            return Special(tokens[0])
        if len(tokens) == 4:
            if any([math.isnan(el) for el in tokens]):
                return Special(0)
            else:
                return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
        elif len(tokens) % 2 == 0 and len(tokens) > 4:
            if any([math.isnan(el) for el in tokens]):
                return Special(0)
            else:
                return Polygon([(x_, y_) for x_, y_ in zip(tokens[::2], tokens[1::2])])
    return None


def read_trajectory_binary(fp: io.RawIOBase):
    import struct
    from cachetools import LRUCache, cached
    from vot.region import Special
    from vot.region.shapes import Rectangle, Polygon, Mask

    buffer = dict(data=fp.read(), offset = 0)

    @cached(cache=LRUCache(maxsize=32))
    def calcsize(format):
        return struct.calcsize(format)

    def read(format: str):
        unpacked = struct.unpack_from(format, buffer["data"], buffer["offset"])
        buffer["offset"] += calcsize(format)
        return unpacked

    _, length = read("<hI")

    trajectory = []

    for _ in range(length):
        type, = read("<B")
        if type == 0: r = Special(*read("<I"))
        elif type == 1: r = Rectangle(*read("<ffff"))
        elif type == 2:
            n, = read("<H")
            values = read("<%df" % (2 * n))
            r = Polygon(list(zip(values[0::2], values[1::2])))
        elif type == 3:
            tl_x, tl_y, region_w, region_h, n = read("<hhHHH")
            rle = np.array(read("<%dH" % (n)), dtype=np.int32)
            r = Mask(rle_to_mask(rle, region_w, region_h), (tl_x, tl_y))
        else:
            raise IOError("Wrong region type")
        trajectory.append(r)
    return trajectory


def read_trajectory(fp: Union[str, TextIO]):
    if isinstance(fp, str):
        try:
            import struct
            with open(fp, "r+b") as tfp:
                v, = struct.unpack("<h", tfp.read(struct.calcsize("<h")))
                binary = v == 1
                # TODO: we can use the same file handle in case of binary format
        except Exception as e:
            binary = False

        fp = open(fp, "rb" if binary else "r")
        close = True
    else:
        binary = isinstance(fp, (io.RawIOBase, io.BufferedIOBase))
        close = False

    if binary:
        regions = read_trajectory_binary(fp)
    else:
        regions = []
        for line in fp.readlines():
            regions.append(parse_region(line.strip()))

    if close:
        fp.close()

    return regions


def read_ground_truth(gt_file):
    lines = open(gt_file).readlines()
    all_masks = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line == '0':
            all_masks.append(all_masks[-1])
        elif line == '1':
            prev_mask, offset = all_masks[-1]
            prev_mask[...] = 0
            all_masks.append((prev_mask, offset))
        else:
            mask, offset_ = create_mask_from_string(line[1:].split(','))
            # print(mask.shape)
            all_masks.append((mask, offset_))
    return all_masks


def gen_video(
        gt_file,
        bin_file,
        object_number,
        sequence,
        output_path,
        fps,
        codec,
):
    folder_name = os.path.basename(os.path.dirname(sequence))
    real_masks = read_ground_truth(gt_file)

    p = open(bin_file, "rb")
    regs = read_trajectory(p)
    all_frames = []
    for i, r in enumerate(regs):
        # print(str(r))
        # print(r.type)
        frame_path = sequence + 'color/{:08d}.jpg'.format(i + 1)
        if not os.path.isfile(frame_path):
            print('Error! Cant find frame: {}'.format(frame_path))
            exit()
        frame = cv2.imread(frame_path)
        mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
        if r.type == RegionType.MASK:
            # print(r.mask.shape)
            # show_image(255 * r.mask)
            # print(r.offset)
            # print(r.bounds())
            m1 = r.rasterize(r.bounds())
            # print(m1.shape)
            # show_image(255 * m1)
            mask_full[r.offset[1]:r.offset[1] + r.mask.shape[0], r.offset[0]:r.offset[0] + r.mask.shape[1]] = r.mask
            # show_image(255 * mask_full)
            frame[mask_full == 1, 2] = 255
        if 1:
            real_mask_small, offset = real_masks[i]
            real_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            real_mask[offset[1]:offset[1] + real_mask_small.shape[0],
            offset[0]:offset[0] + real_mask_small.shape[1]] = real_mask_small
            # show_image(255 * real_mask)
            frame[real_mask == 1, 1] = 255
        # show_image(frame)
        all_frames.append(frame.copy())
    output_video_path = output_path + "{}_{}.avi".format(folder_name, object_number)
    create_video(all_frames, output_video_path, fps, codec)


if __name__ == '__main__':
    from vot.region import RegionType

    # --tracker_name MyTracker --workspace /home/vot_workspace
    m = argparse.ArgumentParser()
    m.add_argument("--tracker_name", "-tn", type=str, help="Name of tracker", required=True)
    m.add_argument("--workspace", "-ws", type=str, help="Path to workspace", required=True)
    m.add_argument("--output_path", "-op", type=str, help="Output path for generated videos. By default it's folder 'videos' in workspace.")
    m.add_argument("--fps", type=float, help="FPS for generated video.", default=24)
    m.add_argument("--codec", type=str, help="Codec for generated video (openCV).", default="XVID")

    options = m.parse_args().__dict__
    print("Options: ".format(options))
    for el in options:
        print('{}: {}'.format(el, options[el]))

    fps = options['fps']
    codec = options['codec']
    tracker_name = options['tracker_name']
    workspace_path = options['workspace']
    if options["output_path"] is not None:
        output_path = options["output_path"]
    else:
        output_path = workspace_path + '/videos/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    all_folders = glob.glob(workspace_path + '/results/{}/baseline/*/'.format(tracker_name))
    all_folders_names = [os.path.basename(os.path.dirname(f)) for f in all_folders]
    print('Found folders with results: {}'.format(len(all_folders)))

    for i in range(len(all_folders_names)):
        folder_name = all_folders_names[i]
        folder_path = all_folders[i]
        print('Go for folder: {}'.format(folder_name))

        # Find original sequence path
        sequence = workspace_path + "/sequences/{}/".format(folder_name)
        if not os.path.isdir(sequence):
            print('Error! Cant find folder with original sequence: {}'.format(sequence))
            continue

        # Find all tracks for current folder
        bin_file_paths = glob.glob(folder_path + '/*.bin')
        # print(len(bin_file_paths))
        for bin_file in bin_file_paths:
            # Find ground file
            object_number = int(os.path.basename(bin_file).split("_")[1])
            gt_file = workspace_path + "/sequences/{}/groundtruth_{}.txt".format(folder_name, object_number)

            if not os.path.isfile(gt_file):
                print('Error! Cant find groundtruth file {}'.format(gt_file))
                continue

            print('Track: {}'.format(object_number))

            gen_video(
                gt_file,
                bin_file,
                object_number,
                sequence,
                output_path,
                fps,
                codec,
            )
