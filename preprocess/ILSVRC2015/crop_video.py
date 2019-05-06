import sys
import cv2
import math
import shutil
import logging
import numpy as np
import os.path as pt
import xml.etree.ElementTree as Et
from typing import Union, Iterable
from multiprocessing import Pool, Value

sys.path.insert(0, pt.join(pt.dirname(pt.abspath(__file__)), '../..'))
from preprocess import arg_set
from preprocess.ILSVRC2015 import video_list

# JPG saving quality set
para_jpg_qual = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

# counter for handled videos
handled_num = Value('i', 0)

# number of all videos
total_num = Value('i', 0)


def frame_xml_parse(xml_path: str) -> dict:
    """
    Parse a xml file of a frame
    :param xml_path: the path to the xml file
    :return: a dict containing the filename and objects info of the frame
    """
    xml_path = pt.normpath(xml_path)
    frame_info = dict()
    xy_seq = ('xmax', 'xmin', 'ymax', 'ymin')
    try:
        root = Et.parse(xml_path).getroot()
        frame_info['filename'] = root.find('filename').text

        # parse each object of this frame
        obj = []
        for child_obj in root.iter('object'):
            c_obj = dict()
            c_obj['name'] = arg_set.CLASS_IDS[child_obj.find('name').text]
            c_obj['trackid'] = int(child_obj.find('trackid').text)
            c_obj['occluded'] = int(child_obj.find('occluded').text)
            c_obj['generated'] = int(child_obj.find('generated').text)
            box = child_obj.find('bndbox')
            c_obj['bndbox'] = [int(box.find(a).text) for a in xy_seq]
            obj.append(c_obj)
        frame_info['object'] = obj
    except Exception:
        raise Et.ParseError('Fail:\n%s\n' % xml_path)
    return frame_info


def box_point(org_square_size: int, x_c: int, y_c: int):
    """
    compute coordinates of the squared box
    :param org_square_size:size of the squared box
    :param x_c:x coordinate of the center point
    :param y_c:y coordinate of the center point
    :return:[xmax, xmin, ymax. ymin]
    """
    org_square_size = int(org_square_size)
    xmin = int(round(x_c - org_square_size / 2))
    ymin = int(round(y_c - org_square_size / 2))
    return [xmin + org_square_size - 1, xmin, ymin + org_square_size - 1, ymin]


def cut_size(w: int, h: int):
    """
    give the size of the squared cropped object box
    :param w: original width
    :param h: original height
    :return: (s_z, s_x), according to the paper
    """
    p2 = (w + h) * arg_set.context_amount
    s_z = math.sqrt((w + p2) * (h + p2))
    s_x = round(arg_set.search_size * s_z / arg_set.exemplar_size)
    s_z = round(s_z)
    return s_z, s_x


def out_edge_flag(box, width: int, height: int) -> bool:
    """
    if or not the box is beyond the range of the original image
    :param box: [xmax, xmin, ymax, ymin]
    :param width: width of the original image
    :param height: height of hte original image
    :return: True for beyond the range, otherwise False
    """
    return any([tt < 0 for tt in box]) or box[0] >= width or box[2] >= height


def location_map(box, width: int, height: int):
    """
    When the coordinate range of the cropped image is beyond the original one's,
    this function give the mapping relationship between their coordinate ranges.
    :param box: cropped image range [xmax, xmin, ymax, ymin]
    :param width:width of the original image
    :param height:height of hte original image
    :return:original coordinate "org" mapped to cropped coordinate "cut".
            that means crop_img[cut] = org_img[org]
    """
    square_size = box[0] - box[1]
    org = [t for t in box]
    cut = [square_size, 0, square_size, 0]
    if box[1] < 0:
        org[1] = 0
        cut[1] = -box[1]
    if box[3] < 0:
        org[3] = 0
        cut[3] = -box[3]
    if box[0] >= width:
        org[0] = width - 1
        cut[0] = square_size + width - box[0] - 1
    if box[2] >= height:
        org[2] = height - 1
        cut[2] = square_size + height - box[2] - 1
    return org, cut


def pad_mean(img, img_mean):
    """
    Make an average image
    :param img: average image
    :param img_mean: a 3D vector
    :return: img
    """
    img[:, :, 0] = img_mean[0]
    img[:, :, 1] = img_mean[1]
    img[:, :, 2] = img_mean[2]


def img_paste(img, square_size: int, img_mean, box, width: int, height: int):
    """
    paste an object in the box from the height x width original image
    into an image with squared size, the other pixels are filled with the average value
    :param img: original image
    :param square_size: new image size
    :param img_mean: the average value of the original image, a 3D vector
    :param box: [xmax, xmin, ymax, ymin] for cropped region
    :param width: width of the original image
    :param height: height of the original image
    :return:cropped image
    """
    img_z = np.empty([square_size, square_size, 3], np.float)
    pad_mean(img_z, img_mean)
    org_loc_z, cut_loc_z = location_map(box, width, height)
    img_z[cut_loc_z[3]:cut_loc_z[2], cut_loc_z[1]:cut_loc_z[0], :] \
        = img[org_loc_z[3]:org_loc_z[2], org_loc_z[1]:org_loc_z[0], :]
    return img_z


def crop_obj_simple(img, crop_size: int, xy_center: Iterable[Union[int]], width_height: Iterable[Union[int]],
                    resize_size: int, img_mean):
    """
    crop the obj and resize it, if the crop size is beyond the image, fill these pixel with average value
    :param img: original image
    :param crop_size: crop size in the original image
    :param xy_center: the center of the target in the original image
    :param width_height: the width and height of the original image
    :param resize_size: resize the cropped one
    :param img_mean: the average color of the original image
    :return: crpped image resized
    """
    x_center, y_center = xy_center
    width, height = width_height
    assert width > 0, 'Invalid width:%d\n' % width
    assert height > 0, 'Invalid height:%d\n' % height
    assert crop_size > 2, 'Too small crop_size: %d\n' % crop_size
    assert resize_size > 2, 'Too small resize_size: %d\n' % resize_size

    box = box_point(crop_size, x_center, y_center)
    pad_flag = out_edge_flag(box, width, height)
    if pad_flag:
        img_crop = img_paste(img, int(crop_size), img_mean, box, width, height)
    else:
        img_crop = img[box[3]:box[2], box[1]:box[0], :]

    img_crop = cv2.resize(img_crop, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    return img_crop


def crop_obj(img, bndbox: Iterable[Union[int]], width: int, height: int, img_mean):
    """
    crop an object with proper range and resize it
    :param img: original image
    :param bndbox:coordinate of the box [xmax, xmin, ymax, ymin]
    :param width:width of the original image
    :param height:height of the original image
    :param img_mean:average value of the original image
    :return:cropped images img_z and img_x
    """
    assert len(bndbox) == 4, 'Invalid bndbox\n'
    assert all([x >= 0 for x in bndbox]), 'Invalid bndbox:[%d %d %d %d]\n' % bndbox
    xmax, xmin, ymax, ymin = bndbox

    w = xmax - xmin + 1
    h = ymax - ymin + 1
    assert w > 0, 'Error:xmax - xmin = %d\n' % w
    assert h > 0, 'Error:ymax - ymin = %d\n' % h

    s_z, s_x = cut_size(w, h)

    x_center = round((xmin + xmax) / 2)
    y_center = round((ymin + ymax) / 2)
    img_z = crop_obj_simple(img, s_z, [x_center, y_center], [width, height], arg_set.exemplar_size, img_mean)
    img_x = crop_obj_simple(img, s_x, [x_center, y_center], [width, height], arg_set.search_size, img_mean)
    return img_z, img_x


def check_exist_video(save_path: str, xml_set_path: str, frame_num: int) -> bool:
    """
    Make a directory for a video to save cropped images, or skip it if it exists already.
    :param save_path: path to make the directory
    :param xml_set_path: path to read the xml file
    :param frame_num: frame numbers of the video
    :return: True for skipping it, or False for making a new one
    """
    if pt.exists(save_path):
        str_idx = arg_set.frame_format % (frame_num - 1)
        xml_path = pt.join(xml_set_path, str_idx + '.xml')
        video_info = frame_xml_parse(xml_path)

        # if there is not any object in this video, just skip it
        if len(video_info['object']) == 0:
            return True

        trackid = video_info['object'][-1]['trackid']
        crop_x_idx = str_idx + arg_set.crop_x_format % trackid
        img_x_path = pt.join(save_path, crop_x_idx)
        img_x = cv2.imread(img_x_path)

        # if the last object in the last frame does not exist, this video should be re-handled
        if img_x is not None:
            return True
        shutil.rmtree(save_path)

    arg_set.make_new_dir(save_path)
    return False


def crop_one_video(relative_path: str, xml_set_path: str, frame_num: int):
    """
    Crop a video and return its information of every frame and every object
    :param relative_path: path to the video
    :param xml_set_path: path to the xml file
    :param frame_num: frame number of the video
    :return: a dict about this video
    """
    assert frame_num > 2, 'Too short video:\nframe_num=%d\nPath=%s\n' % (frame_num, relative_path)
    img, width, height, img_mean = None, None, None, None

    # path to save cropped images
    save_path = pt.join(arg_set.data_temp_path, relative_path)

    # check if or not to skip this video
    skip_flag = check_exist_video(save_path, xml_set_path, frame_num)

    # path to read original images
    read_path = pt.join(arg_set.data_main_path, relative_path)

    # a dict about this video
    video_info = {'path': relative_path, 'trackid': [], 'frame_idx': dict(), 'object': dict()}

    for frm_idx in range(frame_num):
        str_idx = arg_set.frame_format % frm_idx
        frame_info = frame_xml_parse(pt.join(xml_set_path, str_idx + '.xml'))

        # read a frame
        if not skip_flag:
            img = cv2.imread(pt.join(read_path, str_idx + '.JPEG'), cv2.IMREAD_COLOR)
            img_mean = np.mean(img, (0, 1))
            height, width = img.shape[0], img.shape[1]

        # handle every object in this frame
        for obj in frame_info['object']:
            trackid = obj['trackid']
            # add a new trackid
            if trackid not in video_info['trackid']:
                video_info['trackid'].append(trackid)
                video_info['frame_idx'][trackid] = []
                video_info['object'][trackid] = []
            # add a new object
            video_info['frame_idx'][trackid].append(frm_idx)
            video_info['object'][trackid].append(obj)
            # save cropped images
            if not skip_flag:
                img_z, img_x = crop_obj(img, obj['bndbox'], width, height, img_mean)
                crop_z_idx = str_idx + arg_set.crop_z_format % trackid
                img_z_path = pt.join(save_path, crop_z_idx)
                assert cv2.imwrite(img_z_path, img_z, para_jpg_qual), 'Failed to save image:\n%s\n' % img_z_path
                crop_x_idx = str_idx + arg_set.crop_x_format % trackid
                img_x_path = pt.join(save_path, crop_x_idx)
                assert cv2.imwrite(img_x_path, img_x, para_jpg_qual), 'Failed to save image:\n%s\n' % img_x_path
    # get original width and height
    if skip_flag:
        str_idx = arg_set.frame_format % 0
        img = cv2.imread(pt.join(read_path, str_idx + '.JPEG'), cv2.IMREAD_COLOR)
        height, width = img.shape[0], img.shape[1]
        logging.debug('Skip: %s' % relative_path)
    video_info['height'] = height
    video_info['width'] = width
    return video_info


def count_handled_num(relative_path: str):
    """
    Print the number of handled videos when using parallel execution
    :param relative_path: the path to a video
    :return: None
    """
    with handled_num.get_lock():
        handled_num.value += 1
        logging.debug('%d/%d Video: %s\n' % (handled_num.value, total_num.value, relative_path))
        print('%d/%d videos in crop handling...' % (handled_num.value, total_num.value), end='\r')
        # print('%d/%d videos in crop handling...' % (handled_num.value, total_num.value))


def para_crop_one_video(args):
    """
    parallel wrap for the function crop_one_video
    :param args: See function crop_one_video
    :return: See function crop_one_video
    """
    relative_path, xml_set_path, frame_num = args
    video_info = crop_one_video(relative_path, xml_set_path, frame_num)
    count_handled_num(relative_path)
    return video_info


def total_dataset_info(*data_phase: Iterable[Union[str]], **base_info_list):
    """
    Get all information of each video in the specified data set(s)
    :param data_phase: 'train' and/or 'val'
    :param base_info_list: optional, a dict with video_list, xml_list and frame_num_list
    :return: a dict, see function crop_one_video
    """
    for phase in data_phase:
        assert phase in ('train', 'val'), 'Invalid phase:%s\n' % phase

    if base_info_list:
        v_l = base_info_list['v_l']
        x_l = base_info_list['x_l']
        f_l = base_info_list['f_l']
    else:
        # get videos list and xml files list
        v_l, x_l = video_list.get_video_list(arg_set.data_main_path)

        # get frame numbers list
        f_l = video_list.get_frame_number(v_l, arg_set.data_main_path)

    logging.info('WAITING: Get all information of each video')
    video_set_data = dict()
    for phase in data_phase:
        video_num = len(v_l[phase])
        total_num.value = video_num
        handled_num.value = 0
        multi_arg = [(v_l[phase][i], x_l[phase][i], f_l[phase][i]) for i in range(video_num)]
        # using parallel execution
        with Pool(arg_set.worker_num) as p:
            video_set_data[phase] = p.map(para_crop_one_video, multi_arg, arg_set.chunksize)
        # video_set_data[phase] = [para_crop_one_video(x) for x in multi_arg]  # todo equal to the two lines above
        # delete those videos that contain no or only one object with a certain track-id
        keep_idx = []
        for idx, v in enumerate(video_set_data[phase]):
            keep_flag = True
            if len(v['trackid']) <= 0:
                keep_flag = False
            else:
                for trackid in v['trackid']:
                    if len(v['frame_idx'][trackid]) <= 1:
                        keep_flag = False
                        break
            if keep_flag:
                keep_idx.append(idx)
        video_set_data[phase] = [video_set_data[phase][i] for i in keep_idx]
    logging.info('FINISHED: Get all information of each video')
    return video_set_data


if __name__ == '__main__':
    info = total_dataset_info('val')
    print('Done')
