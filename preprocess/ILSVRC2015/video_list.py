import os
import os.path as pt
import glob
import logging

phase_video_dir = ('train', 'val', 'test')


def get_video_list(path: str) -> (dict, dict):
    """
    Give every video and its xml file path list
    :param path: path to the whole data set
    :return: two dicts for images and xml files in different phases
    """
    path = pt.normpath(path)
    subdir = ['Annotations', 'Data']
    subdir = [pt.join(path, x) for x in subdir]
    exist_flag = [pt.exists(x) and pt.isdir(x) for x in subdir]
    exist_flag = all(exist_flag)
    if not exist_flag:
        raise OSError('Found no ILSVRC2015 Data in path:\n%s\n' % path)

    temp_video_path = 'Data/VID'
    temp_xml_path = 'VID'
    video_list = dict()
    xml_list = dict()

    for phase in phase_video_dir:
        pvd = pt.join(path, temp_video_path, phase)
        if phase == 'train':
            sub_list = [pt.join(y, z) for y in os.listdir(pvd) for z in os.listdir(pt.join(pvd, y))]
        else:
            sub_list = os.listdir(pt.join(pvd))
        if phase != 'test':
            xml_phase_path = pt.join(subdir[0], temp_xml_path, phase)
            xml_list[phase] = [pt.join(xml_phase_path, w) for w in sub_list]
        relative_path = pt.join(temp_video_path, phase)
        sub_list = [pt.join(relative_path, w) for w in sub_list]
        video_list[phase] = sub_list
    logging.info('FINISHED: Get video and xml file list')
    return video_list, xml_list


def get_frame_number(video_list: dict, path: str) -> dict:
    """
    give the number of frames in each video
    :param video_list: returned by function get_video_list
    :param path: path to the whole data set
    :return: a dict for different phases
    """
    frame_number = dict()
    n = len(phase_video_dir)
    logging.info('WAITING: Get video frame numbers')
    for i, x in enumerate(phase_video_dir):
        sub_list = video_list[x]
        frame_number[x] = [len(glob.glob(pt.join(path, y, '0*.JPEG'))) for y in sub_list]
        print('%d/%d phases in frames counting' % (i, n), end='\r')
    logging.info('FINISHED: Get video frame numbers')
    return frame_number


if __name__ == '__main__':
    from preprocess import arg_set

    v_l, x_l = get_video_list(arg_set.data_main_path)
    f_n = get_frame_number(v_l, arg_set.data_main_path)
    print('First Example:')
    print('Video:', v_l['train'][0])
    print('XML:', x_l['train'][0])
    print('Frames:', f_n['train'][0])
