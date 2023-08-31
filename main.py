import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation

from data import cfg, set_cfg, set_dataset
import numpy as np
from threading import Thread

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import logging
import imgmod, calc_crack

# google drive modules
import Google_Photo, datetime


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
image_toshow=None

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/crack_base_1725_176000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input:output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.01, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display  mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
    parser.add_argument('--cropsize', default=448, type=int,
                        help='When in need of cropping images to evaluate in small pieces')
    parser.add_argument('--ignore_masksize', default=400, type=int,
                        help='Ignore masks if pixel numbers are smaller than this.')
    parser.add_argument('--width', default=False, dest='width', action='store_true',
                        help='Calculate average width in pixels and mm. It will be shown on the image')
    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False, width=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)
color_cache = defaultdict(lambda: {})
form_class = uic.loadUiType("Main.ui")[0]

# add logging info(for debugging)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(message)s')
class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent
        self.widget.setReadOnly(True)

    def emit(self, record):
        self.widget.appendPlainText(record.asctime+' > '+record.getMessage())


class WindowClass(QMainWindow, form_class) :
    global image_toshow
    # window class (opens window)
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.path = ''

        logger = logging.getLogger()
        logger.addHandler(QTextEditLogger(self.txt_log))
        logging.info(' %s ' %'log start.')
        now = QDateTime.currentDateTime()
        self.dateTime_end.setDateTime(now)
        self.dateTime_start.setDateTime(now.addSecs(-60))

        # enter signals here

        # -> button
        self.btn_start.clicked.connect(self.buttonStart)

    def buttonStart(self):
        self.btn_start.setEnabled(False)
        savePath = datetime.datetime.now().strftime('output/%Y%m%d%H%M%S')
        t = Thread(target=self.eval_crackwidth,
                   args=('weightsrobo/crackroboflow_res50_2916_910000.pth', None,),
                   kwargs={'imagepath':'cloudImages', 'savepath':savePath,
                           'startDatetime': self.dateTime_start.dateTime().toPyDateTime(), 'endDatetime':self.dateTime_end.dateTime().toPyDateTime()
                           })
        t.start()
        # t.finished.connect(
        #     lambda : self.btn_start.setEnabled(True)
        # )

    def displayImage(self, img):
        h, w, _ = img.shape
        img_np = np.array(img)
        outImage = QImage(img_np.data, w, h, QImage.Format_RGB888)
        # BGR>>RGB
        pixmap=QPixmap(outImage)
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setScaledContents(True)

    def prep_display(self, dets_out, img, h, w, thres=0.1, undo_transform=True, class_color=False, mask_alpha=0.2):
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True

            t = postprocess(dets_out, w, h, visualize_lincomb=False,
                            crop_masks=True,
                            score_threshold=thres)
            """  
            postprocess  Returns 4 torch Tensors (in the following order):
                - classes [num_det]: The class idx for each detection.
                - scores  [num_det]: The confidence score for each detection.
                - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
                - masks   [num_det, h, w]: Full image masks for each detection. 
            """
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:5]

            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < thres:
                num_dets_to_consider = j
                break

        final_mask = np.zeros((h, w, 3), np.uint8)
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        if num_dets_to_consider > 0:
            # masks = masks[:num_dets_to_consider, :, :,None]
            idxmask = 0
            for msk in masks.cpu():
                mask = np.reshape(msk, (448, 448))  # TORCH.448,448
                mask = Image.fromarray(np.uint8(mask * 255))  # pil.448,448
                mask_img = np.array(mask)
                # sum=mask_img[(mask==255)].size
                sum = int(np.sum(np.array(mask_img) >= 200))
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
                if sum < args.ignore_masksize or cfg.dataset.class_names[classes[idxmask]] != "crack":
                    masks = masks.cpu()
                    masks = torch.tensor(np.delete(masks, idxmask, 0), device='cuda')
                    classes = np.delete(classes, idxmask, 0)
                    scores = np.delete(scores, idxmask, 0)
                    num_dets_to_consider -= 1
                    idxmask -= 1
                else:
                    final_mask = cv2.bitwise_or(final_mask, mask_img)
                idxmask += 1
        if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            colors = torch.cat(
                [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],
                dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
            inv_alph_masks = masks * (-mask_alpha) + 1
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)
            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        img_numpy = (img_gpu * 255).byte().cpu().numpy()
        if num_dets_to_consider == 0:
            return img_numpy, final_mask
        if args.display_text or args.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]
                if args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
                if args.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
                    # TODO: changed  text_pt = (x1, y1 -3)  cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    font_face = cv2.FONT_HERSHEY_PLAIN
                    font_scale = 0.6
                    font_thickness = 1
                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (x1, y1 + text_h + 2)
                    text_color = [255, 255, 255]
                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 + text_h + 3), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                cv2.LINE_AA)
        return img_numpy, final_mask

    def evalcropimage(self, net: Yolact, cropsize: int, image, save_path: str):
        t = time.time()
        # crop image
        img_arr, x, y = imgmod.crop_image(image, cropsize)
        out_arr, mask_arr, result_arr = [], [], []
        tmp_arr = []
        data_min, data_max, data_avg=[],[],[]
        # run eval on cropped images
        logging.info(' %s ' % 'cropping...')
        for img in img_arr:
            frame = torch.from_numpy(img).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = net(batch)
            img_numpy, mask = self.prep_display(preds, frame, None, None, thres=0.25, undo_transform=False)
            if not mask.any():
                h, w, _ = img_numpy.shape
                mask = np.zeros((h, w, 3), np.uint8)
                data = [0, 0, 0]
                img_result = img
                mask_result = mask.copy()
            else:
                data, img_result, mask_result = calc_crack.calc_crackwidth(img, mask)
                mask_result = cv2.cvtColor(mask_result, cv2.COLOR_GRAY2RGB)
            tmp_arr.append(mask_result)
            out_arr.append(img_numpy)
            mask_arr.append(mask)
            result_arr.append(img_result)
            if data[0]!=0: data_min.append(data[0])
            if data[1]!=0: data_max.append(data[1])
            if data[2]!=0: data_avg.append(data[2])
            self.displayImage(img_result)

        t2 = time.time()
        t2 = t2 - t
        logging.info(' %s ' % '{}seconds elapsed total......'.format(t2))
        logging.info(' %s ' % 'end crop, concatenating...')

        def concat_tile(im_list_2d):
            return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

        tmp_i = np.ndarray((x, y)).tolist()
        tmp_m = np.ndarray((x, y)).tolist()
        tmp_r = np.ndarray((x, y)).tolist()
        tmp_t = np.ndarray((x, y)).tolist()
        for idx in range(x):
            for idy in range(y):
                tmp_i[idx][idy] = out_arr[idx * y + idy]
                tmp_m[idx][idy] = mask_arr[idx * y + idy]
                tmp_r[idx][idy] = result_arr[idx * y + idy]
                tmp_t[idx][idy] = tmp_arr[idx * y + idy]
        img_tile = concat_tile(tmp_i)
        mask_tile = concat_tile(tmp_m)
        result_tile = concat_tile(tmp_r)
        tmp_tile = concat_tile(tmp_t)
        cv2.imwrite(save_path + '.png', img_tile)
        cv2.imwrite(save_path + '_maskth.png', mask_tile)
        cv2.imwrite(save_path + '_result.png', result_tile)
        cv2.imwrite(save_path + '_maskand.png', tmp_tile)
        logging.info(' %s ' % 'saved... {}'.format(save_path))
        self.txt_crackInfo.clear()
        self.txt_crackInfo.setPlainText("filename : {}\ntime elapsed : {}\nwidth\n- max : {:10.4f}\n- min : {:10.4f}\n- avg : {:10.4f}".format(save_path, t2, max(data_max), min(data_min), np.mean(data_avg)))
        with open(save_path + '.txt', 'w') as f:
            f.write('{:.4f},{:.4f},{:.4f}'.format(max(data_max), min(data_min), np.mean(data_avg)))
        return img_tile, mask_tile

    def evalimages(self, net: Yolact, input_folder: str, output_folder: str):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        for p in Path(input_folder).glob('*'):
            path = str(p)
            name = os.path.basename(path)
            name = '.'.join(name.split('.')[:-1])
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            img = cv2.imread(path)
            if img.shape[1] > img.shape[0]:
                img = cv2.resize(img, dsize=(9248, 6936))
            else:
                img = cv2.resize(img, dsize=(6936, 9248))
            out_path = os.path.join(output_folder, name)
            if img.shape[1] > args.cropsize or img.shape[0] > args.cropsize:
                self.evalcropimage(net, 448, img, out_path)
            print('Done.')

    def evaluate(self, net: Yolact, imagepath: str, savepath: str):
        net.detect.use_fast_nms = args.fast_nms
        net.detect.use_cross_class_nms = args.cross_class_nms
        cfg.mask_proto_debug = args.mask_proto_debug

        print('starting evaluation......')
        self.evalimages(net, imagepath, savepath)
        return

    def eval_crackwidth(self, trained_model, config, imagepath: str = 'cloudImages', savepath: str = 'data/crack_newtest',
                        startDatetime: datetime.datetime = datetime.datetime(2022, 8, 4, 16, 29, 51),
                        endDatetime: datetime.datetime = datetime.datetime(2022, 8, 4, 16, 31, 37)):
        # first download from google cloud
        Google_Photo.downloadfile_fromcloud(startDatetime, endDatetime, imagepath)
        parse_args()

        if not os.path.exists(imagepath):
            logging.info(' %s ' %'directory does not exist...')
            return
        elif len(os.listdir(imagepath))==0:
            logging.info(' %s ' %'No image to speculate...')
            return
        else:
            if config is None:
                # get model info from config file
                model_path = SavePath.from_str(trained_model)
                config = model_path.model_name + '_config'
                logging.info(' %s ' %'Config not specified. Parsed %s from the file name.\n' % args.config)
                set_cfg(config)

            with torch.no_grad():
                if not os.path.exists('results'):
                    os.makedirs('results')

                if args.cuda:
                    cudnn.fastest = True
                    torch.set_default_tensor_type('torch.cuda.FloatTensor')
                else:
                    torch.set_default_tensor_type('torch.FloatTensor')

                logging.info(' %s ' % 'Loading model...')
                net = Yolact()
                net.load_weights(trained_model)
                net.eval()
                logging.info(' %s ' % 'Done.')

                if args.cuda:
                    net = net.cuda()
                self.evaluate(net, imagepath, savepath)
                # enable button again!
                self.btn_start.setEnabled(True)


if __name__ == "__main__" :
    # QApplication : class to start app
    app = QApplication(sys.argv)

    myWindow = WindowClass()
    myWindow.show()

    # makes program to enter application exe loop
    app.exec_()