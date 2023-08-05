import os, requests, torch, math, cv2
import numpy as np
import PIL
from PIL import Image
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional
import pandas as pd
import argparse

def make_divisible( x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32, floor=0):

    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

device:str = "gpu"#@param ["gpu", "cpu"]
half:bool = False #@param {type:"boolean"}
cuda = device != 'cpu' and torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

parser = argparse.ArgumentParser(
                    prog = 'Milkweed Detector',
                    description = 'Detect Milkweeeds',
                    epilog = '')
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--image',required=False)

if __name__ == '__main__':
    args = parser.parse_args()
    image_path = args.input
    output_folder = args.output
    checkpoint = args.model
    # if_image = args.image
    image_path_list = []
    for root, dirs, files in os.walk(image_path, topdown=False):
        for name in files:
            if (name.split('.')[-1] == 'jpg') or (name.split('.')[-1] == 'png'):
                image_path_list.append(os.path.join(root, name))
    print('Processing:',len(image_path_list),'images')
    post_image_folder = os.path.join(output_folder,'LabeledImage')
    print(post_image_folder)
    pred_res_folder = os.path.join(output_folder,'Results')
    if not os.path.exists(post_image_folder):
        os.mkdir(post_image_folder)
    if not os.path.exists(pred_res_folder):
        os.mkdir(pred_res_folder)
    pred_res_list = os.listdir(pred_res_folder)
    cur_progress = 0
    if len(pred_res_list)>0:
        file_progress = [int(f.split('.')[0]) for f in pred_res_list]
        cur_progress = np.max(file_progress)
    model = DetectBackend(checkpoint, device=device)
    stride = model.stride
    class_names = load_yaml("./data/dataset.yaml")['names']
    img_size:int = 1280#@param {type:"integer"}
    #@title Run YOLOv6 on an image from a URL. { run: "auto" }
    hide_labels: bool = False #@param {type:"boolean"}
    hide_conf: bool = False #@param {type:"boolean"}

    conf_thres: float =.50 #@param {type:"number"}
    iou_thres: float =.45 #@param {type:"number"}
    max_det:int =  1000#@param {type:"integer"}
    agnostic_nms: bool = False #@param {type:"boolean"}

    img_size = check_img_size(img_size, s=stride)

    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False

    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

    infos = []
    names = []
    for i in tqdm(range(cur_progress+1,len(image_path_list))):
        image_src = cv2.imread(image_path_list[i])
        if image_src is None:
            continue
        image_ori = image_src.copy()
        image = letterbox(image_src, img_size, stride=stride)[0]
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0
        image = image.to(device)
        if len(image.shape) == 3:
            image = image[None]
        pred_results = model(image)
        classes:Optional[List[int]] = None # the classes to keep
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        gn = torch.tensor(image_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det)>0:
            det[:, :4] = Inferer.rescale(image.shape[2:], det[:, :4], image_src.shape).round()
            info = det[:,:].cpu().detach().numpy()
            # image_name = image_path_list[i].split('\\')[-1]
            imgname = np.array(len(info) * [os.path.abspath(image_path_list[i])])
            image_name = os.path.abspath(image_path_list[i]).replace('\\','_')
            out_image_path = os.path.join(post_image_folder,image_name)
            infos.append(info)
            names.append(imgname)
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                Inferer.plot_box_and_label(image_ori, max(round(sum(image_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
            cv2.imwrite(out_image_path,image_ori)
        if (i%2000 == 0) & (len(infos)>0):
            infos = np.concatenate(infos,axis = 0)
            names = np.concatenate(names).reshape(-1,1)
            infos = np.concatenate([infos,names],axis = 1)
            pd.DataFrame(infos,columns=['X1','Y1','X2','Y2','Conf','Class','FileName']).to_csv(os.path.join(pred_res_folder,'{}.csv'.format(i)),index = False)
            infos = []
            names = []
    infos = np.concatenate(infos,axis = 0)
    names = np.concatenate(names).reshape(-1,1)
    infos = np.concatenate([infos,names],axis = 1)
    pd.DataFrame(infos,columns=['X1','Y1','X2','Y2','Conf','Class','FileName']).to_csv(os.path.join(pred_res_folder,'{}.csv'.format(i)),index = False)

        
