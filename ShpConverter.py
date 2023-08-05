import os, cv2
import numpy as np
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
import geopandas


parser = argparse.ArgumentParser(
                prog = 'Milkweed Detector',
                description = 'Detect Milkweeeds',
                epilog = '')
parser.add_argument('--input', required=True, help = 'the Results folder')
parser.add_argument('--label', required=False, help = 'path to labeled images')
parser.add_argument('--output', required=True)
parser.add_argument('--shp', required=True, help= 'shp file that contains the geolocations')


if __name__ == '__main__':
    args = parser.parse_args()
    input_path = args.input
    output_folder = args.output
    shp_path = args.shp
    labeled_img_folder = args.label
    route_locations = geopandas.read_file(shp_path)
    Detection_sheet_list = os.listdir(input_path)
    total_detection = []
    for f in Detection_sheet_list:
        total_detection.append(pd.read_csv(os.path.join(input_path,f)))
    total_detection = pd.concat(total_detection)
    valid_classes = []
    geometries = []
    ratioes = []
    confs = []
    image_names = []
    labeled_image_names = []

    for ind in tqdm(range(len(total_detection))):
        data_info = total_detection.iloc[ind]
        Conf = data_info.Conf
        image_size = cv2.imread(data_info.FileName).shape
        length = data_info.X2 - data_info.X1
        height = data_info.Y2 - data_info.Y1
        ratio = length*height/(image_size[0]*image_size[1])
        info_location = data_info.FileName.split('\\')
        # print(info_location[-1].split('.')[0].split('_'))
        frame_ind = int(info_location[-1].split('.')[0].split('_')[1])
        session = info_location[-4]
        if len(route_locations.loc[(route_locations.SESSION_NA == session)&(route_locations.FRAME == frame_ind)]) == 0:
            continue
        location_index = route_locations.loc[(route_locations.SESSION_NA == session)&(route_locations.FRAME == frame_ind)].index[0]
        labeled_img_name = os.path.abspath(data_info.FileName).replace('\\','_').split(':')[1]
        labeled_img_path = os.path.join(labeled_img_folder,labeled_img_name)
        labeled_image_names.append(labeled_img_path)
        geometries.append(route_locations.iloc[location_index].geometry)
        image_names.append(data_info.FileName)
        valid_classes.append(data_info.Class)
        ratioes.append(ratio)
        confs.append(Conf)
    geoframes = pd.DataFrame({
    'Class':valid_classes,
    'Ratio':ratioes,
    'Conf':confs,
    'Path':image_names,
    'LabelPath':labeled_image_names,
    'geometry':geometries
        })
    gdf = geopandas.GeoDataFrame(geoframes, geometry="geometry")
    gdf.crs = route_locations.crs
    gdf.to_file(os.path.join(output_folder,'result.shp'))