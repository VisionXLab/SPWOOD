"""
-------------
This is the multi-process version
"""
import codecs
import copy
import math
import os
from functools import partial
from multiprocessing import Pool
import argparse
import cv2
import dota_utils as util
import numpy as np
import shapely.geometry as shgeo
from dota_utils import GetFileFromThisRootDir
from tqdm import tqdm
import random

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate) ** 2)
                          for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def split_single_warp(name, split_base, rate, extent):
    split_base.SplitSingle(name, rate, extent)


class splitbase():
    def __init__(self,
                 imagepath,
                 labelpath,
                 outimagepath,
                 outlabelpath,
                 percent,
                 code='utf-8',
                 gap=200,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext='.png',
                 padding=True,
                 num_process=8,
                 save_images=True
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.outimagepath = outimagepath
        self.outlabelpath = outlabelpath
        
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.percent = percent
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.pool = Pool(num_process)
        self.save_images = save_images

        if not os.path.isdir(self.outimagepath):
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)

    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly) / 2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(
            img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]), (poly[(
                                                                                   i + 1) * 2], poly[(i + 1) * 2 + 1]))
                     for i in range(int(len(poly) / 2 - 1))]
        distances.append(cal_line_length(
            (poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            if (count == pos):
                outpoly.append(
                    (poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
                outpoly.append(
                    (poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) / 2)
                count = count + 1
            elif (count == (pos + 1) % 5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                        (obj['poly'][2], obj['poly'][3]),
                                        (obj['poly'][4], obj['poly'][5]),
                                        (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                # print('writing...')
                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + \
                              obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                elif (half_iou > 0):
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(
                            out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + \
                                  obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        # if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
        if self.save_images:
            self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)

        
        #删减
        objects_sub = []
        for i in wordname_15:
            # print(i)
            clslst = []
            for obj in objects:
                if (obj['name'] == i):
                    clslst.append(obj)
            # print(len(clslst))
            randnum = len(clslst) * self.percent
            if randnum < 1 and randnum != 0:
                randnum = 1
            samples = random.sample(clslst, int(randnum))
            objects_sub = objects_sub + samples
        objects=objects_sub

        for obj in objects:
            obj['poly'] = list(map(lambda x: rate * x, obj['poly']))
            # obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

        if (rate != 1):
            resizeimg = cv2.resize(
                img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + '1024' + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                self.savepatches(resizeimg, objects,
                                 subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [util.custombasename(x) for x in imagelist if (
                util.custombasename(x) != 'Thumbs')]
        
        try:
            if self.num_process == 1:
                for name in tqdm(imagenames, desc='Processing images'):
                    self.SplitSingle(name, rate, self.ext)
            else:
                worker = partial(split_single_warp, split_base=self,
                                rate=rate, extent=self.ext)
                chunksize = max(1, len(imagenames) // (self.num_process * 4))
                
                # 添加进程池管理和异常处理
                with tqdm(total=len(imagenames), desc='Processing images') as pbar:
                    try:
                        for _ in self.pool.imap_unordered(worker, imagenames, chunksize=chunksize):
                            pbar.update(1)
                    finally:
                        # 确保资源释放
                        self.pool.close()
                        self.pool.join()
                        
        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            # 双重保险确保进程池关闭
            if hasattr(self, 'pool'):
                self.pool.close()
                self.pool.join()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DOTA dataset splitting parameters')
    
    # 第三步
    # 使用RSST代码进行sparse和分割操作
    # 1. 生成semi数据. 将上一步分来的ori的label和unlabel进行切割. label的percent设为1, unlabel的percent设为0. 结果放入到semi中(这是为半监督准备的训练数据)
    # 2. 生成semi sparse数据. 对上一步ori的label部分,采用0.01 0.02 0.05 0.10进行sparse的分割, 结果放入到label_image和label_annotation里面
    
    # Required arguments
    parser.add_argument('--imagepath', type=str, default='/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss2/10/trainval_ori/label_image',
                       help='imagepath for DOTA data')
    parser.add_argument('--labelpath', type=str, default='/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss2/10/trainval_ori/label_annotation',
                       help='labelpath for DOTA data')
    parser.add_argument('--outimagepath', type=str, default='/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss2/10/semisparse/10/label_image',
                       help='outimagepath for DOTA data')
    parser.add_argument('--outlabelpath', type=str, default='/inspire/hdd/project/wuliqifa/gaoyubing-240108110053/zw/adata/trainval_ss2/10/semisparse/10/label_annotation',
                       help='outlabelpath for DOTA data')    
    
    parser.add_argument('--percent', type=float, default=0.10,
                       help='Percentage of objects to keep (default: 0.01)')
    
    # Processing parameters
    parser.add_argument('--gap', type=int, default=200,
                       help='Overlap between patches (default: 200)')
    parser.add_argument('--subsize', type=int, default=1024,
                       help='Subsize of patch (default: 1024)')
    parser.add_argument('--thresh', type=float, default=0.7,
                       help='IOU threshold for keeping instances (default: 0.7)')
    parser.add_argument('--rate', type=float, default=1.0,
                       help='Resize rate before splitting (default: 1.0)')
    parser.add_argument('--num_process', type=int, default=8,
                       help='Number of parallel processes (default: 8)')
    parser.add_argument('--ext', type=str, default='.png',
                       help='Image file extension (default: .png)')
    
    # Boolean flags
    parser.add_argument('--choosebestpoint', action='store_true',
                       help='Choose optimal point order for polygons')
    parser.add_argument('--padding', action='store', type=bool, default=True,
                       help='Pad images to maintain size')
    parser.add_argument('--save_images', action='store', type=bool, default=True,
                       help='是否保存裁剪后的图像块 (default: False)')
    
    args = parser.parse_args()

    # Update wordname_15 with percentage if needed
    wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                   'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                   'basketball-court', 'storage-tank', 'soccer-ball-field', 
                   'roundabout', 'harbor', 'swimming-pool', 'helicopter']

    split = splitbase(
        imagepath=args.imagepath,
        labelpath=args.labelpath,
        outimagepath=args.outimagepath,
        outlabelpath=args.outlabelpath,
        
        percent = args.percent,
        gap=args.gap,
        subsize=args.subsize,
        thresh=args.thresh,
        choosebestpoint=args.choosebestpoint,
        ext=args.ext,
        padding=args.padding,
        num_process=args.num_process,
        save_images = args.save_images
    )
    
    # Modify the objects_sub selection to use args.percent
    # (You'll need to update the SplitSingle method to accept percent as parameter)
    split.splitdata(args.rate)