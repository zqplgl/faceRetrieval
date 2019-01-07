import os
from mtcnn import detect_face
import caffe
import cv2
import math

class IFaceZoneDetect:
    def __init__(self,model_dir,gpu_id=0):
        self.__minsize = 20
        self.__factor = 0.709

        self.__threshold = [0.6, 0.7, 0.7]


        self.__gpu_id = gpu_id
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        model_dir += '/' if not model_dir.endswith('/') else model_dir
        self.__pnet = caffe.Net(model_dir+"/mtcnn/det1.prototxt",model_dir+"/mtcnn/det1.caffemodel",caffe.TEST)
        self.__rnet = caffe.Net(model_dir+"/mtcnn/det2.prototxt",model_dir+"/mtcnn/det2.caffemodel",caffe.TEST)
        self.__onet = caffe.Net(model_dir+"/mtcnn/det3.prototxt",model_dir+"/mtcnn/det3.caffemodel",caffe.TEST)

    @staticmethod
    def align(im,points):
        eye_center = ((points[0][0]+points[1][0])/2.0,(points[0][1]+points[1][1])/2.0)
        dy = points[1][1] - points[0][1]
        dx = points[1][0] - points[0][0]
        angle = math.atan2(dy,dx)*180/math.pi

        rot_mat = cv2.getRotationMatrix2D(eye_center,angle,1)
        im = cv2.warpAffine(im,rot_mat,(im.shape[1],im.shape[0]))

        return im

    @staticmethod
    def get_align_face(self,im,box,points):
        pad_scale = 0.3
        w = box[2] - box[0]
        h = box[3] - box[1]
        pad_w = w * pad_scale
        pad_h = h * pad_scale

        x1 = int(box[0] - pad_w) if (box[0] - pad_w)>0 else 0
        y1 = int(box[1] - pad_h) if (box[1] - pad_h)>0 else 0
        x2 = int(box[2] + pad_w) if (box[2] + pad_w)<im.shape[1] else im.shape[1]
        y2 = int(box[3] + pad_h) if (box[3] +pad_h)<im.shape[0] else im.shape[0]

        im = im[y1:y2,x1:x2]
        points = [[point[0] - x1,point[1] - y1] for point in points]
        im_align = self.align(im,points)
        boxes,points = self.detect(im_align)
        assert (len(boxes)==1)
        im = im_align[boxes[0][1]:boxes[0][3],boxes[0][0]:boxes[0][2]]

        return im


    def detect(self,im):
        img_matlab = im.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        boundingboxes, tmp_points = detect_face(img_matlab, self.__minsize, self.__pnet, self.__rnet, self.__onet, self.__threshold, False, self.__factor)

        boundingboxes = boundingboxes.tolist()

        for box in boundingboxes:
            for i in range(4):
                box[i] = int(box[i])

        points = []
        for point in tmp_points:
            if len(point)!=10:
                continue

            points.append([[point[0],point[5]],[point[1],point[6]],[point[2],point[7]],[point[3],point[8]],[point[4],point[9]]])

        return boundingboxes,points

def drawBoxesAndPoints(im, boxes,points):
    for box in boxes:
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0,255,0), 1)

    for point in points:
        for p in point[4:5]:
            cv2.circle(im,tuple(p),3,(0,255,0),1)
    return im


if __name__=="__main__":
    model_dir = "/home/zqp/install_lib/models"
    detector = IFaceZoneDetect(model_dir,0)


    pic_dir = "/home/zqp/pic/face/"
    for picname in os.listdir(pic_dir):
        if not picname.endswith(".jpg"):
            continue

        im = cv2.imread(pic_dir+picname)

        boxes,points = detector.detect(im)
        assert (len(boxes) == len(points))
        for i in range(len(boxes)):
            im_temp = IFaceZoneDetect.get_align_face(detector,im,boxes[i],points[i])
            cv2.imshow("im",im_temp)
            cv2.waitKey(0)

        drawBoxesAndPoints(im,boxes,points)

        cv2.imshow("im",im)
        cv2.waitKey(0)






