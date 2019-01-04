#coding=utf-8
import caffe
import numpy as np
import cv2
import os
class ObjTypeClassifier:
    def __init__(self,prototxt,weightfile,meanfile=None,gpu_id=0,meanvalue=None):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.__net = caffe.Net(prototxt,weightfile,caffe.TEST)

        inputBlobShape =  self.__net.blobs['data'].data[0].shape
        self.__input_geometry = (inputBlobShape[-1],inputBlobShape[-2])

        self.__getMean(meanfile,meanvalue)

    def __getMean(self,meanfile=None,meanvalue=None):
        if meanfile:
            data = open(meanfile,'rb').read()
            blob = caffe.proto.caffe_pb2.BlobProto()
            blob.ParseFromString(data)
            mean = np.array(caffe.io.blobproto_to_array(blob))[0]
            mean = mean.mean(1).mean(1)
        else:
            assert(len(meanvalue)==3)
            mean = meanvalue
        self.__mean = np.array([[[mean[0],mean[1],mean[2]]]],dtype=np.float32)

    def __getInputBlob(self,im):
        im_org = im.astype(np.float32,copy=True) - self.__mean
        im_resize = cv2.resize(im_org,self.__input_geometry)
        blob = np.zeros((1,self.__input_geometry[1],self.__input_geometry[0],3),dtype=np.float32)
        blob[0,:,:,:] = im_resize
        channel_swap = (0,3,1,2)
        blob = blob.transpose(channel_swap)
        blob = blob.astype(np.float32)

        return blob

    def extractFeature(self,im,featureBlobName):
        self.__net.blobs['data'].data[...] = self.__getInputBlob(im)
        self.__net.forward()
        feature = self.__net.blobs[featureBlobName].data.flatten()

        return feature

    def classify(self,im,featureBlobName):
        self.__net.blobs['data'].data[...] = self.__getInputBlob(im)
        self.__net.forward()
        prob = self.__net.blobs['prob'].data.flatten()
        index = prob.argsort()[-1]
        confidence = prob[index]
        feature = self.__net.blobs[featureBlobName].data.flatten()

        return (index,confidence,feature)

def run():
    prototxt = r'/home/zqp/install_lib/vehicleDll/models/vehiclePosture/vehiclePostureD3deploy.prototxt'
    weightfile = r'/home/zqp/install_lib/vehicleDll/models/vehiclePosture/vehiclePostureD3model.dat'
    meanfile = r'/home/zqp/install_lib/vehicleDll/models/vehiclePosture/vehiclePostureD3mean.dat'

    classifier = ObjTypeClassifier(prototxt,weightfile,meanfile,0)

    picDir = r'/media/zqp/新加卷/dataSet/data/vehicleHead/carHeadType/奥迪_一汽大众奥迪_奥迪100or红旗小红旗_200x款/'
    for picName in os.listdir(picDir):
        im = cv2.imread(picDir+picName)
        #im = caffe.io.load_image(picDir+picName)
        result = classifier.classify(im,'fc7')
        print (result[0],"****************",result[1])
        if cv2.waitKey(0)==27:
            break


if __name__ == '__main__':
    run()



