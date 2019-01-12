import cv2
import os
from objTypeClassifier import ObjTypeClassifier
import numpy as np
from annoy import AnnoyIndex

class FaceRetrieval:
    def __init__(self, model_dir, gpu_id=0):
        deploy_file = os.path.join(model_dir,"faceretrieval/resnet50_ft.prototxt")
        assert os.path.exists(deploy_file)

        weights_file = os.path.join(model_dir,"faceretrieval/resnet50_ft.caffemodel")
        assert os.path.exists(weights_file)

        xml_file = os.path.join(model_dir,"faceretrieval/config95-V.xml")
        assert os.path.exists(xml_file)

        fs = cv2.FileStorage(xml_file,cv2.FILE_STORAGE_READ)
        self.__pca_mean = fs.getNode("mean").mat()
        self.__pca_eigen_vector = fs.getNode("eigen_vector").mat()
        self.__feature_len = self.__pca_eigen_vector.shape[0]

        mean_value = [91.4953,103.8827, 131.0912]
        self.__extracter = ObjTypeClassifier(deploy_file,weights_file,meanvalue=mean_value)

    def create_retrieval_model(self, retrieval_model_path):
        assert (os.path.exists(retrieval_model_path))
        retrieval_model = AnnoyIndex(self.__feature_len)
        retrieval_model.load(retrieval_model_path)

        return retrieval_model

    def extractFeature(self,im):
        try:
            feature = self.__extracter.extractFeature(im, "classifier")
            im_1 = np.flip(im, axis=1)
            feature_1 = self.__extracter.extractFeature(im_1, "classifier")

            result = np.hstack([feature, feature_1])
            result = result.reshape(1,-1)

            res = cv2.PCAProject(result,self.__pca_mean,self.__pca_eigen_vector)
            return res.reshape(-1).tolist()
        except:
            return []

    def buildRetrievalDatabase(self,features,retrieval_model_path):
        self.__database = AnnoyIndex(self.__feature_len)
        index = 0
        for feature in features:
            self.__database.add_item(index,feature)
            index += 1

        self.__database.build(2*self.__feature_len)
        self.__database.save(retrieval_model_path)

    def query(self,im, k, retrieval_model):
        feature = self.extractFeature(im)
        res = retrieval_model.get_nns_by_vector(feature,k,include_distances=False)
        return  res

def buildRetrievalDatabase():
    model_dir = "/home/zqp/install_lib/models"
    net = FaceRetrieval(model_dir)

    features = []
    index = 0
    pic_dir = "/home/zqp/github/DllFaceDeepFeature/testimage"
    num = len(os.listdir(pic_dir))
    for pic_name in os.listdir(pic_dir):
        im = cv2.imread(os.path.join(pic_dir,pic_name))
        res = net.extractFeature(im)
        if len(res):
            features.append(res)
        else:
            print os.path.join(pic_dir,pic_name)+"************* is not exist"

        index += 1
        print "processed***************%s/%s"%(index,num)

    net.buildRetrievalDatabase(features,"./123.tree")

def query():
    model_dir = "/home/zqp/install_lib/models"
    retireval_model_path="./123.tree"
    net = FaceRetrieval(model_dir)
    retrieval_model = net.create_retrieval_model(retrieval_model_path)

    pic_dir = "/home/zqp/github/DllFaceDeepFeature/testimage"
    for pic_name in os.listdir(pic_dir):
        im = cv2.imread(os.path.join(pic_dir,pic_name))

        res = net.query(im,5,retrieval_model_path)
        print res


if __name__=="__main__":
    # buildRetrievalDatabase()
    query()

