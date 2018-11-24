from UnetModel import *
from skimage.transform import resize
import tensorflow as tf
import time as time
from UnetModel.scripts.utils import *
import numpy as np
class Tester(object):
    def __init__(self, net,testList=[], argsDict={'mod':[1,3]}):
        logging.info('#### -------- Tester object was created -------- ####\n')
        self.net = net
        self.testList = testList
        self.argsDict = argsDict

    def __del__(self):
        # logging.info('#### -------- Tester object was deleted -------- ####\n')
        pass

    def test(self, dataPipe, restorePath='/UnetModel/runData/RunFolder_23_13__21_03_18/unet_3_13_23_19__21_03_18.ckpt'):
        with tf.Session(graph=self.net.graph) as session:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            logging.info('Initialized')

            # restoring data from model file
            logging.info('Loading data from {}'.format((restorePath)))
            saver.restore(session, "{}".format((restorePath)))
            Dicelist = []
            for item in self.testList:
                starttime = time.time()
                batchData, batchLabels,locations = dataPipe.next_image(item)#####to change names
                predictionlist=[]
                for j in range(0,batchData.shape[0]):###need to change the number of batch
                    batchDatatemp = batchData[j:j+1, :, :, :]
                    batchLabelstemp = batchLabels[j:j+1, :]
                    feed_dict = {self.net.X: batchDatatemp, self.net.Y: batchLabelstemp}
                    predictions = session.run(self.net.predictions,feed_dict=feed_dict)
                    predictionlist.append(predictions)
                predictionscheck = np.array(predictionlist)
                batchLabelcheck,imgToview=dataPipe.next_full_label_and_image(item)
                tempimg=prediction_to_image(locations=locations,values=predictionscheck,refimg=batchLabelcheck)
                predictionscheck= np.reshape(tempimg,(-1,dataPipe.optionsDict['paddingSize'],dataPipe.optionsDict['paddingSize'],1))
                batchLabelcheck= np.reshape(batchLabelcheck,(-1, dataPipe.optionsDict['paddingSize'],dataPipe.optionsDict['paddingSize'],1))
                endtime = time.time()
                logging.info('Total example time={}'.format(endtime - starttime))
                print(diceScore_dec(predictionscheck, batchLabelcheck))
                while (True):
                    index = input('\nFor 3d viewer press V\nFor next example press Q:\nFor 3d edge viewer pess E:\n')
                    if index == 'Q':
                        break
                    elif index == 'V':
                        modality = input('Please enter modality to view from the list {}\n'
                                         '0=T1 ,1=T2 ,2=T1g,3=Flair :'.format(dataPipe.modalityList))
                        cmap=input('enter cmap')
                        modview = imgToview[0:imgToview.shape[0], :, :, int(modality)]
                        slidesViewer(modview, predictionscheck[:, :, :, 0], batchLabelcheck[:, :, :, 0],['Image','Prediction','Label'],cmap)
                        plt.show()
                    elif index == 'E':
                        modality = input('Please enter modality to view from the list {}\n'
                                         '0=T1 ,1=T2 ,2=T1g,3=Flair :'.format(dataPipe.modalityList))
                        cmap=input('enter cmap')
                        index2=input('Enter W for whole tumor, Enter T for the tumor tissues')
                        pre=predictionscheck[:, :, :, 0]
                        lab=batchLabelcheck[:, :, :, 0]
                        if index2=='W':
                            pre[pre>0]=1
                            lab[lab>0]=1
                        modview = imgToview[0:imgToview.shape[0], :, :, int(modality)]
                        Edge_Viewer(modview,pre,lab,cmap)
                        plt.show()
                    else:
                        print('Wrong option, please try again:\n')
            logging.info('Mean Dice={}'.format(np.mean(np.array(Dicelist))))


def diceScore_dec(predictions, labels):
    #this function calculate dice for corresponding values in predictions and labels
    eps = 1e-5
    dicelist=[]
    for i in [0,2,4]:
        label = np.copy(labels)
        prediction = np.copy(predictions)
        if i == 0:
            label[label>0]=1
            prediction[prediction>0]=1
        if i == 2:
            label[label == i] = 0
            prediction[prediction == i]=0
            label[label >0] = 1
            prediction[prediction >0] = 1
        if i == 4:
            label[label != i] = 0
            prediction[prediction != i] = 0
            label[label == i] = 1
            prediction[prediction == i] = 1
        dicelist.append(dice_score(prediction,label,eps))
    return dicelist

def accuracy(predictions, labels):
    eq = np.equal(predictions, labels)
    res = np.mean(eq)
    return res
def dice_score(prediction,label,eps):
    intersection = np.sum(np.multiply(prediction, label))
    union = np.sum(prediction) + np.sum(label)
    res = 2 * intersection / (union + eps)
    if np.sum(label) == 0:
        res = -1
    return res