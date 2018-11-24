from Utilities.loadData import *
from UnetModel import *
import skimage.transform as ski
import os


MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR']
MAX_SAMPLES = 30
MAX_SIZE = 240
ROOT_DIR = os.path.realpath(__file__ + "/../../")


class DataPipline(object):

    batch_offset = 0
    optionsDict = {}

    def __init__(self, trainList, valList, testList, modalityList, optionsDict, patchsize,maxpatches,mode,sparse,concatlist=[],predlist=[]):
        '''

        class object for holding and managing all the data for the net train and testing.

        PARAMS:
            numTrain: number of samples to use in train dataset
            numVal: number of samples to use in val dataset
            numTest: number of samples to use in test dataset
            modalityList: list of numbers to represent the number of channels/modalities to use
            MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR'] represented as: [0,1,2,3]

            optionsDict: additional options dictionary:
               'zeroPadding': bool
               'paddingSize': int
               'normalize': bool
               'normType': ['reg', 'clip']
               'binaryLabels': bool - for flattening the labels into binary classification problem
               'resize': bool
               'newSize': int - new image size for resize
               'filterSlices': bool
               'minParentageLabeledVoxals': int - for filtering slices, parentage in [0,1]
        '''
        logging.info('')
        logging.info('#### -------- DataPipline object was created -------- ####\n')
        self.concatList=concatlist
        self.predList=predlist
        self.sparse = sparse
        self.mode=mode
        self.patchSize = patchsize
        self.maxPatches=maxpatches
        self.trainNumberList = []
        self.valNumberList = []
        self.testNumberList = []
        self.batchesDict = {}
        self.modalityList = modalityList
        self.optionsDict = optionsDict
        self._manual_samples(trainList,valList ,testList )
        if self.mode=='Test' or self.mode=='Concat':
            self.maxPatches=240*240*230
        if optionsDict['noise'] == True:
            self.noise = 1/optionsDict['noisePercentage']
        else:
            self.noise = self.maxPatches
        self.get_samples_list()

    def __del__(self):
        # logging.info('#### -------- DataPipline object was deleted -------- ####\n')
        pass

    def _manual_samples(self, trainList, valList, testList):
        '''
            manualy selects the data samples to each list.
        '''
        self.trainNumberList = trainList
        self.valNumberList = valList
        self.testNumberList = testList

    def _normalize_image_modality(self, imgMod):
        if self.optionsDict['normType'] == 'clip':
            b, t = np.percentile(imgMod, (0.5, 99.5))
            imgMod = np.clip(imgMod, b, t)
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var

        elif self.optionsDict['normType'] == 'reg':
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var

        elif self.optionsDict['normType'] == 'zeroToOne':
            normImg = (imgMod-np.min(imgMod)) / (np.max(imgMod)-np.min(imgMod))

        return normImg

    def _normalize_image(self, img):
        normImg = np.zeros(np.shape(img))
        H, W, D, C = np.shape(img)
        for i in range(0, C):
            normImg[:, :, :, i] = self._normalize_image_modality(img[:, :, :, i])
        return normImg

    def _zero_padding_img(self, maxSize, img):
        [H, W, D, C] = np.shape(img)
        if (H == maxSize) and (W == maxSize):
            return img
        else:
            hOffset = int((maxSize - H) / 2)
            wOffset = int((maxSize - W) / 2)
            paddedImg = np.zeros([maxSize, maxSize, D, C])
            paddedImg[hOffset:H + hOffset, wOffset:W + wOffset, :, :] = img
            return paddedImg

    def _zero_padding_label(self, label):
        [H, W, D] = np.shape(label)
        maxSize = self.optionsDict['paddingSize']

        if (H == maxSize) and (W == maxSize):
            return label
        else:
            hOffset = int((maxSize - H) / 2)
            wOffset = int((maxSize - W) / 2)
            paddedLabel = np.zeros([maxSize, maxSize, D])
            paddedLabel[hOffset:H + hOffset, wOffset:W + wOffset, :] = label
            return paddedLabel

    def _resize_image(self, img):
        newSize = self.optionsDict['newSize']
        H, W, D, C = np.shape(img)
        resizeImg = np.zeros([newSize, newSize, D, C])
        for i in range(D):
            resizeImg[:,:,i,:] = ski.resize(img[:,:,i,:], [newSize,newSize,C], mode='constant')
        return resizeImg

    def _resize_label(self, label):
        newSize = self.optionsDict['newSize']
        H, W, D = np.shape(label)
        resizeLabel = np.zeros([newSize, newSize, D])
        for i in range(D):
            resizeLabel[:, :, i] = ski.resize(label[:, :, i], [newSize, newSize], mode='constant')
        return resizeLabel

        # ---- Prepare Lists ---- #

    def pre_process_list(self, listName = 'train',num=-1):
        '''
            Processing a list of samples (may be train, val or test list)
            This funcrion gets the optionsDist and preforms all the pre-processing on the data.
            THe output is [outSampleArray, outLabelArray] , 4D and 3D arrays containing the pre-processed data.
        '''
        if num != -1:
            numbersList = [num]
            self.optionsDict['filterSlices'] = False
        else:
            numbersList = self.trainNumberList

        outSampleArray = []
        outLabelArray = []
        outLabelLocaton=[] #in use only when testing and displaying the results
        delta = self.patchSize//2
        for i in numbersList:
            img = self.data[i][:, :, :, self.modalityList]
            if self.mode == 'Concat':
                label = self.concatList[self.predList.index(i)]
                label = np.swapaxes(label, 0, 2)
            else:
                label = self.labels[i]

            self.numOfLabels = 5

            if 'zeroPadding' in self.optionsDict.keys() and self.optionsDict['zeroPadding']:
                img = self._zero_padding_img(self.optionsDict['paddingSize'], img)
                label = self._zero_padding_label(label)

            if 'resize' in self.optionsDict.keys() and self.optionsDict['resize']:
                img = self._resize_image(img)
                label = self._resize_label(label)

            if 'normalize' in self.optionsDict.keys() and self.optionsDict['normalize']:
                img = self._normalize_image(img)

            H, W, D, C = np.shape(img)
            s=1
            if self.mode != 'Test' and self.mode != 'Concat' and (self.sparse is not None):
                s=self.sparse
            counter1 = 1
            for j in range(0, D,s):
                for h in range(delta,H-delta,s):
                    for w in range(delta,W-delta,s):
                        if (label[h,w,j] !=0) or ((counter1 % self.noise)==0):
                            counter1+=1
                            patch=img[h-delta:h+delta,w-delta:w+delta,j,:]
                            outSampleArray.append(patch)
                            outLabelArray.append(label[h,w,j])
                            outLabelLocaton.append((j,h,w))
                    if counter1 >self.maxPatches:
                        break
                if counter1 >self.maxPatches:
                    break
            if counter1 > self.maxPatches:
                break

        outSampleArray = np.array(outSampleArray).astype(np.float32)

        # reshape to fit tensorflow constrains
        outLabelArray = np.reshape(outLabelArray, [-1,1]).astype(np.float32)

        return outSampleArray, outLabelArray, outLabelLocaton

    def get_samples_list(self):
        '''
            Main function for data loading.
            Loads all the data from the directory.
            Creates the train, val and test samples lists
        '''

        self.trainStack = []
        self.trainStackLabels = []

        self.data, self.labels = get_data_and_labels_from_folder()
        logging.info('Data and labels were uploaded successfully.')
        self.trainStack, self.trainStackLabels,_ = self.pre_process_list()
        logging.info('Train, val and test database created successfully.')


        # logging.infoings for debug:
        logging.info('Train dataset, samples number: ' + str(self.trainNumberList))
        logging.info('Shape of train dataset: ' + str(np.shape(self.trainStack)))
        logging.info('Shape of train labels: ' + str(np.shape(self.trainStackLabels)))


    # ---- Getters ---- #

    def to_string_pipline(self):
        logging.info('\n\nPipline object properties:\n')
        logging.info('Train dataset, samples number: ' + str(self.trainNumberList) + '\n' +
              'Shape of train dataset: ' + str(np.shape(self.trainStack)) + '\n' +
              'Shape of train label ls: ' + str(np.shape(self.trainStackLabels)))

        logging.info('\nPipline object parameters:\n"')
        logging.info(self.optionsDict)

    def reset_train_batch_offset(self, offset = 0):
        self.batch_train_offset = offset


    def init_batch_number(self):
        self.batchNumer = 0

    def next_train_random_batch(self, batch_size):
        ind = np.random.random_integers(0, np.shape(self.trainStack)[0]-1, batch_size)
        return self.trainStack[ind, :, :, :], self.trainStackLabels[ind, :]


# ---- Help Functions ---- #

    @staticmethod
    def print_img_statistics(img):
        modalities = ['T1', 'T2', 'T1g', 'FLAIR']
        for i in range(0, 4):
            logging.info('Image modality: ' + modalities[i] + ': Mean: ' +
                  str(np.mean(img[:, :, :, i])) + ' Variance: ' + str(np.std(img[:, :, :, i])))
            logging.info('Image max: ' + str(np.max(img)) + ' Image min: ' + str(np.min(img)))

    @staticmethod
    def print_histogram(img):

        counts, bins = np.histogram(img.ravel(), bins=255)
        # plt.bar(bins[1:-1],counts[1:])
        plt.bar(bins[:-1], counts)
        plt.show()

    def next_image(self, imgnumber):
        img,labels,loc = self.pre_process_list(listName='train', num=imgnumber)
        return img,labels,loc

    def next_full_label_and_image(self, imgnumber):
        label = self.labels[imgnumber]
        img = self.data[imgnumber][:, :, :, self.modalityList]
        if 'zeroPadding' in self.optionsDict.keys() and self.optionsDict['zeroPadding']:
            img = self._zero_padding_img(self.optionsDict['paddingSize'], img)
            label = self._zero_padding_label(label)

        if 'resize' in self.optionsDict.keys() and self.optionsDict['resize']:
            img = self._resize_image(img)
            label = self._resize_label(label)

        if 'normalize' in self.optionsDict.keys() and self.optionsDict['normalize']:
            img = self._normalize_image(img)

        H, W, D = np.shape(label)
        outLabelArray=[]
        outSampleArray=[]
        for j in range(0,D):
            outLabelArray.append(label[:, :, j])
            outSampleArray.append(img[:, :, j, :])

        outLabelArray = np.reshape(outLabelArray, [-1, W, W, 1]).astype(np.float32)
        outSampleArray = np.array(outSampleArray).astype(np.float32)


        return outLabelArray,outSampleArray


