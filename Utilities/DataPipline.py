from Utilities.loadData import *
from UnetModel import *
# import skimage.transform as ski
import os


MOD_LIST = ['T1', 'T2', 'T1g', 'FLAIR']
MAX_SAMPLES = 30
MAX_SIZE = 240
ROOT_DIR = os.path.realpath(__file__ + "/../../")


class DataPipline(object):

    batch_offset = 0
    optionsDict = {}

    def __init__(self, numTrain, numVal, numTest, modalityList, permotate, optionsDict):

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
               'minPerentageLabeledVoxals': float - for filtering slices, parentage in [0,1]
               'percentageOfLabeledData': float - the total percentage of labeled data in the pipline, in range [0,1]

               'catPatch' - bool, True to cut the slices into patches
               'patchSize' - int, must be a dividor of image hieght and width

        '''
        logging.info('')
        logging.info('#### -------- DataPipline object was created -------- ####\n')
        self.trainNumberList = []
        self.valNumberList = []
        self.testNumberList = []
        self.batchesDict = {}
        self.modalityList = modalityList
        self.optionsDict = optionsDict
        self.to_string()

        if permotate:
            self._permotate_samples(numTrain, numVal, numTest)
        else:
            self._manual_samples(numTrain, numVal, numTest)
        self.get_samples_list()

    def __del__(self):
        # logging.info('#### -------- DataPipline object was deleted -------- ####\n')
        pass

    def _permotate_samples(self, numTrain, numVal, numTest):
        '''
            randomly selects the data samples to each list.
        '''

        list = np.random.permutation(MAX_SAMPLES).tolist()
        for _ in range(numTrain):
            self.trainNumberList.append(list.pop())
        for _ in range(numVal):
            self.valNumberList.append(list.pop())
        for _ in range(numTest):
            self.testNumberList.append(list.pop())

    def _manual_samples(self, numTrain, numVal, numTest):
        '''
            manualy selects the data samples to each list.
        '''
        self.trainNumberList = [28,  6, 29, 17, 22, 10,  7, 18,  2, 15, 21, 26,  8, 14, 12, 23, 16,  0, 11]
        self.valNumberList = [13,  4,  9,  1, 25, 24]
        self.testNumberList = [3, 19, 5, 20, 27]

    def _normalize_image_modality(self, imgMod):
        if self.optionsDict['normType'] == 'clip':
            b, t = np.percentile(imgMod, (0.5, 99.5))
            imgMod = np.clip(imgMod, b, t)
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var

        # standartization
        elif self.optionsDict['normType'] == 'reg':
            mean = np.mean(imgMod)
            var = np.std(imgMod)
            normImg = (imgMod - mean) / var

        elif self.optionsDict['normType'] == 'zeroToOne':
            normImg = imgMod / np.max(imgMod)

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

    def _crop_iamge(self, img, maxH, maxW):
        H, W, D, C = np.shape(img)
        diffH = round((H - maxH) / 2)
        diffW = round((W - maxW) / 2)
        cropedImg = img[diffH:H - diffH, diffW:W - diffW, :, :]
        return cropedImg

    def trainDataAugmentation(self, img, label):

        H, W, D, C = np.shape(img)
        augmentImg = np.fliplr(img)
        augmentLabel = np.fliplr(label)
        return np.concatenate((img, augmentImg), axis=2), np.concatenate((label, augmentLabel), axis=2)

    # def _resize_image(self, img):
    #     newSize = self.optionsDict['newSize']
    #     H, W, D, C = np.shape(img)
    #     resizeImg = np.zeros([newSize, newSize, D, C])
    #     for i in range(D):
    #         resizeImg[:,:,i,:] = ski.resize(img[:,:,i,:], [newSize,newSize,C], mode='constant')
    #     return resizeImg
    #
    # def _resize_label(self, label):
    #     newSize = self.optionsDict['newSize']
    #     H, W, D = np.shape(label)
    #     resizeLabel = np.zeros([newSize, newSize, D])
    #     for i in range(D):
    #         resizeLabel[:, :, i] = ski.resize(label[:, :, i], [newSize, newSize], mode='constant')
    #     return resizeLabel

        # ---- Prepare Lists ---- #

    def pre_process_list(self, listName='train',num=-1):
        '''
            Processing a list of samples (may be train, val or test list)
            This funcrion gets the optionsDist and preforms all the pre-processing on the data.
            THe output is [outSampleArray, outLabelArray] , 4D and 3D arrays containing the pre-processed data.
        '''
        if num!=-1:
            numbersList=[num]
            self.optionsDict['filterSlices']=False
        elif listName == 'train':
            numbersList = self.trainNumberList
        elif listName in ['val','validation']:
            numbersList = self.valNumberList
        elif listName == 'test':
            numbersList = self.testNumberList
        else:
            logging.info('Error while calling pre_process_list')

        samplesList = []
        labelList = []
        for i in numbersList:
            img = self.data[i][:, :, :, self.modalityList]
            label = self.labels[i]

            if 'binaryLabelsWT' in self.optionsDict.keys() and self.optionsDict['binaryLabelsWT']:
                self.numOfLabels = 1
                tmpLabel = np.zeros_like(label)
                tmpLabel[label != 0] = 1
                label = tmpLabel

            if 'binaryLabelsC' in self.optionsDict.keys() and self.optionsDict['binaryLabelsC']:
                self.numOfLabels = 1
                tmpLabel = np.zeros_like(label)
                tmpLabel[label==1] = 1
                tmpLabel[label==4] = 1
                tmpLabel[label==3] = 1

                label = tmpLabel

            else:
                self.numOfLabels = 4

            if 'zeroPadding' in self.optionsDict.keys() and self.optionsDict['zeroPadding']:
                img = self._zero_padding_img(self.optionsDict['paddingSize'], img)
                label = self._zero_padding_label(label)

            if 'resize' in self.optionsDict.keys() and self.optionsDict['resize']:
                img = self._resize_image(img)
                label = self._resize_label(label)

            if 'normalize' in self.optionsDict.keys() and self.optionsDict['normalize']:
                img = self._normalize_image(img)

            # produce a pipeline of patches
            if 'cutPatch' in self.optionsDict.keys() and self.optionsDict['cutPatch']:
                img, label = self.getPatchesFromSlices(img, label)

            # data augmantetion only the train pipeline
            if listName == 'train' and 'isAugment' in self.optionsDict.keys() and self.optionsDict['isAugment']:
                img, label = self.trainDataAugmentation(img, label)

            # filter only the train pipeline
            if listName == 'train' and 'filterSlices' in self.optionsDict.keys() and self.optionsDict['filterSlices']:
                img, label = self.filterTrainDataPipe(img, label)

            else:
                # swap axes of the data array to match TF format
                img = np.swapaxes(img, 0, 2)
                label = np.swapaxes(label, 0, 2)

            samplesList.append(img)
            labelList.append(label)

        # convert to numpy array and concat all the samples with axis 0 (depth / num of samples)
        for i in range(len(samplesList)):
            if i == 0:
                outSampleArray = np.array(samplesList[i]).astype(np.float32)
                outLabelArray = np.array(labelList[i]).astype(np.float32)
            else:
                outSampleArray = np.concatenate((outSampleArray, samplesList[i]), axis=0)
                outLabelArray = np.concatenate((outLabelArray, labelList[i]), axis=0)

        # reshape to fit tensorflow's constrains
        imageSize = np.shape(outLabelArray)[1]
        outLabelArray = np.reshape(outLabelArray, [-1, imageSize, imageSize, 1]).astype(np.float32)

        # outSampleArray = np.swapaxes(outSampleArray, 0, 2)
        # outLabelArray = np.swapaxes(outLabelArray, 0, 2)

        return outSampleArray, outLabelArray

    def get_samples_list(self):
        '''
            Main function for data loading.
            Loads all the data from the directory.
            Creates the train, val and test samples lists
        '''

        self.trainSamples = []
        self.trainLabels = []
        self.valSamples = []
        self.valLabels = []
        self.testSamples = []
        self.testLabels = []

        self.data, self.labels = get_data_and_labels_from_folder()
        logging.info('Data and labels were uploaded successfully.')
        self.trainSamples, self.trainLabels = self.pre_process_list(listName='train')
        logging.info('Train samples list processed successfully.')
        self.valSamples, self.valLabels = self.pre_process_list(listName='val')
        logging.info('Validation samples list processed successfully.')
        self.testSamples, self.testLabels = self.pre_process_list(listName='test')
        logging.info('Train, val and test database created successfully.')

        # logging for debug:
        logging.info('Train dataset, samples number: ' + str(self.trainNumberList))
        logging.info('Shape of train dataset: ' + str(np.shape(self.trainSamples)))
        logging.info('Val dataset, samples number: ' + str(self.valNumberList))
        logging.info('Shape of val dataset: ' + str(np.shape(self.valSamples)))
        logging.info('Test dataset, samples number: ' + str(self.testNumberList))
        logging.info('Shape of test dataset: ' + str(np.shape(self.testSamples)) + '\n')

    def getPatchesFromSlices(self, image, label):
        H, W, D, C = np.shape(image)
        if 'patchSize' in self.optionsDict.keys() and self.optionsDict['patchSize']:
            patchSize = self.optionsDict['patchSize']
        else:
            logging.info('Error - no patchSize in optionsDict')
        # imageSize is 240 so patchSize should be a dividor of 240
        if not(H / patchSize):
            logging.info('Error - patchSize is not a dividor of imageSize')

        n = np.shape(image)[0] // patchSize
        ind = 0
        patchArrayImage = np.zeros([patchSize, patchSize, D*(n*n), C])
        patchArrayLabel = np.zeros([patchSize, patchSize, D*(n*n)])
        for i in range(n):
            for j in range(n):
                patchArrayImage[:, :, D*(ind):D*(ind+1), :] = image[patchSize*i:patchSize*(i+1), patchSize*j:patchSize*(j+1), :, :]
                patchArrayLabel[:, :, D*(ind):D*(ind+1)] = label[patchSize*i:patchSize*(i+1), patchSize*j:patchSize*(j+1), :]
                ind += 1
        return patchArrayImage, patchArrayLabel

    def filterTrainDataPipe(self, imagesArray, labelsArray):
        # the function filters the pipline to create a pipline with a specific percentage of labels patches\slices
        # three parameters:
        # filterSlices - determine if the pipline needs to be filtered
        # minPerentageLabeledVoxals - determine if a patch\slice will be 'labeled'
        # percentageOfLabeledData - the total percentage of labeled data in the pipline

        H, W, D, C = np.shape(imagesArray)

        indOfLabeledData = []
        indOfNotLabeledData = []
        for j in range(0, D):
            labeledVoxals = np.sum(np.not_equal(labelsArray[:, :, j], 0))
            parcentageLabeledVoxals = labeledVoxals / H * W
            if self.optionsDict['minPerentageLabeledVoxals'] < parcentageLabeledVoxals:
                indOfLabeledData.append(j) # keep the indexes of data that is 'labeled'
            else:
                indOfNotLabeledData.append(j)  # keep the indexes of data that is not 'labeled'

        # find the number of 'not labeled' data to exclude from the pipline in order to get the desire label percentage
        parentageOfLabeledData = self.optionsDict['percentageOfLabeledData']
        totalNumberOfNotLabeledData = (parentageOfLabeledData * len(indOfLabeledData)) / (1 - parentageOfLabeledData)
        indToInclude = indOfNotLabeledData[0:int(totalNumberOfNotLabeledData)]
        indToInclude = indToInclude + indOfLabeledData

        # filter out the right amount of not labeled data
        outSampleArray = []
        outLableArray = []
        # newInd = 0
        for j in range(0, D):
            if j in indToInclude:
                outSampleArray.append(imagesArray[:, :, j, :])
                outLableArray.append(labelsArray[:, :, j])

        return outSampleArray, outLableArray

    # ---- Getters ---- #

    def to_string(self):
        logging.info('Pipline object properties:')
        for key, value in self.optionsDict.items():
            logging.info(str(key) + ' : ' + str(value))
        logging.info('\n')

    def reset_train_batch_offset(self, offset = 0):
        self.batch_train_offset = offset

    def next_train_batch(self, batch_size):
        start = self.batch_offset
        end = start + batch_size
        self.batch_offset = end

        return self.trainSamples[:, :, range(start, end), :], self.trainLabels[:, :, range(start, end)]

    def init_batch_number(self):
        self.batchNumer = 0

    def next_train_random_batch(self, batch_size):
        ind = np.random.random_integers(0, np.shape(self.trainSamples)[0]-1, batch_size)
        return self.trainSamples[ind, :, :, :], self.trainLabels[ind, :, :]

# new batching method:

    # create a random permutation vector in size of train data
    def initBatchStackCopy(self):
        self.batchStack = np.random.permutation(np.shape(self.trainSamples)[0])
        self.batchNumer = 0
        return

    # extracts a randomly selected batch for train
    def nextBatchFromPermutation(self, batch_size):
        batchImage = self.trainSamples[self.batchStack[self.batchNumer: self.batchNumer + batch_size]]
        batchLabel = self.trainLabels[self.batchStack[self.batchNumer: self.batchNumer + batch_size]]
        self.batchNumer += batch_size
        return batchImage, batchLabel

        # ---- Help Functions ---- #
    @staticmethod
    def getSlicesFromPatches(patchArrayImage, patchArrayLabel, imageSize):
        # converts an array of patches into slices
        # input needs to be numpy arrays and not tensorflow abject

        patchArrayImage = np.swapaxes(patchArrayImage, 0, 2)
        patchArrayLabel = np.swapaxes(patchArrayLabel, 0, 2)
        H, W, D, C = np.shape(patchArrayImage)
        n = imageSize // H

        ind = 0
        imageArray = np.zeros([imageSize, imageSize, D // (n * n), C])
        labelArray = np.zeros([imageSize, imageSize, D // (n * n)])
        for i in range(n):
            for j in range(n):
                imageArray[H * i:H * (i + 1), H * j:H * (j + 1), :, :] = patchArrayImage[:, :,
                                                                         D // (n * n) * (ind):D // (n * n) * (ind + 1),
                                                                         :]
                labelArray[H * i:H * (i + 1), H * j:H * (j + 1), :] = patchArrayLabel[:, :,
                                                                      D // (n * n) * (ind):D // (n * n) * (ind + 1)]
                ind += 1
        return imageArray, labelArray

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

    def next_image(self,sliceNumber):
        img,labels=self.pre_process_list(listName='train' ,num=sliceNumber)
        return img, labels

# ---- Test Code ---- #
# Example to run different configurations of Datapipline

# pipObj = DataPipline(numTrain=1, numVal=1, numTest=1, modalityList=[1,2,3], permotate=True,
#                      optionsDict={'zeroPadding': True, 'paddingSize': 240, 'normalize': True,
#                                   'normType': 'reg', 'cutPatch': True, 'patchSize': 60,
#                                   'filterSlices': True, 'minPerentageLabeledVoxals': 0.05, 'percentageOfLabeledData': 0.5})

# pipObj = DataPipline(numTrain=2, numVal=2, numTest=2, modalityList=[1,2,3], permotate=True,
#                      optionsDict={'zeroPadding': True, 'paddingSize': 240, 'normalize': True,
#                                   'normType': 'reg', 'filterSlices': True, 'minPerentageLabeledVoxals': 0.05, 'percentageOfLabeledData': 0.5})

# imageArray, labelArray = DataPipline.getSlicesFromPatches(pipObj.trainSamples, pipObj.trainLabels, 240)
# print(np.shape(imageArray))

# pipObj.to_string_pipline()
# # branch few train batches
# batch_size = 64
# pipObj.init_batch_number()
# for i in range(4):
#      train_batch_data, train_batch_labels = pipObj.next_train_random_batch(batch_size)
#      logging.info(np.shape(train_batch_data))
#      logging.info(np.shape(train_batch_labels))
