from UnetModel import *

'''
   main script for parameter search to run from script line. 
'''

class PermutationDict(object):
    '''
    helper class to save and generate all the permutation to be tested
    '''

    def __init__(self):
        self.cutPatch = [True, False]
        self.patch_size = [48, 80]
        self.filterSlices = [True, False]
        self.min_perentage_labeled_voxals = [0.01, 0.05]
        self.percentage_of_labeled_data = [0.7, 0.5]

        self.num_layers = [3, 4]
        self.depth = [32, 64]
        self.weight_val = [10, 15]
        self.weighted_sum = True
        self.isBatchNorm = [True, False]

        self.learning_rate = [0.01, 0.005, 0.001]
        self.batchSize = [32, 64]
        self.isAugmentaion = [True, False]

    def get_per_dict_instance(self):

        '''
        getter for a permutaion of the hyper-params for a single random net run
        '''

        paramsDict = {}

        paramsDict['cutPatch'] = self.cutPatch[self.random_bool()]
        paramsDict['patch_size'] = self.patch_size[self.random_bool()] if paramsDict['cutPatch'] else 240
        paramsDict['filterSlices'] = True if paramsDict['cutPatch'] else False

        num = self.random_bool()
        paramsDict['min_perentage_labeled_voxals'] = self.min_perentage_labeled_voxals[num]
        paramsDict['percentage_of_labeled_data'] = self.percentage_of_labeled_data[num]

        paramsDict['num_layers'] = self.num_layers[self.random_bool()]
        paramsDict['depth'] = self.depth[self.random_bool()]
        paramsDict['weight_val'] = self.weight_val[self.random_bool()]
        paramsDict['weighted_sum'] = False if not(paramsDict['weight_val']) else True
        paramsDict['isBatchNorm'] = self.isBatchNorm[self.random_bool()]

        paramsDict['learning_rate'] = self.learning_rate[self.random_int(1, 3)]
        paramsDict['batchSize'] = self.batchSize[self.random_bool()]
        paramsDict['isAugmentaion'] = True

        return paramsDict

    @staticmethod
    def random_bool():
        randNum = np.random.rand(1)
        return int(randNum < 0.5)

    @staticmethod
    def random_int(lowVal, highVal):
        return int(np.random.randint(lowVal, highVal, 1))




def main_func(number):

    permDict = PermutationDict()

    # init the permutation and create a folder to save logs and files
    createFolder(os.path.realpath(__file__ + "/../"), 'runData')
    runFolderStr = time.strftime('RunFolder_%d_%m_%y__%H_%M_iter_num_{}'.format(number))
    createFolder(os.path.realpath(__file__ + "/../") + "/runData/", runFolderStr)
    runFolderDir = os.path.realpath(__file__ + "/../") + "/runData/" + runFolderStr
    logFolder = runFolderDir
    startLogging(logFolder, False)

    # get permutation of the parameters dict
    paramsDict = permDict.get_per_dict_instance()
    logging.info('###############################################\n')
    logging.info('Parameters search, iteration mum: {}\n'.format(number))
    logging.info('Permutation dict values:')

    # print for permutation dict for debug
    for key, value in paramsDict.items():
        logging.info(str(key) + ' : ' + str(value))
    logging.info('###############################################\n')

    # LOAD DATA
    logging.info('Run mode: logging dir: {}'.format(logFolder))
    dataPipe = DataPipline(numTrain=1,
                           numVal=1,
                           numTest=1,
                           modalityList=[0, 1, 2, 3],
                           permotate=True, # if FALSE - load the manual data lists
                           optionsDict={'zeroPadding': True,
                                        'paddingSize': 240,
                                        'normalize': True,
                                        'normType': 'reg',
                                        'cutPatch': paramsDict['cutPatch'], # Added option not to cut patched - no filter !
                                        'patchSize': paramsDict['patch_size'],
                                        'binaryLabelsC':True,
                                        'filterSlices': paramsDict['filterSlices'],
                                        'minPerentageLabeledVoxals': paramsDict['min_perentage_labeled_voxals'],
                                        'percentageOfLabeledData': paramsDict['percentage_of_labeled_data'],
                                        'isAugment': True}) ##################################################
    # CREATE MODEL
    unetModel = UnetModelClass(layers=paramsDict['num_layers'],
                               num_channels=len(dataPipe.modalityList),
                               num_labels=1,
                               image_size=paramsDict['patch_size'],
                               kernel_size=3,
                               depth=paramsDict['depth'],
                               pool_size=2,
                               costStr='sigmoid',
                               optStr='adam',
                               argsDict={'layersTodisplay':[1],'weightedSum': paramsDict['weighted_sum'],
                                         'weightVal': paramsDict['weight_val'],
                                         'isBatchNorm': paramsDict['isBatchNorm']})

    # TRAIN AND TEST MODEL
    trainModel = Trainer(net=unetModel, argsDict={'printValidation': 3})

    trainModel.train(dataPipe=dataPipe,
                     batchSize=4, ##################################dataPipe.testSamples
                     numSteps=10,
                     printInterval=1,
                     logPath=logFolder,
                     serialNum=number,
                     isRestore=False,
                     restorePath='')

    logging.info('Summery data for permutation number {}:'.format(number))

# run as main
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Missing argument <iteration_number>")
        exit()

    main_func(sys.argv[1])

# Roy: call for tensorboard
# python3 -m tensorboard.main --logdir /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/NetModel/tensorboard

