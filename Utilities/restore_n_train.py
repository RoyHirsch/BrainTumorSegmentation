from UnetModel import *

def get_params_dict(logDir):

    file = open(logDir, 'r')
    logText = file.read()
    file.close()
    filterText = re.findall('parameters_search : (\w.*)', logText)[2:-2]
    splitedText = [item.split(' : ') for item in filterText]
    dictParams = dict()
    for item in splitedText:
        if item[1] in ['True', 'False']:
            dictParams[str(item[0])] = item[1]
        elif float(item[1]) < 1:
            dictParams[str(item[0])] = float(item[1])
        else:
            dictParams[str(item[0])] = int(item[1])
    return dictParams

def main_func(number):

    logDir = '/Users/royhirsch/Documents/GitHub/runDataFromTheServer/08_05__14_55/bestRes/RunFolder_07_05_18__02_02_iter_num_5 copy/logFile_02_02__07_05_18.log'
    restorePath = '/Users/royhirsch/Documents/GitHub/runDataFromTheServer/08_05__14_55/bestRes/RunFolder_07_05_18__02_02_iter_num_5 copy/validation_save_step_3000.ckpt'

    createFolder(os.path.realpath(__file__ + "/../"), 'runData')
    runFolderStr = time.strftime('RunFolder_restore_%d_%m_%y__%H_%M_iter_num_{}'.format(number))
    createFolder(os.path.realpath(__file__ + "/../") + "/runData/", runFolderStr)
    runFolderDir = os.path.realpath(__file__ + "/../") + "/runData/" + runFolderStr
    logFolder = runFolderDir
    startLogging(logFolder, False)

    # get permutation of the parameters dict
    paramsDict = get_params_dict(logDir)
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
                           permotate=False, # if FALSE - load the manual data lists
                           optionsDict={'zeroPadding': True,
                                        'paddingSize': 240,
                                        'normalize': True,
                                        'normType': 'reg',
                                        'cutPatch': False, # Added option not to cut patched - no filter !
                                        'patchSize': 240,
                                        'binaryLabelsC':True,
                                        'filterSlices': paramsDict['filterSlices'],
                                        'minPerentageLabeledVoxals': paramsDict['min_perentage_labeled_voxals'],
                                        'percentageOfLabeledData': paramsDict['percentage_of_labeled_data']})
    # CREATE MODEL
    unetModel = UnetModelClass(layers=paramsDict['num_layers'],
                               num_channels=len(dataPipe.modalityList),
                               num_labels=1,
                               image_size=240,
                               kernel_size=3,
                               depth=paramsDict['depth'],
                               pool_size=2,
                               costStr='sigmoid',
                               optStr='adam',
                               argsDict={'layersTodisplay':[1],'weightedSum': paramsDict['weighted_sum'],
                                         'weightVal': paramsDict['weight_val'],
                                         'isBatchNorm': paramsDict['isBatchNorm']})

    # TRAIN AND TEST MODEL
    trainModel = Trainer(net=unetModel, argsDict={'printValidation': 10})

    trainModel.train(dataPipe=dataPipe,
                     batchSize=2,
                     numSteps=5,
                     printInterval=1,
                     logPath=logFolder,
                     serialNum=number,
                     isRestore=True,
                     restorePath=restorePath)

    logging.info('Summery data for permutation number {}:'.format(number))

# run as main
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Missing argument <iteration_number>")
        exit()

    main_func(sys.argv[1])

