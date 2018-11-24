'''
    Main script for training and testing Vgg16 model for segmentation of
    brain tumor from MRI scans. The model gets patches of MRI scans with
    whole tumor region predictions (output of the first NN).
    The model predicts the tumor's sub-regions.

    Contains three main components (classes):
    - DataPipline : a module for loading and pre-precessing of BRATS2012 dataset
    - UnetModelClass : a module for creating the neural net graph
    - Trainer : a module for running the net graph for optimization

    Tester - side module for testing the results.

    The main script can be run in three modes: Train, Test and Restore.
    The main script creates real-time logging.


Created by Roy Hirsch and Ori Chayoot, 2018, BGU
'''

from Vgg16Model.scripts.Vgg16Model import *
from Vgg16Model.scripts.Tester import *
from Vgg16Model.scripts.Trainer import *
from Utilities.DataPiplineTwo import *

##############################
# CONSTANTS
##############################
flags = tf.app.flags
flags.DEFINE_string('runMode', 'Train',
                    'run mode for the whole sequence: Train, Test or Restore')
flags.DEFINE_bool('debug', False,
                  'logging level - if true debug mode')
tf.app.flags.DEFINE_string('logFolder', '',
                           'logging folder for the sequence, filled automatically')
tf.app.flags.DEFINE_string('restoreFile', '',
                           'path to a .ckpt file for Restore or Test run modes')
FLAGS = flags.FLAGS

# for concatenations of nets
testlist = [3,19,5,20,27,13,4,9,1,25,24]

# Make new logging folder only in Train mode
if FLAGS.runMode == 'Train':
    createFolder(os.path.realpath(__file__ + "/../"), 'runData')
    runFolderStr = time.strftime('RunFolder_%H_%M__%d_%m_%y')
    createFolder(os.path.realpath(__file__ + "/../") + "/runData/", runFolderStr)
    runFolderDir = os.path.realpath(__file__ + "/../") + "/runData/" + runFolderStr
    FLAGS.logFolder = runFolderDir

# Use perilously defined folder for Test or Restore run modes
if FLAGS.runMode in ['Test', 'Restore','Concat']:
    itemsList = FLAGS.restoreFile.split('/')
    FLAGS.logFolder = '/'.join(itemsList[:-1])

##############################
# LOAD DATA
##############################
startLogging(FLAGS.logFolder, FLAGS.debug)
logging.info('All load and set - let\'s go !')
logging.info('Testing the second NN for voxal classification!')
logging.info('Run mode: {} :: logging dir: {}'.format(FLAGS.runMode, FLAGS.logFolder))

dataPipe = DataPipline(trainList=[3,6,29,17,22,10,7,18,2,15,21,26,8,14,12,23,16,0,11],
                       valList=[3],#4,9,1,25,24],
                       testList=[3],#19,5,20,27],
                       modalityList=[0,1,2,3],
                       optionsDict={'zeroPadding': True,
                                    'paddingSize': 240,
                                    'normalize': True,
                                    'normType': 'reg',
                                    'binaryLabelsWT': False,
                                    'binaryLabelsC':False,
                                    'filterSlices': True,
                                    'minParentageLabeledVoxals': 0.1,
                                    'noise':False,'noisePercentage':0.002},
                       patchsize=8,
                       maxpatches=300000,
                       mode=FLAGS.runMode,
                       sparse=3,
                       concatlist=[],
                       predlist=testlist)

##############################
# CREATE MODEL
##############################
netModel = Vgg16Model(num_channels=len(dataPipe.modalityList),
                            num_labels=5,
                            image_size=8,
                            kernel_size=3,
                            depth=32,
                            pool_size=2,
							hiddenSize=64,
                            costStr='softmax',
                            optStr='adam',
                            argsDict={'learningRate':0.0001,'layersTodisplay':[1],'weightedSum': 'True', 'weightVal': 13})

##############################
# RUN MODEL
##############################
if FLAGS.runMode in 'Train':
    trainModel = Trainer(net=netModel, argsDict={})

    trainModel.train(dataPipe=dataPipe,
                     batchSize=96,
                     numSteps=15000,
                     printInterval=500,
                     logPath=FLAGS.logFolder)

elif FLAGS.runMode in ['Test','Concat']:
    testModel = Tester(net=netModel, testList=[3,19,5,20,27,13,4,9,1,25,24], argsDict={})
    testModel.test(dataPipe=dataPipe, restorePath=FLAGS.restoreFile)

else:
    logging.info('Error - unknown runMode.')
