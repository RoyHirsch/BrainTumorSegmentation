'''
    Main script for training and testing Unet model for segmentation of
    brain tumor from MRI scans. The model gets raw multimodal MRI scans
    and outputs prediction of the whole tumor area.

    Contains three main components (classes):
    - DataPipline : a module for loading and pre-precessing of BRATS2012 dataset
    - UnetModelClass : a module for creating the neural net graph
    - Trainer : a module for running the net graph for optimization

    Tester - side module for testing the results.

    The main script can be run in three modes: Train, Test and Restore.
    The main script creates real-time logging.


Created by Roy Hirsch and Ori Chayoot, 2018, BGU
'''

from UnetModel.scripts.Trainer import *
from UnetModel.scripts.Tester import *
from UnetModel.scripts.UnetModelClass import *
from Utilities.DataPipline import *

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

# Make new logging folder only in Train mode
if FLAGS.runMode == 'Train':
    createFolder(os.path.realpath(__file__ + "/../"), 'runData')
    runFolderStr = time.strftime('RunFolder_%H_%M__%d_%m_%y')
    createFolder(os.path.realpath(__file__ + "/../") + "/runData/", runFolderStr)
    runFolderDir = os.path.realpath(__file__ + "/../") + "/runData/" + runFolderStr
    FLAGS.logFolder = runFolderDir

# Use perilously defined folder for Test or Restore run modes
if FLAGS.runMode in ['Test', 'Restore']:
    itemsList = FLAGS.restoreFile.split('/')
    FLAGS.logFolder = '/'.join(itemsList[:-1])

##############################
# LOAD DATA
##############################
startLogging(FLAGS.logFolder, FLAGS.debug)
logging.info('All load and set - let\'s go !')
logging.info('Run mode: {} :: logging dir: {}'.format(FLAGS.runMode, FLAGS.logFolder))
dataPipe = DataPipline(numTrain=6,
                       numVal=1,
                       numTest=1,
                       modalityList=[0, 1, 2, 3],
                       permotate=True,
                       optionsDict={'zeroPadding': True,
                                    'paddingSize': 240,
                                    'normalize': True,
                                    'normType': 'reg',
                                    'cutPatch': True,
                                    'patchSize': 64,
                                    'binaryLabelsC':True,
                                    'filterSlices': True,
                                    'minPerentageLabeledVoxals': 0.05,
                                    'percentageOfLabeledData': 0.5})

##############################
# CREATE MODEL
##############################
unetModel = UnetModelClass(layers=3,
                           num_channels=len(dataPipe.modalityList),
                           num_labels=1,
                           image_size=64,
                           kernel_size=3,
                           depth=32,
                           pool_size=2,
                           costStr='combined',
                           optStr='adam',
                           argsDict={'layersTodisplay':[1],'weightedSum': 'True', 'weightVal': 13, 'isBatchNorm': True})

##############################
# RUN MODEL
##############################
if FLAGS.runMode in 'Train':
    trainModel = Trainer(net=unetModel, argsDict={'printValidation': 50})

    trainModel.train(dataPipe=dataPipe,
                     batchSize=16,
                     numSteps=200,
                     printInterval=20,
                     logPath=FLAGS.logFolder)

elif FLAGS.runMode == 'Test':
    testModel = Tester(net=unetModel, testList=[1], argsDict={'isPatches': True})
    testModel.test(dataPipe=dataPipe, batchSize=64, restorePath=FLAGS.restoreFile)

else:
    logging.info('Error - unknown runMode.')
