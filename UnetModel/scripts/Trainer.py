from UnetModel import *
import tensorflow as tf
import time as time


class Trainer(object):
    def __init__(self, net, argsDict):
        logging.info('#### -------- Trainer object was created -------- ####\n')

        self.net = net
        self.argsDict = argsDict

    def __del__(self):
        pass

    def to_string(self, batchSize, numSteps, printInterval):
        logging.info('Trainer object properties:')
        logging.info('batchSize : ' + str(batchSize))
        logging.info('numSteps : ' + str(numSteps))
        logging.info('printInterval : ' + str(printInterval))
        for key, value in self.argsDict.items():
            logging.info(str(key) + ' : ' + str(value))
        logging.info('\n')

    def train(self, dataPipe, batchSize, numSteps, printInterval, logPath):

        self.to_string(batchSize, numSteps, printInterval)

        with tf.Session() as session:

            tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(logPath + '/train', session.graph)
            val_writer = tf.summary.FileWriter(logPath + '/val', session.graph)

            # Save checkPoint
            saver = tf.train.Saver(max_to_keep=6)

            self.numEpoches = len(dataPipe.trainSamples) // batchSize
            startTime = time.time()
            diceValList = []
            dicetrainList = []

            for step in range(numSteps):

                if step % self.numEpoches == 0:
                    logging.info("######## Epoch number {:} ########\n".format(int(step / (len(dataPipe.trainSamples) // batchSize))))
                    dataPipe.initBatchStackCopy()

                batchData, batchLabels = dataPipe.nextBatchFromPermutation(batchSize)
                feed_dict = {self.net.X: batchData, self.net.Y: batchLabels, self.net.isTrain: True}

                _, loss, predictions, summary = session.run(
                    [self.net.optimizer, self.net.loss, self.net.predictions, self.net.merged], feed_dict=feed_dict)

                train_writer.add_summary(summary, step)

                if step % printInterval == 0:
                    epochAccuracy = accuracy(predictions, batchLabels)
                    epochDice = diceScore(predictions, batchLabels)
                    logging.info("++++++ Iteration number {:} ++++++".format(step))
                    logging.info('Minibatch Loss : {:.4f}'.format(loss))
                    logging.info('Training Accuracy : {:.4f}'.format(epochAccuracy))
                    logging.info('Dice score: {:.4f}\n'.format(epochDice))
                    dicetrainList.append(epochDice)

                # print validation data
                if 'printValidation' in self.argsDict.keys() and self.argsDict['printValidation']:
                    if step % self.argsDict['printValidation'] == 0 and step:
                        feed_dict = {self.net.X: dataPipe.valSamples,
                                     self.net.Y: dataPipe.valLabels, self.net.isTrain: False}

                        valPredictions, valLoss, summary = session.run([self.net.predictions, self.net.loss, self.net.merged], feed_dict=feed_dict)
                        val_writer.add_summary(summary, step)

                        accuracyVal = accuracy(valPredictions, dataPipe.valLabels)
                        diceVal = diceScore(valPredictions, dataPipe.valLabels)

                        logging.info("++++++ Validation for step num {:} ++++++".format(step))
                        logging.info('Minibatch Loss : {:.4f}'.format(valLoss))
                        logging.info('Training Accuracy : {:.4f}'.format(accuracyVal))
                        logging.info('Dice score: {:.4f}\n'.format(diceVal))

                        if step % (4 * self.argsDict['printValidation']) == 0:
                            saver.save(session, str(logPath)+'/validation_save_step_{}.ckpt'.format(step), write_meta_graph=False)


            # test statistics
            testBatchSize = 128
            sizeTestArray = np.shape(dataPipe.testSamples)[0]
            testPredictionList = []

            for testBatchInd in range(0, sizeTestArray, testBatchSize):

                feed_dict = {self.net.X: dataPipe.testSamples[testBatchInd:testBatchInd+testBatchSize],
                             self.net.Y: dataPipe.testLabels[testBatchInd:testBatchInd+testBatchSize], self.net.isTrain: False}

                testBatchPredictions = session.run(self.net.predictions, feed_dict=feed_dict)
                testPredictionList.append(testBatchPredictions)

            testPredictions = np.concatenate(testPredictionList, axis=0)
            accuracyTest = accuracy(testPredictions, dataPipe.testLabels)
            diceTest = diceScore(testPredictions, dataPipe.testLabels)

            logging.info("++++++ Test data +++++++++")
            # logging.info('Minibatch Loss : {:.4f}'.format(meanLossTest))
            logging.info('Training Accuracy : {:.4f}'.format(accuracyTest))
            logging.info('Dice score: {:.4f}\n'.format(diceTest))

            save_path = saver.save(session, str(logPath)+"/{}_{}_{}.ckpt".format('final_save_', time.strftime('%H_%M__%d_%m_%y')))

            logging.info('Saving variables in : %s' % save_path)
            with open('model_file.txt', 'a') as file1:
                file1.write(save_path)
                file1.write(' dice={}\n'.format(diceTest))
            endTime = time.time()
            logging.info('Total run time of train is : {0:.4f} min.'.format(round((endTime-startTime)/60, 4)))

            return


def diceScore(predictions, labels):
    eps = 1e-10
    predictions = tf.round(predictions)
    intersection = tf.reduce_sum(tf.multiply(predictions, labels))
    union = eps + tf.reduce_sum(predictions) + tf.reduce_sum(labels)
    res = (2. * intersection) / (union + eps)
    return res.eval()

def accuracy(predictions, labels):
    predictions = tf.round(predictions)
    eq = tf.equal(predictions, labels)
    res = tf.reduce_mean(tf.cast(eq, tf.float32))
    return res.eval()
