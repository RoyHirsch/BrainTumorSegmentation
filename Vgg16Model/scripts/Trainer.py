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

    def train(self, dataPipe, batchSize, numSteps, printInterval, logPath):

        with tf.Session(graph=self.net.graph) as session:
            tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(logPath, session.graph)
            saver = tf.train.Saver()
            logging.info('Session begun\n')

            for step in range(numSteps):
                batchData, batchLabels = dataPipe.next_train_random_batch(batchSize)
                feed_dict = {self.net.X: batchData, self.net.Y: batchLabels}
                _, loss, predictions, summary = session.run(
                    [self.net.optimizer, self.net.loss, self.net.predictions, self.net.merged_loss],
                    feed_dict=feed_dict)
                if step % printInterval == 0:
                    train_writer.add_summary(summary, step)
                    epochAccuracy = accuracy_tissues(predictions, batchLabels)
                    logging.info("++++++ Iteration number {:} ++++++".format(step))
                    logging.info('Minibatch Loss : {:.4f}'.format(loss))
                    logging.info('Training Accuracy : {:.4f}'.format(epochAccuracy))

            save_path = saver.save(session, str(logPath)+"/{}_{}_{}.ckpt".format('VGG16', self.net.argsDict['weightVal'], time.strftime('%H_%M__%d_%m_%y')))
            logging.info('Saving variables in : %s' % save_path)
            with open('model_file.txt', 'a') as file1:
                file1.write(save_path)

def accuracy_tissues(predictions, labels):
    labels=np.reshape(labels,(1,-1))
    predictions = np.round(predictions)
    eq = np.equal(predictions, labels)
    res = np.mean(eq)
    return res




