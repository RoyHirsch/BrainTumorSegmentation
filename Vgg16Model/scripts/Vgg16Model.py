from UnetModel import *
from UnetModel.scripts.layers import *

class Vgg16Model(object):

    def __init__(self, num_channels, num_labels, image_size,
                 kernel_size, depth, pool_size, hiddenSize, costStr, optStr, argsDict = {}):

        logging.info('#### -------- Vgg16 object was created -------- ####\n')
        self.dispImage=False
        self.layersTodisplay=argsDict['layersTodisplay']
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.depth = depth
        self.pool_size = pool_size
        self.hiddenSize = hiddenSize
        self.costStr = costStr
        self.optStr = optStr
        self.argsDict = argsDict
        self.layersDict = {}
        self.weights_dict = {}
        self.to_string()
        self.logits = self._createNet()
        with self.graph.as_default():
            self.predictions = tf.argmax(self.logits, axis=1)
        self.loss = self._getCost()
        self.optimizer = self._getOptimizer()

    def _createNet(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            # placeholders for training
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, self.num_channels])
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

            # first layer:
            with tf.name_scope('conv_1.1'):
                self.weights_dict['WD1_1'] = weight_variable([self.kernel_size, self.kernel_size, self.num_channels, self.depth])
                self.weights_dict['b1_1'] = bias_variable([self.depth])
                self.layersDict['Conv1_1'] = conv2d(self.X, self.weights_dict['WD1_1'], self.weights_dict['b1_1'], False, False)
            with tf.name_scope('conv_1.2'):
                self.weights_dict['WD1_2'] = weight_variable([self.kernel_size, self.kernel_size, self.depth, self.depth])
                self.weights_dict['b1_2'] = bias_variable([self.depth])
                self.layersDict['Conv1_2'] = conv2d(self.layersDict['Conv1_1'], self.weights_dict['WD1_2'], self.weights_dict['b1_2'], False, False)
            with tf.name_scope('max_pool_1'):
                self.layersDict['Maxpool_1'] = max_pool(self.layersDict['Conv1_2'], self.pool_size)

            # second layer:
            with tf.name_scope('conv_2.1'):
                self.weights_dict['WD2_1'] = weight_variable([self.kernel_size, self.kernel_size, self.depth, self.depth*2])
                self.weights_dict['b2_1'] = bias_variable([self.depth*2])
                self.layersDict['Conv2_1'] = conv2d(self.layersDict['Maxpool_1'], self.weights_dict['WD2_1'], self.weights_dict['b2_1'], False, False)
            with tf.name_scope('conv_2.2'):
                self.weights_dict['WD2_2'] = weight_variable([self.kernel_size, self.kernel_size, self.depth*2, self.depth*2])
                self.weights_dict['b2_2'] = bias_variable([self.depth*2])
                self.layersDict['Conv2_2'] = conv2d(self.layersDict['Conv2_1'], self.weights_dict['WD2_2'], self.weights_dict['b2_2'], False, False)
            with tf.name_scope('max_pool_2'):
                self.layersDict['Maxpool_2'] = max_pool(self.layersDict['Conv2_2'], self.pool_size)

            # third layer:
            with tf.name_scope('conv_3.1'):
                self.weights_dict['WD3_1'] = weight_variable([self.kernel_size, self.kernel_size, self.depth*2, self.depth * 4])
                self.weights_dict['b3_1'] = bias_variable([self.depth * 4])
                self.layersDict['Conv3_1'] = conv2d(self.layersDict['Maxpool_2'], self.weights_dict['WD3_1'], self.weights_dict['b3_1'], False, False)
            with tf.name_scope('conv_3.2'):
                self.weights_dict['WD3_2'] = weight_variable([self.kernel_size, self.kernel_size, self.depth*4, self.depth * 4])
                self.weights_dict['b3_2'] = bias_variable([self.depth * 4])
                self.layersDict['Conv3_2'] = conv2d(self.layersDict['Conv3_1'], self.weights_dict['WD3_2'], self.weights_dict['b3_2'], False, False)
            with tf.name_scope('conv_3.3'):
                self.weights_dict['WD3_3'] = weight_variable([self.kernel_size, self.kernel_size, self.depth*4, self.depth * 4])
                self.weights_dict['b3_3'] = bias_variable([self.depth * 4])
                self.layersDict['Conv3_3'] = conv2d(self.layersDict['Conv3_2'], self.weights_dict['WD3_3'], self.weights_dict['b3_3'], False, False)
            with tf.name_scope('max_pool3'):
                self.layersDict['Maxpool_3'] = max_pool(self.layersDict['Conv3_3'], self.pool_size)

            # first FC
            with tf.name_scope('FC1'):
                self.layersDict['Flat_Maxpool_3'] = tf.reshape(self.layersDict['Maxpool_3'], [-1, self.image_size * self.image_size * self.depth // 16])
                self.weights_dict['FC_WD1_1'] = weight_variable([self.image_size * self.image_size * self.depth // 16, self.hiddenSize])
                self.weights_dict['FC_b1_1'] = bias_variable([self.hiddenSize])
                self.layersDict['FC_1'] = tf.nn.relu(tf.matmul(self.layersDict['Flat_Maxpool_3'], self.weights_dict['FC_WD1_1']) + self.weights_dict['FC_b1_1'])

            # second FC
            with tf.name_scope('FC2'):
                self.weights_dict['FC_WD2_1'] = weight_variable([self.hiddenSize, self.hiddenSize])
                self.weights_dict['FC_b2_1'] = bias_variable([self.hiddenSize])
                self.layersDict['FC_2'] = tf.nn.relu(tf.matmul(self.layersDict['FC_1'], self.weights_dict['FC_WD2_1']) + self.weights_dict['FC_b2_1'])

            # third FC
            with tf.name_scope('FC3'):
                self.weights_dict['FC_WD3_1'] = weight_variable([self.hiddenSize, self.num_labels])
                self.weights_dict['FC_b3_1'] = bias_variable([self.num_labels])
                self.layersDict['FC_3'] = tf.nn.relu(tf.matmul(self.layersDict['FC_2'], self.weights_dict['FC_WD3_1']) + self.weights_dict['FC_b3_1'])

            return self.layersDict['FC_3']

    def to_string(self):
        logging.info('UnetModel object properties:')
        logging.info('num_channels : ' + str(self.num_channels))
        logging.info('num_labels : ' + str(self.num_labels))
        logging.info('image_size : ' + str(self.image_size))
        logging.info('depth : ' + str(self.depth))
        logging.info('pool_size : ' + str(self.pool_size))
        logging.info('costStr : ' + str(self.costStr))
        logging.info('optStr : ' + str(self.optStr))
        for key, value in self.argsDict.items():
            logging.info(str(key) + ' : ' + str(value))
        logging.info('\n')

    def _getCost(self):

        with self.graph.as_default():
            if self.costStr == "softmax":
                oneHotLabel = tf.one_hot(indices=tf.cast(self.Y, tf.int32),depth= self.num_labels)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=oneHotLabel))
            elif self.costStr == "sigmoid":
                if 'weightedSum' in self.argsDict.keys() and self.argsDict['weightedSum']:
                    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.Y, logits=self.logits ,pos_weight=self.argsDict['weightVal']))
                else:
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logits))
            else:
                logging.info ("Error : Not defined cost function.")
            loss_tensorboard=tf.summary.scalar('Train_loss', loss)
            self.merged_loss=tf.summary.merge([loss_tensorboard])
        return loss

    def _getOptimizer(self):
        learningRate = self.argsDict.pop('learningRate', 0.01)
        with self.graph.as_default():
            if self.optStr == 'adam':
                    optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss)

            elif self.optStr == 'momentum':
                    momentum = self.argsDict.pop("momentum", 0.2)
                    optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=momentum).minimize(self.loss)
            else:
                logging.info ("Error : Not defined optimizer.")
            return optimizer

    def getLogits(self):
        return self.logits