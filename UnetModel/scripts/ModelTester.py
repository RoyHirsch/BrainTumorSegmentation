import matplotlib.pyplot as plt

class ModelTester(object):

	'''
		Helper class for model testing.

		Background about tf.train.Saver() model:

		Saving will produce three files:
		mymodel.data-00000-of-00001
		mymodel.index
		mymodel.mata

		.data file is the file that contains our training variables and we shall go after it.
		.mata file contains the graph data structure and can be uploaded directly

		The ModelTester class has two 'modes' - with meta file and without it.
		(the model is alternating automatically between the modes)

		Without meta file:
		The object will restore the graph out of the log prints and will reload the params from
		the lastest checkpoint file.

		Whit meta file:
		The net's graph will be built automatically by the tf.train.Saver() model and wil load the latest
		checkpoint file.

		INPUT:
			runFolderDir - a path to the relevant runData folder contains the saved files.

		METHODS:
			predict_test_data - generate predictions of the test data
			view_predictions_results - to view the predictions, must be after 'predict_test_data'
			export_data_to_second_net - special method to export train, val and test data as an iput to
			the second NN (classification).


	'''

	def __init__(self, runFolderDir):

		logging.info('#### -------- Trainer object was created -------- ####\n')
		self.runFolderDir = runFolderDir
		self.checkPointFile = self._get_check_point_file()
		self.metaFile = self._get_meta_file()

		if len(self.metaFile) == 0:
			self.mode = 'noMeta'
			self.paramsDict = self._get_params_dict()
			self.restoredNet, self.dataPipe = self._restore_model_from_log_file()

		else:
			self.mode = 'meta'
			self._restore_model_from_meta()

	def _get_check_point_file(self):
		checkPointFileList = []
		for root, dirs, files in os.walk(self.runFolderDir):
			for fileName in files:
				match = re.search(r'.ckpt.data', fileName)
				if match:
					checkPointFileList.append(fileName)

		if len(checkPointFileList) == 1:
			prefix = checkPointFileList[0].split('.data')[0]
			return os.path.join(self.runFolderDir, prefix)
		else:
			prefix = checkPointFileList[0].split('.data')[-1]
			return os.path.join(self.runFolderDir, prefix)

	def _get_meta_file(self):
		metaFile = []
		for root, dirs, files in os.walk(self.runFolderDir):
			for fileName in files:
				match = re.search(r'.meta', fileName)
				if match:
					metaFile = os.path.join(self.runFolderDir, fileName)
		return metaFile

	def _get_params_dict(self):

		for root, dirs, files in os.walk(self.runFolderDir):
			for fileName in files:
				match = re.search(r'logFile_', fileName)
				if match:
					logName = fileName

		file = open(os.path.join(self.runFolderDir, logName), 'r')
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

	def _restore_model_from_log_file(self):

		dataPipe = DataPipline(numTrain=1,
							   numVal=1,
		                       numTest=1,
		                       modalityList=[0, 1, 2, 3],
		                       permotate=False,
		                       optionsDict={'zeroPadding': True,
		                                    'paddingSize': 240,
		                                    'normalize': True,
		                                    'normType': 'reg',
		                                    'cutPatch': self.paramsDict['cutPatch'],
		                                    'patchSize': self.paramsDict['patch_size'],
		                                    'binaryLabelsWT': True,
		                                    'filterSlices': self.paramsDict['filterSlices'],
		                                    'minPerentageLabeledVoxals': self.paramsDict['min_perentage_labeled_voxals'],
		                                    'percentageOfLabeledData': self.paramsDict['percentage_of_labeled_data']})
		print('Loaded the pipeline.')
		# CREATE MODEL
		unetModel = UnetModelClass(layers=self.paramsDict['num_layers'],
		                           num_channels=len(dataPipe.modalityList),
		                           num_labels=1,
		                           image_size=self.paramsDict['patch_size'],
		                           kernel_size=3,
		                           depth=self.paramsDict['depth'],
		                           pool_size=2,
		                           costStr='sigmoid',
		                           optStr='adam',
		                           argsDict={'layersTodisplay': [1], 'weightedSum': self.paramsDict['weighted_sum'],
		                                     'weightVal': self.paramsDict['weight_val'],
		                                     'isBatchNorm': self.paramsDict['isBatchNorm']})
		print('Loaded the net model.')
		return unetModel, dataPipe

	def _restore_model_from_meta(self):

		# Restore the graph from the latest checkpoint
		sess = tf.Session()
		importGraph = tf.train.import_meta_graph(self.metaFile)
		importGraph.restore(sess, tf.train.latest_checkpoint('./'))

		# Get the placeholders for the net's input
		graph = tf.get_default_graph()
		self.X = graph.get_tensor_by_name("X:0")
		self.Y = graph.get_tensor_by_name("Y:0")
		self.isTrain = graph.get_tensor_by_name("isTrain:0")

		self.predictionsOp = graph.get_tensor_by_name("predictions:0")
		return

	def predict_test_data(self):
		if self.mode == 'noMeta':
			with tf.Session() as sess:
				tf.global_variables_initializer().run()
				tf.train.Saver().restore(sess, self.checkPointFile)
				print('Restore checkpoint file from {}.'.format(self.checkPointFile))

				batchSize = 64
				diceList = []
				accList = []
				predictionImagesList = []
				self.imagesList = []
				self.labelList = []
				startTime = time.time()

				predictList = np.concatenate((self.dataPipe.testNumberList ,self.dataPipe.valNumberList))
				print('Start predictions for test data.')
				for item in predictList:

					imageArray, labelArray = self.dataPipe.next_image(item)
					predictionList = []

					# Feed the net with batches from the test data in order to create small tensors
					for j in range(0, imageArray.shape[0], batchSize):
						imageSlicesBatch = imageArray[j:j + batchSize, :, :, :]
						labelSlicesBatch = labelArray[j:j + batchSize, :, :]

						feed_dict = {self.restoredNet.X: imageSlicesBatch, self.restoredNet.Y: labelSlicesBatch, self.restoredNet.isTrain: False}
						predictionBatch = sess.run(self.restoredNet.predictions, feed_dict=feed_dict)
						predictionList.append(predictionBatch)

					# Append the predicted image into an array
					predictionImagesList.append(np.concatenate(predictionList, axis=0))

					sampleDice = diceScore(predictionImagesList[-1], labelArray)
					sampleAcc = accuracy(predictionImagesList[-1], labelArray)
					print('Sample number {} statistics: Dice score : {}  :: Accuracy: {}'.format(item, round(sampleDice, 4), round(sampleAcc, 4)))
					diceList.append(sampleDice)
					accList.append(sampleAcc)
					self.imagesList.append(imageArray)
					self.labelList.append(labelArray)

				self.predictionList = predictionImagesList

				endTime = time.time()
				print('Duration time for  prediction = {}'.format(endTime - startTime))
				print('Statistics to the model: Dice: {}, Accuracy: {}'.format(np.mean(diceList), np.mean(accList)))
				print('Dice statistics: mean {}, std {} median {}'.format(np.mean(diceList), np.std(diceList), np.median(diceList)))


		# TODO: NO META MODE
		else:
			pass

	def view_predictions_results(self):
		for num in range(len(self.predictionList)):
			# convert from patches to slices:
			self.paramsDict['cutPatch'] = True # TODO DEBUG
			if 'cutPatch' in self.paramsDict.keys() and self.paramsDict['cutPatch']:
				predicatedLabel, _ = getSlicesFromPatches(self.predictionList[num], np.squeeze(self.labelList[num]), 240)
				image, label = getSlicesFromPatches(self.imagesList[num], np.squeeze(self.labelList[num]), 240)
				predicatedLabel = np.squeeze(predicatedLabel)

			# the data is slices
			else:
				image = self.imagesList[num]
				label = np.squeeze(self.imagesList[num])
				predicatedLabel = np.squeeze(self.predictionList[num])
				predicatedLabel = np.swapaxes(predicatedLabel, 0, 1)

			while (True):
				index = input('\nFor 3d viewer press V\nFor next example press Q:\n')

				if index in ['Q', 'q']:
					break

				elif index in ['V', 'v']:
					modality = input('Please enter modality to view from the list {}\n'.format(self.dataPipe.modalityList))

					imageSingleModality = image[:, :, :, int(modality)]
					binaryPredicatedLabel = np.round(predicatedLabel)
					imageSingleModality = np.swapaxes(imageSingleModality, 0, 2)
					binaryPredicatedLabel = np.swapaxes(binaryPredicatedLabel, 0, 2)
					label = np.swapaxes(label, 0, 2)
					imageSingleModality = np.swapaxes(imageSingleModality, 2, 1)
					binaryPredicatedLabel = np.swapaxes(binaryPredicatedLabel, 2, 1)
					label = np.swapaxes(label, 2, 1)
					slidesViewer(imageSingleModality, binaryPredicatedLabel, label)
					plt.show()

				elif index == 'F':
					ax, fig = make_ax()
					img3d = image[:, :, :, 0]
					img3d = (img3d - np.min(img3d)) / (np.max(img3d) - np.min(img3d))
					img3d = resize(img3d, (img3d.shape[0] // 2, img3d.shape[1] // 2, img3d.shape[2] // 2),
					               mode='constant')
					ax.voxels(img3d, facecolors='#1f77b430', edgecolors='gray')
					plt.show()

				else:
					print('Wrong option, please try again:\n')


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
			                                                         D // (n * n) * (ind):D // (n * n) * (
			                                                         ind + 1),
			                                                         :]
			labelArray[H * i:H * (i + 1), H * j:H * (j + 1), :] = patchArrayLabel[:, :,
			                                                      D // (n * n) * (ind):D // (n * n) * (
			                                                      ind + 1)]
			ind += 1
	return imageArray, labelArray


def save_data_for_the_next_nn(modelTesterObj, outFileDir=''):
# if being used samplesToSave needs to contain the ind that we would like to save

	predictArraySave = []
	for item in range(len(testObj.predictionList)):
		predictArraySave.append(np.squeeze(testObj.predictionList[item]))
	np.save(outFileDir, predictArraySave)
	return

#############################
# Test code
#############################
# runDir = '/Users/royhirsch/Documents/GitHub/runDataFromTheServer/3005_1024/RunFolder_28_05_18__01_41_iter_num_63 copy'
# testObj = ModelTester(runDir)
# testObj.predict_test_data()
# save_data_for_the_next_nn(testObj, outFileDir=runDir+'/predictions_for_test_and_val.npy')
# testObj.view_predictions_results()




