from UnetModel import *
import matplotlib.pyplot as plt

def startLogging(logDir, debug=False):
	# this function starts logging the software outputs.
	# two levels logging: DEBUG and INFO

	# init a logger set logging level
	logFormat = '%(asctime)s - %(levelname)s - %(module)s : %(message)s'
	if debug:
		logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.DEBUG)
		logLevel = logging.DEBUG
	else:
		logging.basicConfig(format=logFormat, stream=sys.stdout, level=logging.INFO)
		logLevel = logging.INFO

	logStr = time.strftime('logFile_%H_%M__%d_%m_%y') + '.log'

	# connect the two streams
	fileHandler = logging.FileHandler(logDir+'/'+logStr)

	fileHandler.setFormatter(logging.Formatter(logFormat))
	fileHandler.setLevel(logLevel)

	logging.getLogger('').addHandler(fileHandler)


def resultDisplay(predictions, labels, images, sampleInd, imageSize, imageMod, thresh = 0.5):

	# get specific sample
	prediction = predictions[sampleInd,:,:]
	label = labels[sampleInd,:,:]
	image = images[sampleInd,:,:,imageMod]

	# clip prediction into bineary mask
	prediction[prediction > thresh] = 1
	prediction[prediction <= thresh] = 0

	# print results
	plt.figure(1)
	plt.subplot(131)
	plt.title('Label')
	plt.imshow(np.reshape(label, [imageSize, imageSize]), cmap='gray')
	plt.subplot(132)
	plt.title('Prediction')
	plt.imshow(np.reshape(prediction, [imageSize, imageSize]), cmap='gray')
	plt.subplot(133)
	plt.title('Image')
	plt.imshow(np.reshape(image, [imageSize, imageSize]), cmap='gray')
	plt.show()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def createFolder(homeDir, folderName):
	directory = homeDir + '/' + folderName
	if not os.path.exists(directory):
		os.makedirs(directory)


#creating 3D plot
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax,fig
#viewer for 3D images (show the slices in 2 d)

def slidesViewer(volume1,volume2,volume3):
	fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
	ax1.set_title('Image')
	ax2.set_title('Prediction')
	ax3.set_title('Label')

	ax1.volume = volume1
	ax2.volume = volume2
	ax3.volume = volume3
	ax1.index = volume1.shape[0] // 2
	ax2.index = volume2.shape[0] // 2
	ax3.index = volume3.shape[0] // 2

	ax1.imshow(volume1[ax1.index])
	ax2.imshow(volume2[ax2.index])
	ax3.imshow(volume3[ax3.index])

	fig.canvas.mpl_connect('key_press_event', process_key)



# help function for slidesveiwer
def process_key(event):
	fig = event.canvas.figure
	axes = fig.axes
	if event.key == 'D' or event.key == 'd' :
		for ax in axes:
			next_slice(ax)
	elif event.key == 'A' or event.key == 'a':
		for ax in axes:
			previous_slice(ax)

	fig.canvas.draw()

# previous_slice change to previes slice in process_key
def previous_slice(ax):
	volume = ax.volume
	ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
	ax.images[0].set_array(volume[ax.index])

# previous_slice change to previes slice in process_key

def next_slice(ax):
	volume = ax.volume
	ax.index = (ax.index + 1) % volume.shape[0]
	ax.images[0].set_array(volume[ax.index])

