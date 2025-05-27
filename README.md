Task description:
The sample/training data provided for this exercise consists of 25 .jpg files along with the corresponding 60 x 30 pixel RGB colour matrices of 5-character captcha data in the input folder, and 25 one-line .txt files each containing the 5 ordered characters in the output folder. Based on this information, we can infer that the specific task (although not explicitly mentioned in the test document itself) is to create a supervised learning model to infer the 5 characters embedded within a given input captcha that is to be provided in .jpg format, and output the result to a one-line .txt file. 

Since the entire pipeline is expected to take a .jpg file as an input, it makes more sense to use the raw .jpg files as part of the input data processing pipeline for training. Although the most obvious approach would be to create a CV model that takes an entire captcha image as an input and then outputs all of the characters at one go (a multioutput multiclass model), this approach is far more complex of a task and the limited size of the sample dataset would likely mean that the resulting model would not perform so well. It would be simpler to segment the images into individual characters, effectively turning the training set into a 25 x 5 = 125 entries and train a multiclass model instead. In this exercise, the tensorflow and keras libraries were used to architect our deep learning model, along with the OpenCV library to perform image processing.

Model details:
Model architecture - 2 sets of convolutional layers coupled with pooling layers, ending with a fully connected layer coupled with a final dropout layer. Convolutional layers use the standard ReLU activation function.
Hyperparameter optimisation - Due to the small number of training samples, a full hyperparameter optimisation suite has not been incorporated, but the manual tweaking of certain key hyperparameters such as the learning rate and number of training epochs for the purposes of smoothening the training process using minor tweaks. Early stopping is included to minimise computational time.
Activation function -  SoftMax since we are dealing with multiclass classification.
Loss function - Sparse categorical crossentropy is used since we are dealing with multiclass classification .
Optimiser - Standard Adam optimiser.

Training data preparation:
Image data - Data in .jpg format is first loaded in grayscale before applying a binary threshold to suppress background noise before applying contour detection to obtain the individual character segments.
Label data - Data in .txt files are loaded, separated into the individual characters and integer encoded to represent training labels.

Usage:
The Captcha class has two main methods: __train__ and __call__.

The __train__ method has multiple input variables where it is mandatory to specify the input (i.e. the .jpg files) and output (i.e. the .txt fies) paths. We can optionally also modify the number of epochs, the batch size, the learning rate of the Adam optimiser, the validation fraction and the minimum delta threhold for early stopping. In addition to training a model which can then be used for subsequent inference, the __train__ method also saves the model weights to a keras model file that can later be loaded using the load_model function of the Captcha class.

The __call__ method is used for model inference, and mandatorily requires a model to have been trained (either using the __train__ method or loaded from an existing keras model file). In addition, the inference input and output paths must be specified and the method will load the input .jpg file, transform it into the appropriate format for model inference, perform and parse and model prediction and finally save the model prediction to the .txt file in the specified output path.

Observations:
The code runs end-to-end on Google Colab, and we are able to achieve a training accuracy above 85%. Applying the model on input100.jpg (the only captcha without a corresponding output file) gave an exact match, which is a positive outcome for this relatively simple model. With more training samples, we would be able to train superior models, along with incorporating robust model evaluation metrics that can eventually be used as benchmarks in production for monitoring or future model iterations. 



