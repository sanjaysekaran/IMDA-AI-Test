# CAPTCHA Recognition with Deep Learning

## Task Description

The sample/training data provided for this exercise consists of:

- **25 `.jpg` files** (input images),
- **25 `.txt` files** (equivalent RGB colour matrices in `.txt` format), and  
- **25 one-line `.txt` files** (corresponding 5-character CAPTCHA text).  

Each input image is a **60 x 30 pixel RGB** colour matrix, and each label contains the **5 ordered characters** embedded in the CAPTCHA.  

Based on this, we infer that the goal (though not explicitly stated) is to:

> **Create a supervised learning model to infer the 5 characters from a given `.jpg` CAPTCHA image and output the result to a one-line `.txt` file.**

Since the pipeline is expected to accept `.jpg` files directly, it makes sense to use the **raw image files** for training instead of preprocessed matrices.

While the most obvious approach is to build a **multi-output, multi-class CV model** that predicts all 5 characters at once, this is complex and unlikely to perform well with such a small dataset.  

A **simpler and more effective strategy** is to:
- **Segment** each CAPTCHA image into 5 individual characters, and  
- Convert the dataset into **125 samples (25 x 5)** to train a **standard multi-class model**.

In this exercise, we use:
- `TensorFlow` and `Keras` to design and train our deep learning model.
- `OpenCV` for image processing and segmentation.

---

## Model Details

- **Model Architecture**  
  Two sets of **convolutional layers** coupled with **pooling layers**, followed by a **fully connected layer** and a **final dropout layer**.  
  Activation Function: **ReLU** in convolutional layers.

- **Hyperparameter Optimisation**  
  Given the small dataset size, no full hyperparameter sweep was performed.  
  Some key parameters (like learning rate and training epochs) were manually tweaked.  
  **Early stopping** is included to reduce unnecessary computation.

- **Activation Function**  
  **Softmax**, as this is a **multi-class classification** problem.

- **Loss Function**  
  **Sparse categorical crossentropy**.

- **Optimiser**  
  Standard **Adam optimiser**.

---

## Training Data Preparation

- **Image Data**  
  `.jpg` images are:
  - Loaded in **grayscale**.
  - **Binary thresholding** is applied to suppress background noise.
  - **Contour detection** is used to extract individual character segments.

- **Label Data**  
  `.txt` files are:
  - Read and split into individual characters.
  - Each character is **integer encoded** to serve as the training labels.

---

## Usage

The `Captcha` class provides two main methods: `__train__` and `__call__`.

### `__train__` Method
This method enables model training. It requires:

- **Mandatory** parameters:
  - Input path (to load raw `.jpg` CAPTCHA images): `train_input_path`
  - Output path (to load `.txt` CAPTCHA text): `train_output_path` 

- **Optional** parameters:
  - Number of epochs: `epochs`
  - Batch size: `batch_size` 
  - Learning rate (Adam optimiser): `learning_rate` 
  - Validation split fraction: `validation_split` 
  - Minimum delta for early stopping: `early_stopping_min_delta`

Additionally:
- Trained model weights are saved to a **Keras model file**, which can be reloaded using the `load_model` method in the `Captcha` class.

### `__call__` Method
This method enables model inference. It requires:

- A trained model (either via `__train__` or `load_model`)
- Input path (to load a single `.jpg` file for inference): `inference_input_path`
- Output path (to save predicted CAPTCHA labels`.txt` prediction): `inference_output_path` 

The method will:
1. Load the input image.
2. Preprocess it for the model.
3. Predict the 5-character CAPTCHA.
4. Save the result to the specified output `.txt` file.

---

## Observations

- The code runs **end-to-end on Google Colab**.
- Achieved a **training accuracy above 85%**.
- The model correctly predicted the CAPTCHA for `input100.jpg` (the only input without a corresponding CAPTCHA text), demonstrating reasonably good generalization.
- With more training data, performance can be improved further.
- Additional model evaluation metrics can be added to serve as **production benchmarks** for monitoring and future iterations.
