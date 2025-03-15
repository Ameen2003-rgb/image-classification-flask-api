# image-classification-flask-api
A deep learning-based image classification API built using TensorFlow and Flask. This project includes data preprocessing, model training, and a REST API for making predictions. The application is containerized using Docker for easy deployment. 🚀
### Data Preprocessing and Feature Engineering Report
1. Dataset Loading
The dataset consists of images categorized into multiple classes. The images were loaded using TensorFlow/Keras and converted into NumPy arrays for efficient processing.
2. Image Resizing
Since deep learning models require a fixed input size, all images were resized to 32x32 pixels to maintain uniformity and optimize training performance.
3. Normalization
To enhance training efficiency and stability, pixel values were scaled to the range [0,1] by dividing all pixel values by 255. This normalization step ensures that the model learns efficiently without being affected by large numerical values.
4. Data Augmentation
To improve generalization and prevent overfitting, data augmentation techniques were applied, including:
•	Random rotations
•	Horizontal and vertical flipping
•	Zooming and shifting This step artificially expands the dataset and helps the model learn more diverse representations.
5. Splitting the Dataset
The dataset was divided into three subsets:
•	Training Set (80%): Used to train the model.
•	Validation Set (10%): Used to tune hyperparameters and prevent overfitting.
•	Test Set (10%): Used to evaluate final model performance.
6. One-Hot Encoding of Labels
Since the dataset consists of categorical labels (e.g., "Airplane", "Car"), one-hot encoding was applied to convert class labels into numerical vectors. Example:
•	"Cat" → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] This ensures compatibility with neural network architectures that require numerical inputs.
7. Handling Imbalanced Data
The dataset was analyzed for class imbalances. If an imbalance was detected, techniques such as class weighting or oversampling were employed to ensure fair model training.
8. Preprocessing for Model Inference
To maintain consistency during real-time predictions, the same preprocessing steps were applied to test images, including:
•	Image resizing
•	Normalization
•	Expanding batch dimensions before inputting to the model

**### Model Selection and Optimization Approach**
Selecting the appropriate model and optimizing its performance involved the following steps:
1.	Model Architecture:
o	A Convolutional Neural Network (CNN) was chosen due to its effectiveness in image classification tasks.
o	Layers included convolutional, pooling, batch normalization, and dense layers to enhance feature extraction.
o	Dropout layers were added to reduce overfitting.
2.	Hyperparameter Tuning:
o	Used techniques like Grid Search and Random Search to optimize hyperparameters such as learning rate, batch size, and number of layers.
o	Experimented with different activation functions (ReLU, Softmax) and optimizers (Adam, SGD).
3.	Training Strategy:
o	Used a learning rate scheduler to adjust the learning rate dynamically during training.
o	Implemented early stopping to prevent overfitting when validation loss stopped improving.
o	Data augmentation was used to improve generalization.
4.	Evaluation Metrics:
o	Accuracy, precision, recall, and F1-score were used to evaluate model performance.
o	Confusion matrices and ROC curves were analyzed for detailed insights.
5.	Model Performance Improvement:
o	Transfer learning was experimented with using pre-trained models like ResNet and VGG16.
o	Fine-tuned the last few layers of pre-trained models to adapt them to the dataset.
o	Regularization techniques (L2 regularization, dropout) were applied to reduce overfitting.

**### Deployment Strategy and API Usage Guide**
Deployment Strategy:
•	Containerized the application using Docker to ensure portability and scalability.

•	Exposed the model as a REST API using FastAPI/Flask.

•	Used a cloud platform (AWS, Azure, or GCP) for deployment, ensuring scalability.

•	Optimized API response time by enabling model caching and efficient inference processing.

API Usage Guide:

Endpoint: Predict

•	URL: /predict

•	Method: POST

•	Request Body: Multipart/form-data with an image file.

Example Request (using cURL):

curl -X POST "http://127.0.0.1:5000/predict" -F "file=@image.jpg"

Example Response:

{
  "predicted_class": "Tumor",
  "confidence_score": 0.92
}

This API allows users to upload an image, and the model will return the predicted class along with a confidence score.

