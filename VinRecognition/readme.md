# VIN-CODE RECOGNITION

VIN-code recognition is a neural network that can classify small squared black&white image (VIN character boxes) with single handwritten character on it.

The software was created using Python, tensorflow, pandas, numpy and cv2.

The class of artificial neural network designed in this project is a Convolutional Neural Network (CNN) model.

## Approach
The network is trained by two separate datasets for handwritten digits and characters.

The handwritten digits dataset used for the model creation - [Digits dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

The handwritten characters dataset used for the model creation - [Characters dataset](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)

For training the model the datasets are mixed together.

### Important note:
The VIN-code does not contain several characters to avoid misunderstanding. Those characters are: 'I', 'O' and 'Q'. The software is created to filter out data for the unused VIN-code characters to improve the accuracy.

Also, the software is designed to infer the background color for the provided images to adjust for the images input.


## Usage
### To run the inference with Docker:

1. To build the Docker image, navigate to the directory containing the Dockerfile and run:
    `docker build -t vin-rec-image .`
2. To work with external folder that contains images the command has to include external folder mounting. To run the inference script use next command:
    `docker run -it --rm -v [absolute path to folder with images]:/app/data vin-rec-image python inference.py --input /app/data`

### To run the inference in python venv:
1. Change directory to the VinRecognition:
    `cd [absolute_path]/VinRecognition`
2. Create virtual env:
    `python3 -m venv .`
3. Activate env:
    `source ./bin/activate`
4. Install requirements:
    `pip3 install -r requirements.txt`
5. Run inference:
    `python3 inference.py --input [absolute path to folder with images]`
