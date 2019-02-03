# Sign Language Recognizer

A simple deep learning project which uses CNNs to recognize the digits from the hand gesture.

**Demo Video:** https://youtu.be/exHGP5kIlDA

## Project Details

* **API :** Keras
* **Backend:** Tensorflow
* **Dataset:** https://github.com/ardamavi/Sign-Language-Digits-Dataset + Some custom images
* **Model Architecture:** Conv->MaxPool->Conv->MaxPool->Flatten->Dense(512)->Dense(128)->Output(10)
* **Python Version:** >=3.6

## How to use

* Install the dependencies using:
```bash
pip3 install -r requirements.txt
```
* (Optional) Prepare you custom dataset. Run the script and press the number key for which you want to save the image.
```bash
python3 capture_digits.py
```
* Train the model using:
```bash
python3 train_model.py
```
* Run the digits recognizer using:
```bash
python3 recognize.py
```
