# Hand_written_alphabets_digits_recognition
Hand written alphabets digits recognition using CNN

#Aphabet_recognition_model.py
Emnist --'byclass' is used to build the Alphabet + Digits recogition model.
tar files should be unzipped to gzip folder

#Alphabet_recognition.py
val_images--> Contains images to be validated
Images are read, converted to black and white and resized to 28*28 before being sent for prediction.

-----------------------------------------------------------------------------------
#Digit_recognition_model.py
Tensor flow tf.keras.datasets.mnist and mnist.load_data() method are used to load training and test sets.

#Digit_recognition.py
val_images--> Contains images to be validated
Images are read, converted to black and white and resized to 28*28 before being sent for prediction.
