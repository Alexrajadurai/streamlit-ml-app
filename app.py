import streamlit as st

st.title('Customizable Nueral Network')

num_nuerons = st.sidebar.slider('Number of nuerons in hidden layer:', 1, 64)
num_epochs = st.sidebar.slider('Number of epochs', 1, 10)
activation = st.sidebar.text_input('Activation Function')

"the number of nuerons is : " + str(num_nuerons)
"the activation function is : " + activation

if st.button('Train Model'):
    import tensorflow as tf
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.callbacks import ModelCheckpoint

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def preprocess_image(images):
        images = images / 255
        return images
    
    x_train = preprocess_image(x_train)
    x_test = preprocess_image(x_test)

    model = Sequential()
    model.add(InputLayer((28,28)))
    model.add(Flatten())
    model.add(Dense(num_nuerons,activation))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    save_cp = ModelCheckpoint('model.keras',save_best_only=True)
    history_cp = tf.keras.callbacks.CSVLogger('history.csv',separator=',')
    model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=num_epochs,callbacks=[save_cp,history_cp])



if st.button('Evaluate Model'):
    import pandas as pd
    import matplotlib.pyplot as plt

    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'],history['accuracy'])
    plt.plot(history['epoch'],history['val_accuracy'])
    plt.title('Model Accuracy vs Epochs')
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.legend(['train','val'])
    fig
