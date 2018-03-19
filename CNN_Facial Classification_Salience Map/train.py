import pandas as pd
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU,PReLU
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def load(readnpy=True):
    if readnpy:
        y = np.load('./feature/label.npy')
        X = np.load('./feature/feature.npy')
    else :
        df = pd.read_csv('./feature/train.csv')
        y = df['label'].as_matrix()
        y = pd.get_dummies(y).values
        X = df['feature'].as_matrix()
        X = np.array([np.array([*map(int, x.split())]).reshape(48,48) for x in X])
        np.save('./feature/label.npy', y)
        np.save('./feature/feature.npy',X)
    print(y.shape)
    print (X.shape)
    return X,y

def show_train_history(history,train,validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def split(x,y):
    index=[i for i in range(len(x))]
    random.shuffle(index)
    val=x[index[0:2800]]
    val_y=y[index[0:2800]]
    train=x[index[2801:]]
    train_y=y[index[2801:]]
    return val,val_y,train,train_y

def dnn(X,y):
    X = X.astype('float32')
    X /=255
    X = X.reshape(len(X),48,48,1)
    checkpoint_loss = ModelCheckpoint('model_loss.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
    checkpoint_acc = ModelCheckpoint('model_acc.h5', monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'max')

    model = Sequential()
    model.add(Flatten(input_shape=(48,48,1)))
    model.add(Dense(1024,kernel_regularizer=regularizers.l2(0.005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024,kernel_regularizer=regularizers.l2(0.005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))


    model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))


    model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history=model.fit(X, y,
              batch_size=1024,
              epochs=200,
              verbose=1,
              validation_split=0.1,class_weight='auto')#,callbacks=[checkpoint_loss,checkpoint_acc])
    show_train_history(history,'acc','val_acc')
    show_train_history(history,'loss','val_loss')


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def cnn(X,y):
    X = X.astype('float32')
    X /=255
    X = X.reshape(len(X),48,48,1)
    checkpoint_loss = ModelCheckpoint('model_loss2.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
    checkpoint_acc = ModelCheckpoint('model_acc2.h5', monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'max')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))   
    model.add(Conv2D(64, (3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))   
    model.add(Conv2D(128, (3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))   
    model.add(Conv2D(256, (3, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    

    model.add(Flatten())
    model.add(Dense(1024,kernel_regularizer=regularizers.l2(0.025)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512,kernel_regularizer=regularizers.l2(0.025)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256,kernel_regularizer=regularizers.l2(0.025)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    history=model.fit(X, y,
              batch_size=512,
              epochs=400,
              verbose=1,
              validation_split=0.1,class_weight='auto',callbacks=[checkpoint_loss,checkpoint_acc])
    

def test_load():
    df = pd.read_csv('./feature/test.csv')
    X = df['feature'].as_matrix()
    X = np.array([np.array([*map(int, x.split())]).reshape(48,48) for x in X])
    X = X.astype('float32')
    X /=255
    X = X.reshape(len(X),48,48,1)
    return X

def predict(model,test):
    predict_val=model.predict_classes(test, batch_size=512, verbose=0)
    id=range(np.shape(predict_val)[0])
    output=pd.DataFrame ({'id':id,'label':predict_val})
    return output



def main(*args):
    df = pd.read_csv('args[0][1]')
    y = df['label'].as_matrix()
    y = pd.get_dummies(y).values
    x = df['feature'].as_matrix()
    x = np.array([np.array([*map(int, x.split())]).reshape(48,48) for x in X])
    cnn(x,y)
    

if __name__ == '__main__':
    main(sys.argv)