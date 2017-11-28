#!/bin/env python
import numpy as np
from skimage import io
from skimage.transform import resize


def create_data():
    image_path_array=[]
    lable_array=[]
    with open('train.list','r') as f:
        for line in f:
            image_path,label=line.strip().split('\t')
            image_path_array.append(image_path)
            if int(label)==1:
                label=1
            else:
                label=0
            lable_array.append(label)
    x_train=np.array([resize(io.imread(file_name,as_grey=True),(200,200)) for file_name in image_path_array])
    #x_train=np.array([resize(io.imread(file_name),(200,200)) for file_name in image_path_array])
    y_train=np.array(lable_array)
    
    test_image_path_array=[]
    test_lable_array=[]
    with open('val.list','r') as f:
        for line in f:
            image_path,label=line.strip().split('\t')
            test_image_path_array.append(image_path)
            if int(label)==1:
                label=1
            else:
                label=0
            test_lable_array.append(label)
    x_test=np.array([resize(io.imread(file_name,as_grey=True),(200,200)) for file_name in test_image_path_array])
    #x_test=np.array([resize(io.imread(file_name),(200,200)) for file_name in test_image_path_array])
    y_test=np.array(test_lable_array)
    np.savez("wx_model.npz",x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
    print x_train.shape,y_train.shape,x_test.shape,y_test.shape


def load_data():
    f = np.load("wx_model.npz")
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def train_reader():
    def reader():
        (x_train, y_train), (x_test, y_test) = load_data()
        for i in range(len(x_train)):
            data = x_train[i].flatten()
            label = y_train[i]
            yield (data).astype(np.float32), int(label)

    return reader


def test_reader():
    def reader():
        (x_train, y_train), (x_test, y_test) = load_data()
        for i in range(len(x_train)):
            data = x_test[i].flatten()
            label = y_test[i]
            yield (data).astype(np.float32), int(label)

    return reader


if __name__=="__main__":
    create_data()

