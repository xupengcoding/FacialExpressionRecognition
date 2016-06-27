import os
import sys
import numpy as np
import pickle
#all pkl file format is label + content
#load data_vec data_lable

def load_data_xy(file_names):
    datas = []
    lables = []
    for file_name in file_names:
        f = open(file_name, 'rb')
        x, y = pickle.load(f)
        datas.append(x)
        lables.append(y)

    combine_label = np.hstack(lables)
    combine_data = np.vstack(datas)
    return combine_data, combine_label

def cPickle_output(var, file_name):
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(var, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def out_put_data_xy(vector_vars, vector_folder, batch_size = 1000):
    if not vector_folder.endswith('/'):
        vector_folder += '/'
    if not os.path.exists(vector_folder):
        os.mkdir(vector_folder)
    x, y = vector_vars
    n_batch = len(x) / batch_size
    for i in range(n_batch):
        file_name = vector_folder + str(i) + '.pkl'
        batch_x = x[i*batch_size: (i+1)*batch_size]
        batch_y = y[i*batch_size: (i+1)*batch_size]
        cPickle_output((batch_x, batch_y), file_name)
    if n_batch * batch_size < len(x):
        batch_x = x[n_batch*batch_size: ]
        batch_y = y[n_batch*batch_size: ]
        file_name = vector_folder + str(n_batch) + '.pkl'
        cPickle_output((batch_x, batch_y), file_name)


def out_put_data_xy_small(vector_vars, vector_folder):
    if not vector_folder.endswith('/'):
        vector_folder += '/'
    if not os.path.exists(vector_folder):
        os.mkdir(vector_folder)
    x, y = vector_vars
    n_batch = len(x) / batch_size
    for i in range(n_batch):
        file_name = vector_folder + str(i) + '.pkl'
        batch_x = x[i*batch_size: (i+1)*batch_size]
        batch_y = y[i*batch_size: (i+1)*batch_size]
        cPickle_output((batch_x, batch_y), file_name)
    if n_batch * batch_size < len(x):
        batch_x = x[n_batch*batch_size: ]
        batch_y = y[n_batch*batch_size: ]
        file_name = vector_folder + str(n_batch) + '.pkl'
        cPickle_output((batch_x, batch_y), file_name)


def scandir(startdir, file, last_dir):
    os.chdir(startdir)
    childlist = os.listdir(os.curdir)
    for obj in childlist:
        if os.path.isdir(obj):
            scandir(os.getcwd() + os.sep + obj, file, last_dir)
        else:
            file.append(os.getcwd() + os.sep + obj)
            last_dir.append(os.getcwd())

    os.chdir(os.pardir)
    return file, last_dir


def get_files(vec_folder):
    file_names = os.listdir(vec_folder)
    file_names.sort()
    if not vec_folder.endswith('/'):
        vec_folder += '/'
    for i in range(len(file_names)):
        file_names[i] = vec_folder + file_names[i]
    return file_names
