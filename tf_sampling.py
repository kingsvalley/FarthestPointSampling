''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017. 
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))
def prob_sample(inp,inpr):
    '''
input:
    batch_size * ncategory float32
    batch_size * npoints   float32
returns:
    batch_size * npoints   int32
    '''
    return sampling_module.prob_sample(inp,inpr)
ops.NoGradient('ProbSample')
# TF1.0 API requires set shape in C++
#@tf.RegisterShape('ProbSample')
#def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp,idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return sampling_module.gather_point(inp,idx)
#@tf.RegisterShape('GatherPoint')
#def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sampling_module.gather_point_grad(inp,idx,out_g),None]
def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')
    

def load_data_from_hd5(hd5file):
    f = h5py.File(hd5file)
    data = f['test_set_x'][:]
    return data

def save_data_to_h5(data,hd5_outfile):
    h5file = h5py.File(hd5_outfile, 'w')
    h5file.create_dataset('data', data = data)
    h5file.close() 

def farthest_points_sampling(h5files):
    import numpy as np   
    with tf.device('/gpu:5'):
        for h5file in h5files:
            hd6_data = load_data_from_hd5(h5file)
            hd6_data = np.array([hd6_data]).astype('float32')
            pt_sample=tf.Variable(hd6_data)
                        
            reduced_sample=gather_point(pt_sample,farthest_point_sample(1024,pt_sample)) 
            
            out_h5file = os.path.splitext(h5file)[0]
            out_h5file = out_h5file + '.hd5'
            with tf.Session('') as sess:
                sess.run(pt_sample.initializer)
                ret=sess.run(reduced_sample)
                print ret
                save_data_to_h5(ret,out_h5file)
            print ret.shape,ret.dtype

if __name__ == '__main__':
    h5files = []
    h5files.append('test.h5')
    h5files.append('test2.h5')
    print h5files
    
    farthest_points_sampling(h5files)
        
        
        
        
        
        
        
        
        
        
        

