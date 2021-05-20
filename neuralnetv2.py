#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:24:10 2019

@author: manuel
"""
from keras.models import Model
from keras import backend as K
from keras import layers
from keras.models import Input
from keras import regularizers
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from numpy.random import seed
seed(42)# keras seed fixing
import tensorflow as tf
tf.random.set_random_seed(42)# tensorflow seed fixing
C = 1
lbeta = 0.5
#@tf.custom_gradient
class ESLU(Activation):
    
    def __init__(self, activation, **kwargs):
        super(ESLU, self).__init__(activation, **kwargs)
        self.__name__ = 'ESLU_s'



def myeslu(x):
    #  x = tf.placeholder(dtype=tf.float32, shape=[1,100])
    # #  def grad(x):
     #  cond12 = tf.cast(tf.math.greater(x, 1.0), tf.float32)
     #  cond22 = tf.cast(tf.math.logical_and(tf.math.less_equal(x, 1.0), tf.math.greater_equal(x, 0.0)), tf.float32)
     #  cond42 = tf.cast(tf.math.less(x, 0.0), tf.float32)
     #  a2 = tf.math.multiply(cond12, lalpha*K.pow(x,(lbeta-1)))
     #  b2 = tf.math.multiply(cond22, lalpha)
     #  d2 = tf.math.multiply(cond42, lalpha*K.exp(x))
     #  f2 =  d2 +b2+a2
     #  if (x>=1):
     #      return lalpha*x**(lbeta-1)
     #  elif(x<1 and x>0):
     #      return lalpha
     #  else:
     #     return lalpha*np.exp(x)
    #      return x*0
    #  cond12 = tf.cast(tf.math.greater(x, 0.0), tf.float32)
    #  cond22 = tf.cast(tf.math.less_equal(x, 0.0), tf.float32)
      
      cond1 = tf.cast(tf.math.greater(x, 1.0), tf.float32)
      cond2 = tf.cast(tf.math.logical_and(tf.math.less_equal(x, 1.0), tf.math.greater_equal(x, 0.0)), tf.float32)
      cond4 = tf.cast(tf.math.less(x, 0.0), tf.float32)
      
      #a1 = tf.math.multiply(cond12, lalpha*x+lalpha)
      #b1 = tf.math.multiply(cond22, lalpha*K.exp(x))
      
      a = tf.math.multiply(C/lbeta*K.sqrt(x)+C*(2-1/lbeta),cond1)#lalpha/lbeta*K.pow(x,lbeta)+lalpha
      b = tf.math.multiply( C*(x+1),cond2)
      d = tf.math.multiply( C*K.exp(x),cond4)#lalpha*K.exp(x)
      
      f = d +b+a
    #  y = np.linspace(-5,5,num=100)
    #  with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   print(sess.run(f, feed_dict={x: [y]}))    
            
      return f#,grad
get_custom_objects().update({'myeslu': myeslu})




#https://stackoverflow.com/questions/39921607/how-to-make-a-custom-activation-function-with-only-python-in-tensorflow
# def eslu(x,lalpha = 1.0,lbeta = 0.5):
#     if (x>=1):
#         return lalpha*x**(lbeta)/lbeta + lalpha
#     elif(x<1 and x>0):
#         return lalpha*x+lalpha
#     else:
#         return np.exp(x)*lalpha
# def d_eslu(x,lalpha = 1.0,lbeta = 0.5):
#     if (x>=1):
#         return lalpha*x**(lbeta-1)
#     elif(x<1 and x>0):
#         return lalpha
#     else:
#         return lalpha*np.exp(x)
#     # (lalpha*np.exp(x))*(x<0)+ lalpha*()
# np_d_eslu = np.vectorize(d_eslu)
# np_elsu = np.vectorize(eslu)
# #casting to 32 since tensorflow only works in float32
# np_eslu_32 = lambda x: np_elsu(x).astype(np.float32)
# np_d_eslu_32 = lambda x: np_d_eslu(x).astype(np.float32)

# def py_func(func, inp, Tout, stateful=True, name = None, grad=None):
#     rnd_name = 'PyFuncGrad'+ str(np.random.randint(0,1E+8))
    
#     tf.RegisterGradient(rnd_name)(grad)
#     g=tf.compat.v1.get_default_graph()
#     with g.gradient_override_map({"PyFunc": rnd_name}):
#         return tf.py_func(func, inp, Tout, stateful=stateful,name=name)

# def eslugrad(op,grad):
#     x = op.inputs[0]
#     n_gr = tf_d_eslu(x)
#     return grad*n_gr

# def tf_d_eslu(x, name =None):
#     with  tf.name_scope(name, "d_eslu", [x]) as name:
#         y = tf.py_func(np_d_eslu_32,
#                        [x], 
#                        [tf.float32],
#                        name= name,
#                        stateful=False)
#         return y[0]
# def tf_eslu(x, name = None):
#     with tf.name_scope(name, "eslu", [x]) as name:
#         y= py_func(np_eslu_32,
#                   [x],
#                   [tf.float32],
#                   name=name,
#                   grad = eslugrad)
#         return y[0]
#channels is n+4 since n is v_1-v_n 3 for position and 1 for go signal
def buildneuralnet(time,channels,dropped,numRNNneurons,m1a,mua,kernm1a,kernmua):
    
    inputlayer1 = Input(shape=(time,channels))
    #units was channels, now 30
    dotmatinput = Input(shape=(6,2))
    model2 = buildneuralnet2(time,channels,dropped,numRNNneurons,m1a,mua,kernm1a,kernmua)
    Force2,m1context2,muscleactivation2 =model2([inputlayer1,dotmatinput])
    
    model = Model(inputs=[inputlayer1,dotmatinput],outputs=Force2)
    
    return model,model2
def buildneuralnet2(time,channels,dropped,numRNNneurons,m1a,mua,kernm1a,kernmua):
    actualinput = Input(shape=(time,channels))
    #units was channels, now 30
    m1context = layers.SimpleRNN(units=numRNNneurons,input_shape=(time,channels),activation = myeslu,return_sequences=True,dropout=dropped, activity_regularizer=regularizers.l2(m1a),kernel_regularizer=regularizers.l2(kernm1a))(actualinput)#kernel_regularizer = regularizers.l2(0.5), activity_regularizer=regularizers.l1(0.5)
    muscleactivation = layers.Dense(6, activation='elu', activity_regularizer=regularizers.l2(mua))(m1context)#kernel_regularizer = regularizers.l2(0.01),#kernel_regularizer = regularizers.l2(0.1), , activity_regularizer=regularizers.l1(0.01)
    dotmatinput = Input(shape=(6,2))
    Force = layers.dot([muscleactivation,dotmatinput],axes=[2,1])
    model = Model(inputs=[actualinput,dotmatinput],outputs=[Force,m1context,muscleactivation])
    return model
#tuned activity_regularizer=regularizers.l1(0.00000001)
# kernel_regularizer=regularizers.l2(0.05), activity_regularizer=regularizers.l1(0.000001)
