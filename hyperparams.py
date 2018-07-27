# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    diction = '/notebooks/source/cn_trans_0/sgns.merge.char.txt'
    train = 'train/zhidao_train.txt'
    test = 'train/test_01.txt'
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 30 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 15
    num_heads = 8
    dropout_rate = 0.01
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    beam_width = 5
    
    
    
    
