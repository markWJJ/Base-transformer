# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys
import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
     
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
                
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
             
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                    predx = np.zeros((hp.batch_size, hp.maxlen, hp.beam_width), np.int32)
                    predx_prob = np.zeros_like(predx, np.float64)
                    
                    logits = np.zeros((hp.batch_size, hp.maxlen, len(en2idx)), np.float64)
                    print(x[1:2,:])


                    for j in range(hp.batch_size): #For testing, the range will be changed to accelerate the testing for j in range(hp.maxlen)
                        print(j)
                        preds_sent = np.zeros((1, hp.maxlen, hp.beam_width))
                        probs_sent = np.zeros_like(preds_sent, np.float64)
                        #probs_ref = np.zeros_like(preds_sent, np.float64)
                        x_a = x[j:j+1,:] #input one sentence each time
                        #sent_len = x_a[0, :].tolist().index(0)
                        #print(x_a)
                        preds = np.zeros((1, hp.maxlen), np.int32)
                        preds_prob = np.zeros_like(preds, np.float64)
                        _logits = np.array(sess.run(g.logits, {g.x:x_a, g.y:preds}))
                        sent_j = _logits[0,0]
                        #print(sent_j)
                        sos = sent_j.argsort()[-1:] #retrieve the token of first character (Start of sentence)
                        preds[0,0] = sos #settle the sos token at the beginning of preds
                        sos_prob = sent_j[sos]
                        preds_prob[0,0] = sos_prob
                        #print(preds[0,0])
                        for bw in range(hp.beam_width):
                            preds_sent[0, 0, bw] = preds[0,0]
                            probs_sent[0, 0, bw] = preds_prob[0,0]
                        #print(probs_sent)
                        _logits = np.array(sess.run(g.logits, {g.x:x_a, g.y:preds}))
                        sent_j = _logits[0]
                        word_1 = sent_j[1]
                        word_1 = word_1+preds_prob[0,0]
                        top_bw_idx = word_1.argsort()[-hp.beam_width:]
                        #print(top_bw_idx)
                        top_bw_probs = word_1[top_bw_idx]
                        #print(top_bw_probs)
                        for bw in range(hp.beam_width):
                            preds_sent[0,1,bw] = np.copy(top_bw_idx[bw])
                            #print(top_bw_probs[bw])
                            probs_sent[0,1,bw] = top_bw_probs[bw]

                            #这里打印一下

                        #print(probs_sent)
                        #settle top_bw tokens for the second character (first word)

                        #print(probs_sent)
                        for k in range(2, hp.maxlen):  #this part need special design
                            added_probs = []
                            paths_candidate = []
                            preds_prob_list = []

                            for bw in range(hp.beam_width):
                                preds[0,:] = preds_sent[0,:,bw].copy()

                                preds_prob[0,:] = probs_sent[0,:,bw].copy()
                                #print(preds_prob)

                                if (preds_sent[0, k-1, bw] == 3):
                                    preds_sent[0, k, bw] = 3
                                    current_path = preds_sent[0,:, bw]
                                    new_path = np.copy(current_path)
                                    new_path[k] = 3
                                    


                                    paths_candidate.append(new_path)

                                    probs_sent[0, k, bw] = 0
                                    pre_prob = probs_sent[0, :, bw]
                                    current_prob = np.copy(pre_prob)
                                    current_prob[k] = 0
                                    #current_preds_prob = np.copy(preds_prob)
                                    print(current_prob)
                                    added_probs = np.concatenate((added_probs, [np.sum(current_prob)]), 0)

                                    preds_prob_list.append(current_prob)

                                if (preds_sent[0, k-1, bw] != 3):

                                    current_path = preds_sent[0, :, bw]
                                    current_prob = probs_sent[0, :, bw]
                                    _logits = np.array(sess.run(g.logits, {g.x:x_a, g.y:preds}))
                                    sent_j = _logits[0]
                                    word_k = sent_j[k] #+np.sum(preds_prob[0]) #log(a*b) = log a + log b
                                    top_bw_idx = word_k.argsort()[-hp.beam_width:]

                                    top_bw_probs = sent_j[k][top_bw_idx]


                                    for bmw in range(hp.beam_width):
                                        new_path = np.copy(current_path)
                                        new_path[k]=top_bw_idx[bmw]
                                        current_step_probs = top_bw_probs[bmw]
                                        current_path_probs = np.copy(current_prob)
                                        current_path_probs[k] = current_step_probs
                                        added_probs = np.concatenate((added_probs, [np.sum(current_path_probs)]), 0)
                                        #print(new_path)
                                        paths_candidate.append(new_path)
                                        preds_prob_list.append(current_path_probs)

                                #print("what hell is going on")
                                #print(sub_candidates)
                                #print("this is a =========")                               


                            a_idx = np.array(added_probs).argsort()[-hp.beam_width:]
                            a_prob = added_probs[a_idx]
                            #print(a_prob)

                            print(preds_prob_list)
                            for bw in range(hp.beam_width):

                                preds_sent[0, :, bw] = np.copy(paths_candidate[a_idx[bw]])
                                #print(paths_candidate[a_idx[bw]])
                                #print(preds_sent[0, :, bw])


                                probs_sent[0, :, bw] = np.copy(preds_prob_list[a_idx[bw]])
                                print(probs_sent)


                            #print("probs_sent:")
                            #print(probs_sent)

                        predx[j, :, :] = preds_sent
                        predx_prob[j, :, :] = probs_sent
                        #print("checkpoint")
                        #sys.exit()
    ### Write to file

                    print("done")
                    for source, target, pred, prob in zip(sources, targets, predx, predx_prob): # sentence-wise
                        candits = []
                        candits_probs = []
                        for i in range(hp.beam_width):
                            pres = pred[:,i]
                            pros = prob[:,i]
                            got = "".join(idx2en[idx] for idx in pres).split("</S>")[0].strip()
                            candits.append(got)
                            candits_probs.append(pros)

                        fout.write("- source:   " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        print(candits)

                        for i in range(len(candits)):
                            fout.write("- got:      " + candits[i] + "\n")
                            m = len(candits[i])
                            fout.write(' '.join(str(each) for each in candits_probs[i].tolist()[:m-2]))#each for each in 
                            fout.write("\n")
                        fout.write("\n")

                        
                        fout.flush()
                          
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                ## Calculate bleu score
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
