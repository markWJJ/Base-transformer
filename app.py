# -*- coding: utf-8 -*-

from flask import Flask, render_template, jsonify, request
import tensorflow as tf 
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys
from train import Graph
from data_load import load_de_vocab, load_en_vocab
import regex as re
from hyperparams import Hyperparams as hp
model_path = "logdir"
app = Flask(__name__)

de2idx, idx2de = load_de_vocab()
en2idx, idx2en = load_en_vocab()
g=Graph(is_training = False)

def create_data(input_sent):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    x = []
    y = []
    if len(input_sent)<(hp.maxlen-1):

        x.append(de2idx.get("<S>", 1))
        for each in input_sent:
            x.append(de2idx.get(each, 1))
        x.append(de2idx.get("</S>", 1))
        y.append(np.array(x))
        y = np.array(y)
    print(y.shape)
    Input = []
    Input.append(input_sent)

    X = np.zeros([len(y), hp.maxlen], np.int32)
    print(X.shape)
    X[0] = np.lib.pad(y[0], [0, hp.maxlen-len(y[0])], 'constant', constant_values=0)
    print(X.shape)
    return X, Input



@app.route('/synonymous/infer', methods=['GET', 'POST'])
def syn_infer_api():
    
    print("Graph loaded")
    if len(request.form) > 0:
        form = request.form
    else:
        form  = request.json
        
    query = form.get('query','')
    print("query>>>>",query)
    epoches = int(form.get('epoches', 0))
    beam_width = int(form.get('beam_width', 0))


    def _process(sent):
        sent = re.sub("[^a-zA-Z0-9\u4e00-\u9fa5\s']", "", sent)
        return sent

    query = _process(query)
    X, Input = create_data(query)

    def _beam_search(_x):
        preds_sent = np.zeros((1, hp.maxlen, beam_width))
        probs_sent = np.zeros_like(preds_sent, np.float64)
        #sent_len = _x[0, :].tolist().index(0)
        #print(x_a)
        preds = np.zeros((1, hp.maxlen), np.int32)
        preds_prob = np.zeros_like(preds, np.float64)
        _logits = np.array(sess.run(g.logits, {g.x:_x, g.y:preds}))
        sent_j = _logits[0,0]
        #print(sent_j)
        sos = sent_j.argsort()[-1:] #retrieve the token of first character (Start of sentence)
        preds[0,0] = sos #settle the sos token at the beginning of preds
        sos_prob = sent_j[sos]
        preds_prob[0,0] = sos_prob
        for bw in range(beam_width):
            preds_sent[0, 0, bw] = preds[0,0]
            probs_sent[0, 0, bw] = preds_prob[0,0]
        _logits = np.array(sess.run(g.logits, {g.x:_x, g.y:preds}))
        sent_j = _logits[0]
        word_1 = sent_j[1]
        word_1 = word_1+preds_prob[0,0]
        top_bw_idx = word_1.argsort()[-beam_width:]
        top_bw_probs = word_1[top_bw_idx]
        #print(top_bw_probs)
        for bw in range(beam_width):
            preds_sent[0,1,bw] = np.copy(top_bw_idx[bw])
            probs_sent[0,1,bw] = np.copy(top_bw_probs[bw])
        for k in range(2, hp.maxlen):  #this part need special design
            added_probs = []
            paths_candidate = []
            preds_prob_list = []

            for bw in range(beam_width):
                preds[0,:] = np.copy(preds_sent[0,:,bw])
                preds_prob[0,:] = np.copy(probs_sent[0,:,bw])

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
                    added_probs = np.concatenate((added_probs, [np.sum(current_prob)]), 0)
                    preds_prob_list.append(current_prob)


                    #added_probs = np.concatenate((added_probs, [1]), 0)
                    #paths_candidate.append(new_path)
                    #preds_prob[0,k] = 1
                    #current_preds_prob = np.copy(preds_prob)
                    #preds_prob_list.append(current_preds_prob)

                if (preds_sent[0, k-1, bw] != 3):
                    current_path = preds_sent[0, :, bw]
                    current_prob = probs_sent[0, :, bw]
                    _logits = np.array(sess.run(g.logits, {g.x:_x, g.y:preds}))
                    sent_j = _logits[0]
                    word_k = sent_j[k]#+preds_prob[0,k-1] #log(a*b) = log a + log b
                    top_bw_idx = word_k.argsort()[-beam_width:]
                    top_bw_probs = sent_j[k][top_bw_idx]


                    for bmw in range(beam_width):
                        new_path = np.copy(current_path)
                        new_path[k]=top_bw_idx[bmw]
                        current_step_probs = top_bw_probs[bmw]
                        current_path_probs = np.copy(current_prob)
                        current_path_probs[k] = current_step_probs


                        added_probs = np.concatenate((added_probs, [np.sum(current_path_probs)]), 0)
                        paths_candidate.append(new_path)
                      
                        
                        preds_prob_list.append(current_path_probs)


            a_idx = np.array(added_probs).argsort()[-beam_width:]
            print(a_idx)
            a_prob = added_probs[a_idx]
            for bw in range(beam_width):
                preds_sent[0,:,bw] = np.copy(paths_candidate[a_idx[bw]])

                probs_sent[0, :, bw] = np.copy(preds_prob_list[a_idx[bw]])

        return preds_sent, probs_sent, a_prob
        #return probs_sent

    ret = {"code": 100, "data": [], "prob":[]}

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config = tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(model_path))
            #mname = open(model_path + '/checkpoint', 'r').read().split('"')[1]
            x = X[0:1]
            Input_sent = Input[0:1]
            predx = np.zeros((1, hp.maxlen, int(beam_width)), np.int32)
            predx_prob = np.zeros_like(predx, np.float64)

            logits = np.zeros((1, hp.maxlen, len(en2idx)), np.float64)

            predx, probx, prob = _beam_search(x)
            print(predx.shape)
            candidate = []
            for pred in predx:
                for i in range(beam_width):
                    pres = pred[:, i]
                    got = "".join(idx2en[idx] for idx in pres).split("</S")[0].strip()
                    candidate.append(got)
                    print(got)
            print(predx.shape)
            ret = {"code": 200, "data": candidate, "prob":prob.tolist()}

            all_sents = []
            for i in range(beam_width):
                all_sents.append(predx[0, :, i])
            all_sents = np.array(all_sents)
            print(all_sents.shape)    
            all_probs = []

            for j in range(epoches-1):
                alls = np.copy(all_sents)
                new_all_sents = []
                new_all_probs = []

                for i in range(alls.shape[0]):
                    print("%d -- %d"%(j, i))
                    x = alls[i:i+1, :]
                    predx = np.zeros((1, hp.maxlen, int(beam_width)), np.int32)
                    predx_prob = np.zeros_like(predx, np.float64)
                    logits = np.zeros((1, hp.maxlen, len(en2idx)), np.float64)
                    predx1, probx1, prob1 = _beam_search(x)
                    
                    for n in range(beam_width):
                        new_all_sents.append(predx1[0, :, n])
                    new_all_probs = np.concatenate((new_all_probs, prob1),0)
                new_all_sents = np.array(new_all_sents)
                all_sents = new_all_sents
                print(all_sents.shape)
                all_probs = new_all_probs

            x_idx = np.array(all_probs).argsort()[-beam_width:]
            print(x_idx.shape)
            x_prob = all_probs[x_idx]
            x_sent = []
            for m in range(beam_width):
                x_sent.append(all_sents[x_idx[m]])

            candidate = []
            for each in x_sent:
                got = "".join(idx2en[idx] for idx in each).split("</S>")[0].strip()
                candidate.append(got)
            ret = {"code":200, "data":candidate, "prob":x_prob.tolist()}








            #return predx
    return jsonify(ret)
         

        

if __name__ == '__main__':
    app.run(host='0.0.0.0' ,port=6126)



@app.route('/synonymous/train', methods=['POST'])
def syn_train_api():


    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")


    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print("Restored!")
    # Start session
 
            if sv.should_stop(): 
                break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                    
            loss = sess.run(g.mean_loss) 
            print("============loss=========: %f" % loss)
            gs = sess.run(g.global_step)   
            sv.saver.save(sess, tf.train.latest_checkpoint(model_path))
            print(sess.run(g.acc))
    print("Done")
