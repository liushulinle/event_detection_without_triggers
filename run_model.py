# -*-coding:utf8-*-
import sys
import time
import os
import tensorflow as tf
import numpy as np
import random
import json

from utils import tools, load_data, show_result
from model import tbnnam_model

Prifix = os.path.join(os.getcwd(), os.path.dirname(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
def convert2binary(data, ydict, neg_prob=0.4):
    sen, ent, y = data
    
    ret_sen, ret_ent, ret_evt, ret_label, ret_mask = [], [], [], [], []
    for idx in range(len(sen)):
        for ly in ydict.values():
            lb = 1 if ly in y[idx] else 0
            if lb == 0 and random.random() > neg_prob:continue
            ret_sen.append(sen[idx])
            ret_ent.append(ent[idx])
            ret_evt.append(ly)
            ret_label.append(lb)
            ret_mask.append([1 if x >=0 else 0 for x in sen[idx]])
    return np.asarray(ret_sen, dtype='int32'), np.asarray(ret_ent, dtype='int32'), np.asarray(ret_evt, dtype='int32'), \
           np.asarray(ret_label, dtype='int32'), np.asarray(ret_mask, dtype='float32')

def save_dicts(path, dicts):
    ans = json.dumps(dicts)
    fout = open(path, 'w')
    fout.writelines(ans)
    fout.close()

def load_dicts(path):
    ans = json.loads(open(path).read())
    return ans

def predict_sen(sess,sen, ent, ydict, cmodel, max_ans = 3):
    ans = []
    sens, ents, evts, masks = [], [], [], []
    labels = list(ydict.values())
    for y in labels: 
        sens.append(sen)
        ents.append(ent)
        masks.append([1 if x >=0 else 0 for x in sen])
        evts.append(y)
    sens = np.array(sens, dtype='int32')
    evts = np.array(evts, dtype='int32')
    masks = np.array(masks, dtype='float32')
    evts = evts[:,np.newaxis]

    feeddict={cmodel.sent:sens,cmodel.ent:ents,cmodel.evt:evts,cmodel.mask:masks}
    pred = sess.run(cmodel.pred,feed_dict=feeddict)
    
    for y, p in zip(labels, pred):
        if y != ydict['negative'] and p > 0.5:
            ans.append((y, p))
        if y == ydict['negative'] and p < 0.5:
            #print s
            pass
    ans = sorted(ans, cmp=lambda a, b: cmp(a[1], b[1]), reverse=True)

    ret = []
    if len(ans) > 0:
        for k in ans[:max_ans]:
            ret.append(k[0])
    else:
        ret.append(ydict['negative'])
    return ret


def run_model(train_data, WORDS, settings, wdict, ydict, edict): # wdict, ydict used to show predicted result

    t_train_sen, t_train_ent, t_train_evt, t_train_y, t_train_mask = convert2binary(train_data, ydict)
    tf.reset_default_graph()

    #if dataset size is not multiple of batch_size, relicate
    if len(t_train_sen) % settings['batch_size'] > 0:
        extra_size = settings['batch_size'] - len(t_train_sen) % settings['batch_size']
        rand_train = np.random.permutation(range(len(t_train_sen)))[:extra_size]
        extra_y = t_train_y[rand_train]
        extra_sen = t_train_sen[rand_train]
        extra_evt = t_train_evt[rand_train]
        extra_ent = t_train_ent[rand_train]
        extra_mask = t_train_mask[rand_train]

        t_train_y = np.concatenate((t_train_y, extra_y))
        t_train_evt = np.concatenate((t_train_evt, extra_evt))
        t_train_ent = np.concatenate((t_train_ent, extra_ent))
        t_train_sen = np.concatenate((t_train_sen, extra_sen))
        t_train_mask = np.concatenate((t_train_mask, extra_mask))

    cmodel = tbnnam_model.TBNNAM(settings, WORDS)

    epchs = 0
    n_batchs = len(t_train_y) / settings['batch_size']
    batch_size = settings['batch_size']
    best_f = -1
    init = tf.global_variables_initializer()

    
    saver = tf.train.Saver()
    settings['word_count'] = WORDS.shape[0]
    dicts = {'wdict':wdict, 'ydict':ydict, 'edict': edict, 'settings': settings}
    save_dicts("trained_models/dicts.json", dicts)
    with tf.Session() as sess:
        sess.run(init)
        while epchs < settings['n_eps']:
            shuff = np.random.permutation(len(t_train_sen))
            epchs += 1
            ers = []
            tic = time.time()
            for k in xrange(n_batchs):
                batch_sent = t_train_sen[shuff[k * batch_size: (k + 1) * batch_size]]
                batch_evt = t_train_evt[shuff[k * batch_size: (k + 1) * batch_size]]
                batch_ent = t_train_ent[shuff[k * batch_size: (k + 1) * batch_size]]
                batch_y = t_train_y[shuff[k * batch_size: (k + 1) * batch_size]]
                batch_evt = batch_evt[:,np.newaxis]
                batch_y = batch_y[:,np.newaxis]
                batch_mask = t_train_mask[shuff[k * batch_size: (k + 1) * batch_size]]
                feeddict={cmodel.sent:batch_sent,cmodel.ent:batch_ent,cmodel.evt:batch_evt,cmodel.mask:batch_mask,cmodel.y:batch_y}
                _,loss=sess.run([cmodel.optimizer,cmodel.cost],feed_dict=feeddict)
                ers.append(loss)
                print '\r[learning] epoch %i >> %2.2f%%' % (epchs, (k + 1) * 100.0 / n_batchs), \
                    'completed in %.2f (s)' % (time.time() - tic), 'loss: %.4f' % np.mean(ers),
                sys.stdout.flush()
            print
            
            saver.save(sess, "trained_models/iter_%d.ckpt" % epchs)

def train(alpha=0.25):
    s = {
        'emb_dim': 200, #word embedding size
        'max_l': 40, #max sen length
        'n_class': 35,
        'n_ent': 55,
        'dim_ent': 50,
        'l2_weight': 0.00001,
        'n_eps': 25,
        'batch_size': 100,
        'alpha': alpha,
        }
    
    train_path = '%s/data/corpus_train.txt' % Prifix
    edict_path = '%s/data/dicts/ent_dict.txt' % Prifix
    wdict_path = '%s/data/dicts/word_dict.txt' % Prifix
    ydict_path = '%s/data/dicts/label_dict.txt' % Prifix
    
    wdict = tools.load_dict(wdict_path)
    edict = tools.load_dict(edict_path)
    ydict = tools.load_dict(ydict_path)
    ydict = {k.lower(): v for k, v in ydict.items()}
    
    train_data = load_data.load_data_ent(train_path, wdict, edict, ydict, s['max_l'])
    word_dest_p = '%s/data/embeddings/200.txt' % Prifix
    WORDS = tools.load_embedding(word_dest_p)
    
    run_model(train_data, WORDS, s, wdict, ydict, edict)


def eval_model(test_path, model_dir, model_version): # wdict, ydict used to show predicted result
    def test_sent(test_sents, test_ents, test_y):
        n_test_batch = len(test_sents)
        t_result = []
        for k in xrange(n_test_batch):
            pred = predict_sen(sess,test_sents[k], test_ents[k], ydict, cmodel)
            t_result.append((pred, test_y[k]))
            ori_sen = ' '.join([rwdict[x] for x in test_sents[k] if x >= 0])
            pred_ans = ','.join([rydict[x] for x in pred])
            gold_ans = ','.join([rydict[x] for x in test_y[k]])
            print 'Sample %d: [Sen=%s] \n\t [ans=%s], [pred_events=%s]\n' % (k, ori_sen, gold_ans, pred_ans)

    
        ptr_str, f = show_result.evaluate_results_binary(t_result, ydict['negative'])
        print ptr_str

   
    dicts = load_dicts(model_dir + '/dicts.json')
    wdict, ydict, edict, settings = dicts['wdict'], dicts['ydict'], dicts['edict'], dicts['settings']
    rwdict = {v : k for k, v in wdict.items()}
    rydict = {v: k for k, v in ydict.items()}
    test_data = load_data.load_data_ent(test_path, wdict, edict, ydict, settings['max_l'])
    test_sents, test_ents, test_y = test_data
    tf.reset_default_graph()

    #if dataset size is not multiple of batch_size, relicate
    cmodel = tbnnam_model.TBNNAM(settings)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver()
        model_path = model_dir + '/' + model_version
        saver.restore(sess, model_path)
        test_sent(test_sents, test_ents, test_y)
           


def run_eval():
    test_path = 'data/corpus_test_10.txt'
    model_dir = 'trained_models'
    model_ver = 'model.ckpt'
    eval_model(test_path, model_dir, model_ver)   



if __name__ == '__main__':
    if sys.argv[1].strip().lower() == 'train':
        train(0.25)
    elif sys.argv[1].strip().lower() == 'evaluation':
        run_eval()
    else:
        print 'Error: Unkown Command'
    

