# coding=utf-8
import numpy as np
from collections import defaultdict, OrderedDict
import re
import sys
import time
import tensorflow as tf

class TBNNAM:

    def __init__(self, settings, U=None):
        rng = np.random.RandomState(3435)
        if U is None:
            U = np.asarray(rng.standard_normal(size=(settings['word_count'], settings['emb_dim'])), dtype=np.float32) 
        self.settings = settings    

        hidden_dim = self.settings['emb_dim'] + self.settings['dim_ent']
        self.sent = tf.placeholder(tf.int64, [None, settings['max_l']], name='sent')
        self.ent = tf.placeholder(tf.int64, [None, settings['max_l']], name='ent')
        self.mask=tf.placeholder(tf.float32,[None,settings['max_l']],name='mask')
        self.lengths=tf.reduce_sum(self.mask,1)
        self.y=tf.placeholder(tf.float32,[None,1],name='y')
        self.evt=tf.placeholder(tf.int64,[None,1],name='evt')
        
        #word embedding
        self.w_embedding=tf.Variable(U,name='w_emb',trainable=False)

        #entity embedding
        E = np.asarray(0.01 * rng.standard_normal(size=(settings['n_ent'], 
            settings['dim_ent'])), dtype=np.float32)
        self.ent_embedding=tf.Variable(E,name='ent_emb')

        evt_emb = np.asarray(0.01 * rng.standard_normal(size=(settings['n_class'],hidden_dim)),dtype=np.float32)
        #event type embedding 1
        self.evt_embedding=tf.Variable(evt_emb,name='evt_emb')

        #event type embedding 2
        self.evt_embedding_last=tf.Variable(evt_emb,name='evt_emb_last')

        self.x_w=tf.nn.embedding_lookup(self.w_embedding,self.sent)
        self.x_e=tf.nn.embedding_lookup(self.ent_embedding,self.ent)
        self.x_evt=tf.nn.embedding_lookup(self.evt_embedding,self.evt)
        self.x_evt_last=tf.nn.embedding_lookup(self.evt_embedding_last,self.evt)
        self.x=tf.concat([self.x_w,self.x_e],2)

        #lstm 
        self.encoder = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        output, state = tf.nn.dynamic_rnn(self.encoder, self.x, sequence_length=self.lengths, dtype=tf.float32)

        #attention
        attention1_logits=tf.matmul(output,tf.transpose(self.x_evt,[0,2,1]))
        attention1_logits=tf.reshape(attention1_logits,[-1,settings['max_l']])*self.mask
        attention1=tf.exp(attention1_logits)*self.mask
        
        #attention score
        _score1=attention1_logits*attention1/tf.reduce_sum(attention1,1,keep_dims=True)
        score1=tf.reduce_sum(_score1,1,keep_dims=True)
        cell,hidden=state

        #global score
        self.x_evt_last=tf.reshape(self.x_evt_last,[-1,hidden_dim])
        score2=tf.reduce_sum(hidden*self.x_evt_last,axis=1,keep_dims=True)

        self.alpha = settings['alpha']
        self.score=score1*self.alpha+score2*(1-self.alpha)
        self.pred=self.score

        #loss
        self.cost=tf.reduce_mean(tf.square(self.score-self.y))
        all_vars   = tf.trainable_variables() 
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in all_vars ]) * settings['l2_weight']
        self.cost=self.cost+lossL2
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        self.att_value = _score1

