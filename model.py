#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        batch_size : int
            batch_size.
        num_steps : int
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)

        with tf.variable_scope('rnn'):
            ##################
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_size, state_is_tuple=True)
            
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                                 input_keep_prob=1.0,
                                                 output_keep_prob=self.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.rnn_layers, state_is_tuple=True)
           
            #依然是初始化一个[batch_size,state_size]的矩阵
            self.state_tensor = cell.zero_state(self.batch_size, dtype=tf.float32)
            """
            tf.nn.dynamic_runn的输出shape为(batch_size,num_steps,dim_embedding),其state
            的shape为(batch_size,dim_embedding)
            """
            rnn_outputs, self.outputs_state_tensor = tf.nn.dynamic_rnn(cell, 
                                                              rnn_inputs, 
                                                              initial_state=self.state_tensor)
            ##################

        # flatten it
        seq_output_final = tf.reshape(rnn_outputs, [-1, self.dim_embedding])

        with tf.variable_scope('softmax'):
            ##################
            W = tf.get_variable(
                'W',
                [self.state_size, self.num_words],     
                initializer=tf.random_normal_initializer(stddev=0.01)
            )
            b = tf.get_variable(
                'b',
                [self.num_words],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )
            ##################
        logits = tf.matmul(seq_output_final, W) + b
        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Y, [-1])
        mean, var = tf.nn.moments(logits, -1)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)

        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))
        tf.summary.scalar('var_loss', var_loss)
        # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
