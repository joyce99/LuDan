from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np


class TR_GraphAttention(Layer):

    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 depth = 4,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 dropout_rate=0.3,
                 activation='relu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.attn_heads = attn_heads 
        self.attn_heads_reduction = attn_heads_reduction  
        self.dropout_rate = dropout_rate  
        self.activation = activations.get(activation)  
        self.use_bias = use_bias
        self.depth = depth

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.kernels = []      
        self.biases = []        
        self.attn_kernels = []
        self.attn_kernels1 = []
        self.attn_kernels2 = []

        super(TR_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_F = input_shape[0][-1]
        rel_F = input_shape[1][-1]
        ent_F = node_F+rel_F
        if self.depth == 0:
            self.built = True
            return
        
        for head in range(self.attn_heads):
            if self.use_bias:
                bias = self.add_weight(shape=(ent_F, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            attn_kernel_self = self.add_weight(shape=(ent_F,1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(ent_F, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            attn_kernel_rels = self.add_weight(shape=(rel_F, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_rel_{}'.format(head))
            
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs,attn_kernel_rels])
            self.attn_kernels1.append([attn_kernel_self, attn_kernel_neighs, attn_kernel_rels])
            self.attn_kernels2.append([attn_kernel_self, attn_kernel_neighs, attn_kernel_rels])
        self.built = True
        
    
    def call(self, inputs):
        ent_emb = inputs[0]
        rel_emb = inputs[1]

        adj = tf.SparseTensor(K.cast(K.squeeze(inputs[2],axis = 0),dtype = "int64"),
                         K.ones_like(inputs[2][0,:,0]),(self.node_size,self.node_size))
        sparse_indices = tf.squeeze(inputs[3],axis = 0)  
        sparse_val = tf.squeeze(inputs[4],axis = 0)
        
        rel_adj = K.cast(K.squeeze(inputs[5],axis = 0),dtype="int64")
        rel_adj = tf.SparseTensor(indices=rel_adj, values=tf.ones_like(rel_adj[:,0],dtype = 'float32'), dense_shape=(self.node_size,self.rel_size))
        rel_adj = tf.sparse_softmax(rel_adj)
        rel_features = tf.sparse_tensor_dense_matmul(rel_adj, rel_emb)
        
        ent_adj = K.cast(K.squeeze(inputs[6], axis=0), dtype="int64")
        ent_adj = tf.SparseTensor(indices=ent_adj, values=tf.ones_like(ent_adj[:,0],dtype = 'float32'), dense_shape=(self.node_size,self.node_size))
        ent_adj = tf.sparse_softmax(ent_adj)
        ent_features = tf.sparse_tensor_dense_matmul(ent_adj,ent_emb)
        
        features = K.concatenate([ent_features,rel_features])
        outputs = [self.activation(features)]



        for _ in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[head]

                # 修改部分
                attention_kernel1 = self.attn_kernels1[head]
                attention_kernel2 = self.attn_kernels2[head]
                # 截止


                # indices 表示想放进矩阵的位置 values表示在那个位置放得值 dense_shape是矩阵的大小
                attn_for_rels = tf.SparseTensor(indices=sparse_indices, values=sparse_val, dense_shape=(self.triple_size, self.rel_size))
                # 行是已知实体对个数，列是关系种类个数

                attn_for_rels = tf.squeeze(tf.sparse_tensor_dense_matmul(attn_for_rels, K.dot(rel_emb, attention_kernel[2])), axis = -1)
                # 行是已连接的实体对，列是关系种类的矩阵   *   （关系权重*关系嵌入向量）的矩阵

                attn_for_rels = tf.SparseTensor(indices=adj.indices, values=attn_for_rels, dense_shape=adj.dense_shape)

                #  行是实体个数， 列也是
                attn_for_self = K.dot(features, attention_kernel[0])

                # 行是实体个数，列是1
                attn_for_neighs = tf.transpose(K.dot(features, attention_kernel[1]), [1, 0])

                # 行是1，列是实体个数
                att = tf.sparse_add(tf.sparse_add(attn_for_rels, adj * attn_for_self), adj * attn_for_neighs)

                # 修改部分
                # attention_inputs1 = tf.matmul(att, attention_kernel1)
                # attention_inputs2 = tf.matmul(att, attention_kernel2)
                # con_sa_1 = tf.reduce_sum(tf.multiply(attention_inputs1, att), 1, keepdims=True)
                # con_sa_2 = tf.reduce_sum(tf.multiply(attention_inputs2, att), 1, keepdims=True)
                # con_sa_1 = tf.keras.activations.tanh(con_sa_1)
                # con_sa_2 = tf.keras.activations.tanh(con_sa_2)
                # if self.dropout_rate > 0.0:
                #     con_sa_1 = tf.nn.dropout(con_sa_1, self.dropout_rate)
                #     con_sa_2 = tf.nn.dropout(con_sa_2, self.dropout_rate)
                # con_sa_1 = tf.cast(features, dtype=tf.float32) * con_sa_1
                # con_sa_2 = tf.cast(features, dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])
                # att = tf.sparse.add(con_sa_1, con_sa_2)
                # 截止

                att = tf.SparseTensor(indices=att.indices, values=tf.nn.leaky_relu(att.values), dense_shape=att.dense_shape)
                att = tf.sparse_softmax(att)
                # att是一个实体个数*实体个数的权重矩阵
                new_features = tf.sparse_tensor_dense_matmul(att, features)
                    
                if self.use_bias:
                    new_features = K.bias_add(new_features, self.biases[head])
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = K.concatenate(features_list)
            else:
                features = K.mean(K.stack(features_list), axis=0)

            features = self.activation(features)
            outputs.append(features)
        outputs = K.concatenate(outputs)
        return [outputs, att.indices, att.values]

    def compute_output_shape(self, input_shape):          
        node_shape = self.node_size, (input_shape[0][-1] + input_shape[1][-1]) * (self.depth + 1)
        return [node_shape, None, None]