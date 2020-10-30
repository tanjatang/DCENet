"""
Created on Wed Oct 15 20:57:00 2020
@author: cheng,tang,liao
"""
from keras_multi_head import MultiHeadAttention
from keras.layers import Dense
from keras import backend as K
from keras.layers.core import Dropout, Layer
from keras.models import Sequential
import numpy as np


class LayerNormalization(Layer):
    '''big thx to git@github.com:kpot/keras-transformer.git'''

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.att = MultiHeadAttention(head_num=num_heads, name='att_layer')
    def call(self, inputs, training):
        q = inputs
        k = inputs
        v = inputs
        attn_output = self.att([v,k,q])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]

        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)
    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b

class Encoder(Layer):
    def __init__(self,embed_dim,num_heads,ff_dim,num_layers):
        self.num_layers = num_layers
        self.transformer_block_list = [TransformerBlock(embed_dim,num_heads, ff_dim)  for _ in range(self.num_layers)]
        self.fc = Dense(embed_dim,activation='relu')

        self.pos_encoding = PositionEncoding(embed_dim)
        #
    def __call__(self,x):
        x = self.fc(x)
        x_pos_enc = self.pos_encoding(x)
        x = Add()([x,x_pos_enc])
        x = Dropout(0.1)(x)
        for enc in self.transformer_block_list:
            x = enc(x,training=False)
        return x

