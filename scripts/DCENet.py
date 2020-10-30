# -*- coding: utf-8 -*-
import os
from keras.layers import Input, Dense, Lambda, concatenate, LSTM, Activation, Flatten, MaxPooling2D, GlobalAveragePooling1D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.core import RepeatVector, Dropout
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.losses import mse
from keras.utils import plot_model
from self_attention_layer import Encoder


class dcenet():

    def __init__(self, args):
        # Store the hyperparameters
        self.args = args
        self.num_pred = args.num_pred
        self.obs_seq = args.obs_seq - 1  ### minus one is for residual
        self.pred_seq = args.pred_seq
        self.train_mode = args.train_mode
        self.n_hidden = args.n_hidden
        self.z_dim = args.z_dim
        self.encoder_dim = args.encoder_dim
        self.z_decoder_dim = args.z_decoder_dim
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.o_drop = args.o_drop
        self.s_drop = args.s_drop
        self.z_drop = args.z_drop
        self.lr = args.lr
        self.epochs = args.epochs
        self.beta = args.beta
        self.query_dim = args.query_dim
        self.keyvalue_dim = args.keyvalue_dim
        self.enviro_pdim = args.enviro_pdim

        #################### MODEL CONSTRUCTION STARTS FROM HERE ####################

        # (1-1) Construct the dynamic map model for past time
        self.occu_in = Input(shape=self.enviro_pdim, name='x_DMap_in')
        self.occu_Conv1 = Conv2D(6, kernel_size=2, strides=1, padding='same', activation='relu', name='x_Map_Conv1')(
            self.occu_in)

        self.occu_MP1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same', name='DMap_MP1')(self.occu_Conv1)
        self.occu_DP = Dropout(self.o_drop, name="occu_DP")(self.occu_MP1)
        self.occu_FT = Flatten(name='occu_FT')(self.occu_DP)
        self.occu_model = Model(self.occu_in, self.occu_FT)

        # (1-2) Add the time axis
        self.occus_in = Input(shape=(self.obs_seq, self.enviro_pdim[0], self.enviro_pdim[1], self.enviro_pdim[2]),
                              name='x_DMaps_in')
        self.occus_layers = TimeDistributed(self.occu_model, name='occus_layers')(self.occus_in)

        # Add self-attention encoder network for the occupancy information
        self.occu_encoder_x = Encoder(self.args.occu_encoder_x_dim, self.args.occu_encoder_x_head, self.args.occu_encoder_x_dim, self.args.occu_encoder_x_layers)
        self.att_layer_in = self.occu_encoder_x(self.occus_layers)
        self.occus_out = LSTM(self.hidden_size,
                              return_sequences=False,
                              stateful=False,
                              dropout=self.s_drop,
                              name='x_DMaps_out')(self.att_layer_in)

        self.interaction_model = Model(self.occus_in,
                                       self.occus_out)

        # (2) Construct the sequence model
        # sequence model for the observed data
        self.x = Input(shape=(self.obs_seq, 2), name='x')
        self.x_conv1d = Conv1D(self.n_hidden // 16, kernel_size=3, strides=1, padding='same', name='x_conv1d')(self.x)
        self.x_dense = Dense(self.n_hidden // 8, activation='relu', name='x_dense')(self.x_conv1d)
        self.x_encoder = Encoder(self.args.x_encoder_dim, self.args.x_encoder_head, self.args.x_encoder_dim, self.args.x_encoder_layers)
        self.x_dense_in = self.x_encoder(self.x_dense)
        self.x_state = GlobalAveragePooling1D()(self.x_dense_in)


        # encoded x
        self.x_endoced = concatenate([self.x_state, self.occus_out], name='x_endoced')
        self.x_encoded_dense = Dense(self.encoder_dim, activation='relu', name='x_encoded_dense')(self.x_endoced)

        # Construct the dynamic map model for future time
        # (3-1) Construct the dynamic map model for past time
        # Reuse the same occu_model as in the observation time

        # (3-2) Add the time axis
        self.y_occus_in = Input(shape=(self.pred_seq, self.enviro_pdim[0], self.enviro_pdim[1], self.enviro_pdim[2]),
                                name='y_DMaps_in')
        self.y_occus_layers = TimeDistributed(self.occu_model, name='y_DMaps_layers')(self.y_occus_in)


        # Add self-attention encoder network for the occupancy information
        self.occu_encoder_y = Encoder(self.args.occu_encoder_y_dim, self.args.occu_encoder_y_head, self.args.occu_encoder_y_dim, self.args.occu_encoder_y_layers)
        self.y_occus_out_in = self.occu_encoder_y(self.y_occus_layers)

        # (3-4) Construct LSTM to parse the att_layer for the environment
        self.y_occus_out = LSTM(self.hidden_size,
                                return_sequences=False,
                                stateful=False,
                                dropout=self.o_drop,
                                name='y_DMaps_out')(self.y_occus_out_in)


        self.y_interaction_model = Model(self.y_occus_in, self.y_occus_out)

        # sequence model for the ground truth
        self.y = Input(shape=(self.pred_seq, 2), name='y')
        self.y_conv1d = Conv1D(self.n_hidden // 16, kernel_size=3, strides=1, padding='same', name='y_conv1d')(self.y)
        self.y_dense = Dense(self.n_hidden // 8, activation='relu', name='y_dense')(self.y_conv1d)
        self.y_encoder = Encoder(self.args.y_encoder_dim, self.args.y_encoder_head, self.args.y_encoder_dim, self.args.y_encoder_layers)
        self.y_dense_in = self.y_encoder(self.y_dense)
        self.y_state = GlobalAveragePooling1D()(self.y_dense_in)


        # encoded y
        self.y_endoced = concatenate([self.y_state, self.y_occus_out], name='y_endoced')
        self.y_encoded_dense = Dense(self.encoder_dim, activation='relu', name='y_encoded_dense')(self.y_endoced)

        # CONSTRUCT THE DCENet ENCODER BY FEEDING THE CONCATENATED ENCODED HEATMAP, OCCUPANCY GRID, AND TRAJECTORY INFORMATION
        # the concatenated input
        self.inputs = concatenate([self.x_encoded_dense, self.y_encoded_dense], name='inputs')
        self.xy_encoded_d1 = Dense(self.n_hidden, activation='relu', name='xy_encoded_d1')(self.inputs)
        self.xy_encoded_d2 = Dense(self.n_hidden // 2, activation='relu', name='xy_encoded_d2')(self.xy_encoded_d1)
        self.mu = Dense(self.z_dim, activation='linear', name='mu')(self.xy_encoded_d2)  # 2
        self.log_var = Dense(self.z_dim, activation='linear', name='log_var')(self.xy_encoded_d2)  # 2

        # (4) THE REPARAMETERIZATION TRICK FOR THE LATENT VARIABLE z
        # sampling function
        z_dim = self.z_dim

        def sampling(params):
            mu, log_var = params
            eps = K.random_normal(shape=(K.shape(mu)[0], z_dim), mean=0., stddev=1.0)
            return mu + K.exp(log_var / 2.) * eps

        # sampling z
        self.z = Lambda(sampling, output_shape=(self.z_dim,), name='z')([self.mu, self.log_var])
        # concatenate the z and x_encoded_dense
        self.z_cond = concatenate([self.z, self.x_encoded_dense], name='z_cond')

        # CONSTRUCT THE DCENet DECODER
        self.z_decoder1 = Dense(self.n_hidden // 2, activation='relu', name='z_decoder1')
        self.z_decoder2 = RepeatVector(self.pred_seq, name='z_decoder2')
        self.z_decoder3 = LSTM(self.z_decoder_dim,
                               return_sequences=True,
                               stateful=False,
                               dropout=self.z_drop,
                               name='z_decoder3')
        self.z_decoder4 = Activation('tanh', name='z_decoder4')
        self.z_decoder5 = Dropout(self.z_drop, name='z_decoder5')
        self.y_decoder = TimeDistributed(Dense(2), name='y_decoder')  # (12, 2)

        # Instantiate the decoder by feeding the concatenated z and x_encoded_dense
        # Reconstrcting y
        self.z_d1 = self.z_decoder1(self.z_cond)
        self.z_d2 = self.z_decoder2(self.z_d1)

        self.z_d3 = self.z_decoder3(
            self.z_d2)


        self.z_d4 = self.z_decoder4(self.z_d3)
        self.z_d5 = self.z_decoder5(self.z_d4)
        self.y_prime = self.y_decoder(self.z_d5)


    def training(self):
        """
        Construct the DCENet model in training time
        Both observation and prediction are available
        y is the ground truth trajectory
        """
        print('Contruct the DCENet model for training')

        def vae_loss(y,y_prime):
            '''
            This is the customized loss function
            It consists of L2 and KL loss
            '''
            reconstruction_loss = K.mean(mse(y, self.y_prime) * self.pred_seq)
            kl_loss = 0.5 * K.sum(K.square(self.mu) + K.exp(self.log_var) - self.log_var - 1, axis=-1)
            DCENet_loss = K.mean(reconstruction_loss * self.beta + kl_loss * (1 - self.beta))
            return DCENet_loss

        # BUILD THE DCENet MODEL
        dcenet_model = Model([self.occus_in, self.x, self.y_occus_in, self.y],
                     [self.y_prime])
        opt = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=1e-6, amsgrad=False)
        dcenet_model.compile(optimizer=opt, loss=vae_loss)


        save_dir = '../models'
        filepath = os.path.join(save_dir, 'DCENet.pdf')
        plot_model(dcenet_model, to_file=filepath, show_shapes=True)
        return dcenet_model

    def X_encoder(self):
        """
        Construct the encoder to get the x_encoded_dense,
        including environment maps, occupancy, and trajectory sequence information
        NOTE:
            In inference phase, ONLY environment maps, occupancy, and trajectory for x from observation time is availabel
        Returns
        x_encoder : TYPE
            DESCRIPTION.
        """
        print('Construct the X-Encoder for inference')
        x_encoder = Model([self.occus_in, self.x], self.x_encoded_dense)
        x_encoder.summary()

        save_dir = '../models'
        filepath = os.path.join(save_dir, 'X-Encoder.pdf')
        plot_model(x_encoder, to_file=filepath, show_shapes=True)

        y_encoder = Model([self.y_occus_in, self.y], self.y_encoded_dense)
        filepath = os.path.join(save_dir, 'Y-Encoder.pdf')
        plot_model(y_encoder, to_file=filepath, show_shapes=True)

        return x_encoder

    def Decoder(self):
        # CONSTRUCT THE DECODER
        print('Construct the Decoder for trajectory oreidction')
        decoder_input = Input(shape=(self.z_dim + self.encoder_dim,), name='decoder_input')
        _z_d1 = self.z_decoder1(decoder_input)
        _z_d2 = self.z_decoder2(_z_d1)
        _z_d3 = self.z_decoder3(_z_d2)
        _z_d4 = self.z_decoder4(_z_d3)
        _z_d5 = self.z_decoder5(_z_d4)
        _y_prime = self.y_decoder(_z_d5)
        generator = Model(decoder_input, _y_prime)
        # generator.summary()
        save_dir = '../models'
        filepath = os.path.join(save_dir, 'Decoder.pdf')
        plot_model(generator, to_file=filepath, show_shapes=True)
        return generator

