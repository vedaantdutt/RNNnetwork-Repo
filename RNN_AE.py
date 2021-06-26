from __future__ import division, print_function, absolute_import
import tensorflow as tf
import sys,os,pdb
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.dataLayer import DataLayer
import numpy as np
import _pickle as pickle
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import mycode.cost as costfunc
from mycode.cost import _modified_mse
from mycode.utility import get_gt_target_xyz,generate_fake_batch,snapshot,split_into_u_var
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential
import torch.nn as nn
import torch

class LSTM_Autoencoder:
    
    def __init__(self, latent_space, input_features):

        self._latent_space = latent_space
        self._input_cells = input_features

        self._encoder = None
        self._decoder = None
        self._autoencoder = None
        self._configure_network()

    def _configure_network(self):
       
        def repeat_vector(args):
            [layer_to_repeat, sequence_layer] = args
            return RepeatVector(K.shape(sequence_layer)
  
   def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

 
  def build_model(self):
    timesteps = self.timesteps
    n_features = self.n_features
    model = Sequential()
    
    # Encoder
    model.add(LSTM(timesteps, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.add(RepeatVector(timesteps))
   
  
    # Decoder
    model.add(LSTM(timesteps, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(optimizer=self.optimizer, loss=self.loss)
    model.summary()
    self.model = model
    
  def fit(self, X, epochs=3, batch_size=32):
    self.timesteps = X.shape[1]
    self.build_model()
    
    input_X = np.expand_dims(X, axis=2)
    self.model.fit(input_X, input_X, epochs=epochs, batch_size=batch_size)
    
  def predict(self, X):
    input_X = np.expand_dims(X, axis=2)
    output_X = self.model.predict(input_X)
    reconstruction = np.squeeze(output_X)
    return np.linalg.norm(X - reconstruction, axis=-1)

if __name__ == '__main__':
    # execfile('code/dataIO.py')
    # print('data preparation done!')
    
    tag = ''
    # Training Parameters
    is_test = False
    learning_rate = 0.001
    batch_size = 32
    display_step = 2
    training_epochs = 200
    fps = 30
    data_dim = 3
    num_user = 47
    if cfg.include_own_history:
        num_user = num_user+1
    # Network Parameters
    # num_input = num_user*cfg.running_length
    # num_output = 1*cfg.predict_len
    # if cfg.has_reconstruct_loss:
    #     num_output = 1*cfg.running_length
    dropout = 0.75 # Dropout, probability to keep units

    X = tf.placeholder(tf.float32, [None,num_user,2*cfg.running_length*fps,data_dim])
    Y = tf.placeholder(tf.float32, [None,cfg.running_length*fps,data_dim])

    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    weights = {
        'wc1': tf.Variable(tf.random_normal([9, 9, data_dim, 256], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([9, 9, 256, 512], stddev=0.1)),
        'wc3': tf.Variable(tf.random_normal([9, 9, 512, 512], stddev=0.1)),
   
        'out': tf.Variable(tf.random_normal([9, 9, 512, data_dim], stddev=0.1)),

        
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([256], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([512], stddev=0.1)),
        'bc3': tf.Variable(tf.random_normal([512], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([data_dim], stddev=0.1)),
   
    }   
 

    conv1,conv2,conv3,out = conv_net(X, weights, biases, keep_prob, num_user, cfg.running_length)
  
    predicted_tail = out[:,-1,-cfg.running_length*fps:,:]
    loss_op = tf.losses.mean_squared_error(predicted_tail,Y)

    lr = tf.Variable(cfg.LEARNING_RATE, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss_op)



    # summary
    all_losses_dict = {}
    # all_losses_dict['MSE_loss'] = loss_op1
    all_losses_dict['modified_MSE_loss'] = loss_op
    event_summaries = {}
    event_summaries.update(all_losses_dict)
    summaries = []
    for key, var in event_summaries.items():
        summaries.append(tf.summary.scalar(key, var))
    summary_op = tf.summary.merge(summaries)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    ## data IO
    if cfg.use_xyz:
        all_video_data = pickle.load(open('./data/exp_1_xyz.p','rb'))
        # all_video_data = pickle.load(open('./data/exp_2_xyz.p','rb'))
    elif cfg.use_cos_sin:
        all_video_data = pickle.load(open('./data/exp_2_raw_pair.p','rb'))
    else:
        all_video_data = pickle.load(open('./data/exp_2_raw.p','rb'))

    datadb = clip_xyz(all_video_data)
    data_io = DataLayer(datadb, random=False, is_test=is_test)

    if not is_test:
        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            summary_writer = tf.summary.FileWriter('./tfsummary/', sess.graph)
            total_batch = 8*int(datadb[0]['x'].shape[1]/cfg.running_length/fps/batch_size)
            for epoch in range(training_epochs):
                avg_cost = 0.
                for step in range(1, total_batch):
                    # print('step',step)
                    batch_x, batch_y = data_io._get_next_minibatch(datadb,batch_size,'CNN')
                    # Run optimization op (backprop)
                    _, c, summary = sess.run([train_op, loss_op, summary_op], feed_dict={X: batch_x,
                                                                    Y: batch_y,
                                                                    keep_prob: 0.85})
                    avg_cost += c / total_batch
                    summary_writer.add_summary(summary, float(epoch*total_batch+step))


                if epoch % display_step == 0:
                    print("epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                if epoch!=0 and epoch % 100 == 0:
                    snapshot(sess, (epoch), saver,'CNN', tag)
                    lr_temp = cfg.LEARNING_RATE*(0.5**(epoch/100))
                    print('epoch: ',epoch, ', change lr=lr*0.5, lr=', lr_temp)
                    sess.run(tf.assign(lr, lr_temp))

            snapshot(sess, (epoch), saver,'CNN', tag)
            print("Optimization Finished!")

    elif is_test:
        # testing
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            # Restore variables
            filename = 'CNN_'+tag + '_epoch_{:d}'.format(100) + '.ckpt'
            filename = os.path.join(cfg.OUTPUT_DIR, filename)
            saver.restore(sess, filename)
            print("Model restored.")
            data_io_test = DataLayer(datadb, random=False, is_test=True)


            test_out = []
            gt_out = []
            for ii in range(10):
                batch_x, batch_y = data_io_test._get_next_minibatch(datadb,batch_size,'CNN')
                test_out,gt_out = print_test(sess,batch_x,batch_y,test_out,gt_out)
        



        pickle.dump(test_out,open('CNN_test_out'+tag+'.p','wb'))
        pickle.dump(gt_out,open('CNN_gt_out'+tag+'.p','wb'))

