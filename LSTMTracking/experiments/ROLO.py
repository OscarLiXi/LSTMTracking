'''
train LSTM module as tracking filter, hlcv project ss17
'''

import tensorflow as tf

import numpy as np
import faster_rcnn_reader as reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "train", "type of data")
flags.DEFINE_string("GTList_path", "../../py-faster-rcnn/data/MOT2016/GT_list.txt", "Path of available gt data list")
flags.DEFINE_string("debug", 1, "debug mode, display intermediate results")
flags.DEFINE_string("weights_file", "./output/weights_MOT2016.ckpt", "location of weight file")

FLAGS = flags.FLAGS

def getIoU(boxA, boxB):
    """return intersection-over-union of two bbox"""
    
    boxA = boxA[0]
    #boxB = boxB[0]
    
    # determine the (x, y)-coordinates of the intersection rectangle
    #print boxA, boxB
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    
    interArea = 0
    
    if xB - xA < 0 or yB - yA < 0:
        interArea = 0
    else:
        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)
    

 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


class ROLO_TF:
    def __init__(self, config, is_training):
        self.disp_console = FLAGS.debug #display the information
        self.display_step = FLAGS.debug #display step by step
        if self.disp_console:
            print("ROLO init")
        #get parameter from config
        self.restore_weights = config.restore_weights
        self.weights_file = FLAGS.weights_file
        self.num_steps = config.num_steps
        self.num_fts = config.num_fts
        self.num_targets = config.num_targets
        self.num_input = self.num_fts + self.num_targets
        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.keep_prob = config.keep_prob
        self.is_training = is_training
        self.num_layers = config.num_layers
        self.epoches = config.epoches
        self.hidden_size = config.hidden_size
        
        #define the input and output
        self.x = tf.placeholder("float32", [self.num_steps, self.num_input])
        self.istate = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.num_input]) #state & cell => 2x num_input
        self.y = tf.placeholder("float32", [self.num_steps, self.num_targets])
    
        # Define a regulizer
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=config.regularizer_param)
        # Define weights
        self.softmax_w = tf.get_variable(
        "softmax_w", [self.num_input, self.num_targets], dtype='float32', regularizer=self.regularizer)
        self.softmax_b = tf.get_variable("softmax_b", [1, self.num_targets], dtype='float32')
        
        #Load the object list
        self.gt_list, self.num_objects = reader.read_GT_list(FLAGS.GTList_path) #TODO: 4/5 for training, 1/5 for validation
        print("Loaded ground truth file", self.gt_list)
        
        if self.disp_console:
            print("Build Networks...")
        #log_file = open("output/trainging-step1-exp2.txt", "a") #open in append mode
        self.build_networks()
        
        if self.disp_console:
            print("Run epoch")
        self.run_epoch()

    def LSTM_single(self, name,  _X, _istate, _weights, _biases):
        
        # input shape: (batch_size, n_steps, n_input)
        #_X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # Reshape to prepare input to hidden activation
        #_X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        # _X = tf.split(0, self.num_steps, _X) # n_steps * (batch_size, num_input)
        # edited by Shrestha 29.06.2017
        #_X = tf.split(_X, self.num_steps , 0) # n_steps * (batch_size, num_input)

        #print("_X: ", _X)

        # cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)
        # Edited by Shrestha 30.06.17
        state_per_layer_list = tf.unstack(_istate, axis=0)
        rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])for idx in range(self.num_layers)])
        
        def LSTMcell():
            cell = tf.contrib.rnn.LSTMCell(self.num_input, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            return cell
        
        if self.is_training and self.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(LSTMcell(), output_keep_prob=self.keep_prob)
        else:
            def attn_cell():
                return LSTMcell()

        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.num_layers)], state_is_tuple=True)
        
        if self.is_training and self.keep_prob < 1: #don't need drop-out layer when testing
            _X = tf.nn.dropout(_X, self.keep_prob)
            
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        # state = _istate
        state = cell.zero_state(self.batch_size, tf.float32)
        #embedding =  tf.get_variable("embedding", [self.num_input, self.hidden_size], dtype=tf.float32)
        #inputs = tf.nn.embedding_lookup(embedding, _X)
        #inputs = tf.split(_X, self.num_steps, 1)
        inputs = _X
        #if self.is_training and self.keep_prob < 1: #don't need drop-out layer when testing
        #    inputs = tf.nn.dropout(inputs, self.keep_prob)
        # for step in range(self.num_steps):
            # outputs, state = tf.nn.rnn(cell, [_X[step]], state)
            # Edited by Shrestha 29.06.17
        with tf.variable_scope(tf.get_variable_scope()) as scope:  
            inputs = tf.unstack(inputs, num=self.num_steps, axis=0) 
            for i in range(self.num_steps):
                inputs[i] = tf.reshape(inputs[i], [1, self.num_input])
            outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=rnn_tuple_state, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
        
        output = tf.reshape(tf.stack(axis=0, values=outputs), [-1, self.num_input])
        
        bbox = tf.matmul(output, _weights) + _biases
        
        #state = tf.reshape(tf.stack(axis=0, values=state), [-1, self.num_input*2])
        
        self._final_state = state

        #print("output: ", outputs)
        #print("state: ", state)
        return bbox

    def build_networks(self):
        if self.disp_console: 
            print "Building ROLO graph..."

        # Build rolo layers
        self.lstm_module = self.LSTM_single('lstm_train', self.x, self.istate, self.softmax_w, self.softmax_b)
        #self.ious= tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, self.rolo_weights_file)
        if self.disp_console: 
            print "Loading complete!" + '\n'
            
    def loss_function(self):
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        pred = self.lstm_module
        self.pred_location = pred
        correct_prediction = tf.square(pred - self.y)
        loss = tf.reduce_mean(correct_prediction) + reg_term
    
        
        return loss
        
    def run_epoch(self):
        
        if self.disp_console:
            print('Graph Building')
        # Use rolo_input for LSTM training
        
        self.loss = self.loss_function()
        
        if (self.optimizer == "RMS"): #default
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss) # RMS Optimizer
        elif (self.optimizer == "Adam"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) # Adam Optimizer 
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
            
        training_set = 1#int(self.num_objects * 0.8)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.weights_file)
                if self.disp_console:
                    print "Loading weights complete!" + '\n'
            else:
                sess.run(init)
            
            _current_state = np.zeros((self.num_layers, 2, self.batch_size, self.num_input))
            
            if self.disp_console:
                print('Start Training Process')
            for epoch in range(0, self.epoches):
                if self.disp_console:
                    print "~~~~~~~~~Epoch: ", epoch
                self.is_training = True #use drop-out layer
                for training_id in range(training_set):
                    GT_data, bbox_data, fts_data = reader.read_concatenate_all(self.gt_list[training_id])
                    total_loss = 0
                    id = 0
                    training_iters = len(bbox_data) #frames containing same object
    
                    # Keep training until reach max iterations
                    while id  < training_iters - self.num_steps:
                        
                        #get single item of input and target
                        batch_xs, frame_id = reader.extractSingleSample(id, bbox_data, fts_data, self.num_steps, self.num_input)
                        batch_ys = reader.extractSingleGTBBox(frame_id, GT_data, self.num_steps)
    
                        # Reshape data to get 3 seq of 5002 elements
                        #batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                        #batch_ys = np.reshape(batch_ys, [self.batch_size, self.num_targets])
                        
                        if self.disp_console: 
                            print("Batch_ys: ", batch_ys)
    
                        cost, optimizer, location, _current_state = sess.run([self.loss, self.optimizer, self.pred_location, self._final_state]
                                            ,feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: _current_state})
                        if self.disp_console: 
                            print("ROLO Pred: ", location)
    
                        #sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                        
                        # Calculate batch loss
                        #loss = sess.run(self.loss, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                        if self.disp_console: 
                                print "Iter " + str(id*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(cost) #+ "{:.5f}".format(self.accuracy)
                        total_loss += cost
                        id += self.num_steps
    
                    #print "Optimization Finished!"
                    avg_loss = total_loss/id
                    if self.disp_console:
                        print "Avg loss: " + self.gt_list[training_id] + ": " + str(avg_loss)
                        
                #Run validation set
                error = 0.0
                itemCount = 0.0
                self.is_training = False #we don't use dropout layer for validation set
                for validation_id in range(training_set, self.num_objects):
                    GT_data, bbox_data, fts_data = reader.read_concatenate_all(self.gt_list[validation_id])
                    sampleId = 0
                    batch_xs, frame_id = reader.extractSingleSample(id, bbox_data, fts_data, self.num_steps, self.num_input)#we start from frame id
                    validation_iters = len(GT_data) #Note we use GT data here, since faster rcnn does not provide fts and bbox for each frame
                    for det in range(frame_id[0], validation_iters - self.num_steps):
                        next_batch_xs, frame_id = reader.extractSingleSample(sampleId, bbox_data, fts_data, self.num_steps, self.num_input) #FIXME: could be more efficient
                        batch_ys = reader.extractSingleGTBBox(range(det, det+self.num_steps), GT_data, self.num_steps)
                        
                        if (frame_id[0] == det):
                            batch_xs = next_batch_xs #if we have results from rcnn, use this one
                            sampleId = sampleId + 1 #then we go to next rcnn result
                        
                        #batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                        #batch_ys = np.reshape(batch_ys, [self.batch_size, self.num_targets])
                        
                        if self.disp_console: 
                            print("Val-set, Batch_ys: ", batch_ys)
                        
                        #get the estimated location
                        location= sess.run([self.pred_location] ,feed_dict={self.x: batch_xs, self.istate: _current_state}) 
                        
                        if self.disp_console: 
                            print("Val-set, pred: ", location)
                        
                        #compute IoU score
                        IoUScore = getIoU(location[0], batch_ys[0])
                        if self.disp_console: 
                            print ("Val-set, IoU: ", IoUScore)
                        if (IoUScore < 0.5):
                            error = error + 1;
                        itemCount = itemCount + 1
                        #id = id + 1
                if self.disp_console:
                    print "Evaluate ", itemCount, " frame, validation error is: ", float(error / itemCount)
            save_path = self.saver.save(sess, self.weights_file)
            if self.disp_console:
                print("Training finished, Model saved in file: %s" % save_path)
        #log_file.close()
        return
class TrainConfig(object):
    batch_size = 1
    hidden_size = 4100 #FIXME: what is hidden size
    keep_prob = 0.75 #drop-out rate, TODO: validation set should not use dropout layer
    num_layers = 2
    num_fts = 4096 #number of features
    num_targets = 4 #number of bbox
    num_input = num_fts + num_targets
    num_steps = 1 #the number of unrolled steps of LSTM
    learning_rate = 0.00001
    restore_weights = False
    optimizer = "RMS"
    regularizer_param = 0.01
    epoches = 100
    
class TestConfig(object):
    batch_size = 1
    hidden_size = 0 #FIXME: what is hidden size
    keep_prob = 0.5 #drop-out rate
    num_layers = 2
    num_fts = 4096 #number of features
    num_targets = 4 #number of bbox
    num_input = num_fts + num_targets
    num_steps = 1 #the number of unrolled steps of LSTM
    learning_rate = 0.00001
    restore_weights = False
    optimizer = "RMS"
    regularizer_param = 0.01
    epoches = 1
    
def get_config():
    if FLAGS.model == "train":
        return TrainConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    
    '''----------------------------------------main-----------------------------------------------------'''
def main():
    config = get_config()
    
    if FLAGS.model == "train":
        ROLO_TF(config, is_training = True)

if __name__=='__main__':
        main()