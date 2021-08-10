import os
import numpy as np
from timeit import default_timer as timer
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nnet_data_submitted import *

class nnet(object):
    def __init__(self, NN_architecture):
        self.dataDim = NN_architecture['dataDim']
        self.loss_choice = NN_architecture['loss_choice']
        self.transfer_func = NN_architecture['transfer_func']
        self.layer_dim = NN_architecture['layer_dim']
        self.W_init = NN_architecture['W_init']
        self.labels = np.array(NN_architecture['labels'])
        GPU_memory_fraction = NN_architecture['GPU_memory_fraction']
        GPU_which = NN_architecture['GPU_which']
        Tensorflow_randomSeed = NN_architecture['Tensorflow_randomSeed']
        self.dtype = NN_architecture['dtype']        
        self.depth = len(self.layer_dim)
        self.showdevice = NN_architecture['show_device']
                
        # reset graph     
        tf.reset_default_graph()
        if Tensorflow_randomSeed is not None:
            tf.set_random_seed(Tensorflow_randomSeed)
                
        # create graph, either using CPU or 1 GPU
        if GPU_which is not None:
            with tf.device('/device:GPU:' + str(GPU_which)):
                self._create_graph()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_memory_fraction)    
            config = tf.ConfigProto(gpu_options = gpu_options,\
                                    allow_soft_placement=True, log_device_placement=self.showdevice)
        else:
            self._create_graph()            
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=self.showdevice, \
                                device_count = {'GPU': 0})
        
        # initialize tf variables & launch the session
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        
        # to save model
        self.saver = tf.train.Saver()    
    
    
    #--------------------------------------------------------------------------
    # Create the graph
    #        
    def _create_graph(self):
        self.x = tf.placeholder(self.dtype, [None, self.dataDim[0], self.dataDim[1], self.dataDim[2]])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.learning_rate = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        
        self._create_nnet()
        self._create_loss_optimizer()                    
    
    #--------------------------------------------------------------------------
    # Create the neural net graph
    #    
    def _create_nnet(self):     
        x = self.x
        self.W = []
        for layer in range(1,self.depth+1):            
            # Reshape if needed
            if (layer==1):
                d_in, _ = self._layer_dim(layer)
                x = tf.reshape(x, (-1, d_in-1))
                x = tf.pad(x, tf.constant([[0, 0], [0, 1]]) , constant_values=1.0)
            
            # Applying weight
            W = self._make_weight(layer)            
            self.W.append(W)
            if layer>1:                
                d_in, _ = self._layer_dim(layer)
                mul = 1.0/d_in
            else:
                mul = 1.0
            x = tf.matmul(x, W)*mul
            
            # Nonlinearity
            x = self._transfer(x, layer)             
            
        self.yhat = x     
        
    #--------------------------------------------------------------------------
    # Return the dimension for the 'layer'-th layer
    #        
    def _layer_dim(self, layer):    
        if layer==1:
            d_in = np.prod(self.dataDim) + 1 # plus 1 for bias in first layer
        else:                
            d_in = self.layer_dim[layer-1][0]
        d_out = self.layer_dim[layer-1][1]
        return d_in, d_out
    
    #--------------------------------------------------------------------------
    # Initialization of the weights, biases
    #
    def _weight_init(self, layer):
        if self.W_init['scheme']=='Gaussian':
            d_in, d_out = self._layer_dim(layer)
            if layer==1:
                std = self.W_init['params']['std'][layer-1]/np.sqrt(d_in)
                mean = 0.0
            else:
                std = self.W_init['params']['std'][layer-1]
                mean = self.W_init['params']['mean'][layer-1]
            return tf.random.normal(shape=[d_in, d_out], mean=mean, stddev=std)        
        elif self.W_init['scheme']=='external':
            return self.W_init['overload'][layer-1]
        else:            
            raise NameError('Initialization scheme not available!')
        
    def _make_weight(self, layer):
        val = self._weight_init(layer)
        return tf.Variable(val, trainable=True, dtype=self.dtype)        
    
    #--------------------------------------------------------------------------
    # Define the nonlinearity
    #
    def _transfer(self, x, layer):
        return self._nonlinearity(x, self.transfer_func[layer-1])    
        
    def _nonlinearity(self, x, transfer_func):
        if transfer_func=='tanh':
            return tf.tanh(x)
        else:
            return x        
        
    #--------------------------------------------------------------------------
    # Create the loss and the optimizer
    #
    def _create_loss_optimizer(self):
        self.loss = self._create_loss()       
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        
        # compute gradients for special params and process them
        grads_and_vars = []            
        gv = opt.compute_gradients(self.loss, self.W)
        for i in range(1,self.depth+1):
            grads_and_vars.append((self._process_W_grad(gv[i-1][0], i), gv[i-1][1]))
        
        self.optimizer = opt.apply_gradients(grads_and_vars) 
    
    def _create_loss(self):
        if self.loss_choice=='huber':
            tmp = self.yhat - self.y
            cond = tf.less_equal(tf.abs(tmp), tf.constant(1.0))
            cost = tf.reduce_mean(tf.where(cond, 0.5*tf.square(tmp), tf.abs(tmp) - 0.5))
        else:            
            raise NameError('loss_choice not available!')
        return cost
                
    def _process_W_grad(self, grad, layer):                        
        if layer==1:
            d_in, _ = self._layer_dim(layer+1)
            mul = d_in
        elif layer==self.depth:
            d_in, d_out = self._layer_dim(layer)
            mul = d_in*d_out
        else:
            dim1, _ = self._layer_dim(layer)
            dim2, _ = self._layer_dim(layer+1)
            mul = dim1*dim2
        return grad*mul    
    
    #--------------------------------------------------------------------------
    # Auxilliary functions
    #        
    def _compute_error(self, y, pred): 
        y = np.squeeze(y)
        pred = np.squeeze(pred)        
        num = y.shape[0]
        newy = np.array([np.argmin([np.sum((self.labels[i] - y[cnt])**2) for i in range(len(self.labels))]) for cnt in range(num)])
        newpred = np.array([np.argmin([np.sum((self.labels[i] - pred[cnt])**2) for i in range(len(self.labels))]) for cnt in range(num)])
        error = np.mean(newy!=newpred)
        return error    
    
    #--------------------------------------------------------------------------
    # Public methods
    #
    def fit(self, x, y, learning_rate):
        _, loss = self.sess.run((self.optimizer, self.loss), \
                                feed_dict={self.x: x, self.y: y, \
                                           self.learning_rate: learning_rate,
                                           self.is_training: True})
        return loss
    
    def predict(self, x, y=None, batch_size=100):
        if y is None:
            ind = 0
            pred = np.zeros((x.shape[0],1))
            while ind < x.shape[0]:
                temp = x[ind:(ind+batch_size)]
                temppred = self.sess.run(self.yhat, feed_dict={self.x: temp, self.is_training: False})
                pred[ind:(ind+batch_size)] = temppred
                ind += temp.shape[0]
            return pred
        else:            
            ind = 0
            pred = np.zeros((x.shape[0],1))
            loss = 0
            while ind < x.shape[0]:
                temp = x[ind:(ind+batch_size)]
                tempy = y[ind:(ind+batch_size)]
                temppred, temp_loss = self.sess.run((self.yhat, self.loss), \
                                                    feed_dict={self.x: temp, self.y: tempy, self.is_training: False})
                pred[ind:(ind+batch_size)] = temppred
                ind += temp.shape[0]
                loss += temp_loss*temp.shape[0]
            loss /= x.shape[0]                             
            error = self._compute_error(y, pred)
            return pred, loss, error        
    
    def get_weights(self):
        return self.sess.run(self.W, feed_dict={self.is_training: False})
    
    def save_model(self, path_to_saved_model):
        self.saver.save(self.sess, path_to_saved_model)
        
    def load_model(self, path_to_saved_model):
        self.saver.restore(self.sess, path_to_saved_model)
    
    
#-----------------------------------------------------------------------------------------------------
# Simulation of the neural net
#        
class nnet_simul(object):
    def __init__(self, params):
        self.NN = params['neural_net']
        self.data = params['data']
        self.SGD = params['SGD']                                
        self.statsCollect = params['statsCollect']  
        self.depth = len(self.NN['layer_dim'])
        self.stats = {}
        self.trainpred = {}
        self.subsample = {}        
                    
    #--------------------------------------------------------------------------
    # Generate the data
    #    
    def _generate_data_module(self):
        self.data['format'] = 'row features'
        nn_data = nnet_data(self.data) 
        return nn_data
        
    def _generate_data(self, numData, whichSet='train'):
        return self.nnet_data.get_data(numData, whichSet=whichSet)
    
    #--------------------------------------------------------------------------
    # Create the neural net
    #    
    def _generate_nnet(self, NN_architecture=None):
        if NN_architecture is None:
            NN_architecture = self._get_NN_architecture()
        return nnet(NN_architecture)
    
    def _get_NN_architecture(self):        
        NN_architecture = self.NN
        NN_architecture['dataDim'] = self.data['dataDim']
        NN_architecture['labels'] = self.data['data_structure']['labels']
        return NN_architecture
        
    #--------------------------------------------------------------------------
    # Collect statistics
    #    
    def _statistics(self, numMonteCarlo):
        # basic stats
        X, y, _ = self._generate_data(numMonteCarlo, whichSet='test')
        _, loss_test, error_test = self.nnet.predict(x=X, y=y)    
        X, y, _ = self._generate_data(numMonteCarlo, whichSet='train')
        _, loss_train, error_train = self.nnet.predict(x=X, y=y)                
        stats = dict(
            loss_test = loss_test,           
            loss_train = loss_train,
            error_test = error_test, 
            error_train = error_train
        )   
        return stats
    
    #--------------------------------------------------------------------------
    # Perform training with SGD
    #    
    def _SGD_update(self, iteration):        
        batch_size = self.SGD['batchSize']
        x, y, _ = self._generate_data(batch_size, whichSet='train')
        lr = self.SGD['stepSize']
        self.nnet.fit(x, y, lr)
        return lr
            
    def _SGD_run(self, iter_start=1):
        num = self.statsCollect['numMonteCarlo']        
        
        # run SGD
        time = timer()    
        if self.statsCollect['is_verbose']:
            print('Iter | Time (min) | Learning rate | Train loss | Test loss | Train error | Test error')
        for iteration in range(iter_start, iter_start+self.SGD['iteration_num']):
            lr = self._SGD_update(iteration)            
            if iteration in self.statsCollect['output_schedule']:
                self._update_stats(iteration, num)                
                time = timer() - time  
                if self.statsCollect['is_verbose']:
                    print('%08d' % (iteration), 
                          "|",  '%.3f' % (time/60),
                          "|", '%.3e' % (lr),
                          "|", '%.3e' % (self.stats[iteration]['loss_train']),
                          "|", '%.3e' % (self.stats[iteration]['loss_test']),
                          "|", '%.3e' % (self.stats[iteration]['error_train']),
                          "|", '%.3e' % (self.stats[iteration]['error_test'])
                         )
                time = timer()
            if iteration in self.statsCollect['savemodel_schedule']:
                path = self.statsCollect['savemodel_path']
                header = self.statsCollect['savemodel_header']                
                self.save_model(os.path.join(path, header + '__' + str(iteration)))
            if iteration in self.statsCollect['save_prediction_schedule']:
                self._save_prediction(iteration)
            if iteration in self.statsCollect['subsample_schedule']:
                self._subsample(iteration)
                
                                           
    def _update_stats(self, iteration, numMonteCarlo):
        self.stats[iteration] = self._statistics(numMonteCarlo)
        
    def _save_prediction(self, iteration):        
        pred, id_ = self._predict_whole_set('train')
        self.trainpred[iteration] = {'pred': pred, 'identifier': id_}
    
    def _predict_whole_set(self, whichSet, obj=None):
        X, _, id_ = self._generate_data(np.Inf, whichSet)
        if obj is None:
            pred = self.nnet.predict(x=X, y=None)
        else:
            pred = obj.predict(x=X, y=None)
        ind = np.argsort(id_)                
        id_ = id_[ind]
        pred = pred[ind]
        return pred, id_
    
    def _subsample(self, iteration):
        self.subsample[iteration] = {}
        params = self.statsCollect['subsample_params']
        sample_sizes = params['sample_sizes']
        seeds = params['seeds']        
        
        W_inf = self.get_weights()
        for N in sample_sizes:
            self.subsample[iteration][N] = {}
            for seed in seeds:               
                rng = np.random.default_rng(seed)
                W_samples = []
                for layer in range(1, len(W_inf)+1):
                    if layer==1:
                        samples_prev = rng.integers(low=0, high=W_inf[layer-1].shape[1], size=N)
                        W_samples.append(W_inf[layer-1][:, samples_prev])
                    elif layer==len(W_inf):
                        W_samples.append(W_inf[layer-1][samples_prev, :])
                    else:
                        samples_curr = rng.integers(low=0, high=W_inf[layer-1].shape[1], size=N)
                        W_samples.append(W_inf[layer-1][samples_prev, :][:, samples_curr])
                        samples_curr = samples_prev
                NN_architecture = self._get_NN_architecture()
                for i in range(len(NN_architecture['layer_dim'])):
                    if i==0:
                        NN_architecture['layer_dim'][i][1] = N
                    elif i==len(NN_architecture['layer_dim'])-1:
                        NN_architecture['layer_dim'][i][0] = N
                    else:
                        NN_architecture['layer_dim'][i] = [N, N]                
                NN_architecture['show_device'] = False
                NN_architecture['Tensorflow_randomSeed'] = None # important, so that Tensorflow seed of the main nnet is not reset
                NN_architecture['W_init']['scheme'] = 'external'
                NN_architecture['W_init']['overload'] = W_samples
                obj = self._generate_nnet(NN_architecture)
                pred_train, id_train = self._predict_whole_set(whichSet='train', obj=obj)
                self.subsample[iteration][N][seed] = {'pred_train': pred_train,
                                                      'id_train': id_train}
                
    
    #--------------------------------------------------------------------------
    # Public methods
    #        
    def generate_nnet(self):
        self.nnet = self._generate_nnet()
        self.nnet_data = self._generate_data_module()
    
    def run(self, iter_start=1, nnet_ready=False):
        if not nnet_ready:
            self.generate_nnet()
        self._SGD_run(iter_start)
    
    def collect_stats(self):
        return self.stats
    
    def get_nnet(self):
        return self.nnet

    def generate_data(self, numData, whichSet='train'):
        return self._generate_data(numData, whichSet=whichSet)
    
    def predict(self, X):
        return self.nnet.predict(X)    
    
    def get_weights(self):
        return self.nnet.get_weights()
    
    def save_model(self, path_to_saved_model):
        self.nnet.save_model(path_to_saved_model)
        
    def load_model(self, path_to_saved_model):
        self.nnet.load_model(path_to_saved_model)        
    
    def get_trainpred(self):
        return self.trainpred
    
    def get_subsample_results(self):
        return self.subsample
    
    def plot_evolution(self):
        import matplotlib.pyplot as plt        
        loss_test = []
        error_test = []
        loss_train = []
        error_train = []
        for i in self.statsCollect['output_schedule']:
            loss_test.append(self.stats[i]['loss_test'])
            error_test.append(self.stats[i]['error_test'])
            loss_train.append(self.stats[i]['loss_train'])
            error_train.append(self.stats[i]['error_train'])
        fig, ax = plt.subplots(1,4, figsize=(12,3))
        ax[0].semilogx(self.statsCollect['output_schedule'], loss_train)
        ax[0].grid(color='k', linestyle='--', linewidth=0.5) 
        ax[0].set_title('loss_train')
        
        ax[1].semilogx(self.statsCollect['output_schedule'], error_train)        
        ax[1].grid(color='k', linestyle='--', linewidth=0.5) 
        ax[1].set_title('error_train')
        
        ax[2].semilogx(self.statsCollect['output_schedule'], loss_test)
        ax[2].grid(color='k', linestyle='--', linewidth=0.5) 
        ax[2].set_title('loss_test')
        
        ax[3].semilogx(self.statsCollect['output_schedule'], error_test)        
        ax[3].grid(color='k', linestyle='--', linewidth=0.5) 
        ax[3].set_title('error_test')
        plt.show()        