import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class nnet_data(object):
    def __init__(self, params):
        self.data_choice = params['data_choice']
        self.struct = params['data_structure']
        self.dataDim = params['dataDim']
        self.format = params['format']
        
        if params['Numpy_randomSeed'] is not None:
            np.random.seed(params['Numpy_randomSeed']) 
                
        self._generate_struct()
            
    def generate_struct(self):        
        self._generate_struct()
        
    def get_data(self, numData, whichSet='train'):
        X, y, id_ = self._generate_data(numData, whichSet=whichSet)
        X, y = self._change_format(X, y, _format=self.format)
        return X, y, id_
    
    #---------------------------------------- Generate data
    def _generate_struct(self):        
        if self.data_choice == 'mnist':
            classes = self.struct['classes']
            labels = self.struct['labels']
            numTrain = self.struct['number_training_samples']
            X_train, X_val, X_test, y_train, y_val, y_test, i_train, i_val, i_test, \
            numTrain, numVal, numTest, dataDim = self._mnist_import(numTrain=numTrain, classes=classes)            
            self.X_train, self.y_train, self.i_train = self._get_classes(X_train, y_train, i_train, classes, labels)
            self.X_val, self.y_val, self.i_val = self._get_classes(X_val, y_val, i_val, classes, labels)
            self.X_test, self.y_test, self.i_test = self._get_classes(X_test, y_test, i_test, classes, labels)    
            self.shufflecounter_train = 0
            self.shufflecounter_test = 0
            self.shufflecounter_val = 0
            del X_train, X_val, X_test
        else:
            raise NameError('data_choice not available!')

    def _generate_data(self, numData, whichSet='train'):
        if self.data_choice == 'mnist':
            if whichSet=='train':
                num = self.X_train.shape[0]
                numData = min(numData, num)
                if self.shufflecounter_train+numData < num:
                    ind = np.arange(start=self.shufflecounter_train, stop=self.shufflecounter_train+numData)
                    self.shufflecounter_train += numData
                else:
                    perm = np.arange(num)
                    np.random.shuffle(perm)
                    self.X_train = self.X_train[perm]
                    self.y_train = self.y_train[perm]
                    self.i_train = self.i_train[perm]
                    ind = np.arange(numData)
                    self.shufflecounter_train = numData                         
                X = self.X_train
                y = self.y_train
                id_ = self.i_train
            elif whichSet=='test':
                num = self.X_test.shape[0]
                numData = min(numData, num)
                if self.shufflecounter_test+numData < num:
                    ind = np.arange(start=self.shufflecounter_test, stop=self.shufflecounter_test+numData)
                    self.shufflecounter_test += numData
                else:
                    perm = np.arange(num)
                    np.random.shuffle(perm)
                    self.X_test = self.X_test[perm]
                    self.y_test = self.y_test[perm]
                    self.i_test = self.i_test[perm]
                    ind = np.arange(numData)
                    self.shufflecounter_test = numData                         
                X = self.X_test
                y = self.y_test
                id_ = self.i_test
            elif whichSet=='val':
                num = self.X_val.shape[0]
                numData = min(numData, num)
                if self.shufflecounter_val+numData < num:
                    ind = np.arange(start=self.shufflecounter_val, stop=self.shufflecounter_val+numData)
                    self.shufflecounter_val += numData
                else:
                    perm = np.arange(num)
                    np.random.shuffle(perm)
                    self.X_val = self.X_val[perm]
                    self.y_val = self.y_val[perm]
                    self.i_val = self.i_val[perm]
                    ind = np.arange(numData)
                    self.shufflecounter_val = numData                         
                X = self.X_val
                y = self.y_val
                id_ = self.i_val
            else:
                raise NameError('not train/test/val set!')
            X = X[ind]
            labelDim = np.size(self.struct['labels'][0])
            y = np.reshape(y[ind], (numData, labelDim))
            id_ = id_[ind]            
        else:            
            raise NameError('data_choice not available!')
        return X, y, id_
    
    #---------------------------------------- Data sets         
    
    def _mnist_import(self, numTrain=60000, classes=[0,1,2,3,4,5,6,7,8,9]):
        classes = np.array(classes).flatten()
        
        (train_data, train_labels), (X_test, y_test) = tf.keras.datasets.mnist.load_data() # not one-hot labels        
        train_data = np.array(train_data)/255.0
        train_labels = np.array(train_labels)
        X_test = np.array(X_test)[..., np.newaxis]/255.0
        y_test = np.array(y_test)                        
        
        ind = np.isin(train_labels, classes)
        train_data = train_data[ind]
        train_labels = train_labels[ind]
        ind = np.isin(y_test, classes)
        X_test = X_test[ind]
        y_test = y_test[ind]
        
        train_identifier = np.arange(train_data.shape[0])
        i_test = np.arange(X_test.shape[0])        
        
        numVal = max(train_data.shape[0] - numTrain, 0)
        ind = np.random.permutation(train_data.shape[0])
        train_data = train_data[ind]
        train_labels = train_labels[ind]   
        train_identifier = train_identifier[ind]
        X_val = train_data[numTrain:][..., np.newaxis]
        X_train = train_data[0:numTrain][..., np.newaxis]        
        y_val = train_labels[numTrain:]
        y_train = train_labels[0:numTrain]        
        i_val = train_identifier[numTrain:]
        i_train = train_identifier[0:numTrain]
        
        X_train = self._do_range_normalize(X_train, [-1.0, 1.0], [0.0, 1.0])
        X_val = self._do_range_normalize(X_val, [-1.0, 1.0], [0.0, 1.0])
        X_test = self._do_range_normalize(X_test, [-1.0, 1.0], [0.0, 1.0])
        
        numTrain = X_train.shape[0]
        numVal = X_val.shape[0]
        numTest = X_test.shape[0]
        dataDim = X_train.shape[1:]
                         
        X_train, y_train, i_train = self._shuffle(X_train, y_train, i_train)
        X_val, y_val, i_val = self._shuffle(X_val, y_val, i_val)
        X_test, y_test, i_test = self._shuffle(X_test, y_test, i_test)
        return X_train, X_val, X_test, y_train, y_val, y_test, i_train, i_val, i_test, numTrain, numVal, numTest, dataDim        


    #---------------------------------------- Misc
    def _change_format(self, X, y, _format='column features'):
        if _format=='row features':
            return X, y
        elif _format=='column features':
            return X.T, y.T            

    def _get_classes(self, X, y, id_, classes, labels):        
        if len(labels) != len(classes):
            raise NameError('Error: number of classes differs from number of labels!!!')
        numClass = len(classes)
        labelDim = np.size(labels[0])
        for cnt in range(numClass):
            ind = [i for i in range(len(y)) if y[i] in classes[cnt]]
            if cnt==0:
                X_new = X[ind]
                y_new = np.tile(labels[0], (len(ind), 1))
                id_new = id_[ind]
            else:
                X_new = np.append(X_new, X[ind], axis=0)
                y_new = np.append(y_new, np.tile(labels[cnt], (len(ind), 1)), axis=0)
                id_new = np.append(id_new, id_[ind], axis=0)
        X_new, y_new, id_new = self._shuffle(X_new, y_new, id_new)
        return X_new, y_new, id_new
    
    def _shuffle(self, X, y, id_): 
        if X.shape[0]>0:
            ind = np.random.permutation(range(X.shape[0]))
            return X[ind], y[ind], id_[ind]
        else:
            return X, y, id_

    def _do_range_normalize(self, X, range_normalize=None, original_range=None):    
        if range_normalize is not None:
            a, b = np.min(range_normalize), np.max(range_normalize)
            if original_range==None:                
                c, d = np.min(X), np.max(X)
            else:
                c, d = np.min(original_range), np.max(original_range)
            m = (b-a)*1.0/(d-c)        
            p = b - m*d
            X = X*m + p
        return X
    