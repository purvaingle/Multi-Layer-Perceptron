import numpy as np
import sys
import csv
import pandas as pd

input_layer_size=2
hidden_layers_size=26 
output_layer_size=2
reg=0.01
lr=0.01
batchSize=32
epochs=120

def one_hot(Ls):
    encoded = []
    #print(Ls)
    for L in Ls:
        if L==1:
            #print(L)
            encoded.append([0, 1])
        else:
            #print(L)
            encoded.append([1, 0])
    return np.array(encoded)

'''def cross_entropy_loss(L, pred):
    sh = L.shape[1]
    C = np.sum(np.multiply(L, np.log(pred)) + np.multiply(1 - L, np.log(1 - pred)))*(-1 / sh) 
    return np.squeeze(C)'''

def relu(x):
    x1=np.where(x<0, 0, x)
    return x1
    
def softmax(x):
    val = []
    for i in x:
        exp = np.exp(i - np.max(i))
        a=exp / np.sum(exp, axis=0)
        val.append(a)
    return np.array(val)

class MultiLayerPercept:
    
    def __init__(self):
        self.NN = {'W1': np.random.randn(input_layer_size, hidden_layers_size),'W2': np.random.randn(hidden_layers_size, output_layer_size),
                   'b1': np.random.randn(hidden_layers_size),'b2': np.random.randn(output_layer_size)
        }
    
    def ModelTest(self, train_data, train_labels, test_data):
        self.mini_batch(train_data, train_labels)
        Q, pred_L = self.forwardProp(test_data)
        #print(pred_L)
        #predictions=np.copy(pred_L)
        pred_L = np.array([np.argmax(pred) for pred in pred_L])
        np.savetxt('test_predictions.csv', pred_L, delimiter=',', fmt='%d')
        #print(pred_L)
    
    def forwardProp(self, IP):
        W1=self.NN['W1']
        W2=self.NN['W2']
        b1=self.NN['b1']
        b2=self.NN['b2']
        
        a1 = np.dot(IP, W1) + b1
        self.NN['Z1'] = relu(a1)
        
        a2 = np.dot(self.NN['Z1'], W2) + b2
        self.NN['Z2'] = softmax(a2)
        
        return self.NN['Z1'], self.NN['Z2']
    
    def mini_batch(self, IP, L):
        epoch=-1
        while (epoch<epochs-1):
            epoch=epoch+1
            p = np.random.permutation(len(L))
            L=L[p]
            #print(p)
            IP=IP[p]
            for i in range(0, len(L), batchSize):
                bData=IP[i:i + batchSize] 
                bLabel = L[i:i + batchSize]
                self.gradients(bLabel, bData)
    
    
    def backProp(self, X, L):
        #print(len(L))
        W1=self.NN['W1']
        b1=self.NN['b1']
        W2=self.NN['W2']
        b2=self.NN['b2']
        
        Z1, Z2 = self.forwardProp(X)
        
        #dC/db2=L-Z2
        #dC/dW1=X.TI[(Z2−y)WT2]
        
        L = one_hot(L)
        db2Prime = L - Z2
        db2 = np.sum(db2Prime, axis=0)
        dW2 = np.dot(Z1.T, db2Prime)
        
        
        db1Prime = np.dot(db2Prime, W2.T)
        db1Prime[Z1 <= 0] = 0
        #db1Prime=np.where(Z1<=0, 0, Z1)
        db1 = np.sum(db1Prime, axis=0)
        dW1 = np.dot(X.T, db1Prime)
        
        dW2 = dW2 + reg * W2
        dW1 = dW1+ reg * W1
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    def gradients(self, L, X):
        G = self.backProp(X, L)
        for x in G.keys():
            #print(layer)
            #if (self.NN[x]=='W1' or self.NN[x]=='W2'):
            self.NN[x]= self.NN[x]+ lr * G[x]
        
def main() :
    train_data = pd.read_csv(sys.argv[1], header=None)
    train_labels = pd.read_csv(sys.argv[2], header=None)
    test_data = pd.read_csv(sys.argv[3], header=None)

    train_data = train_data.values
    train_labels = train_labels.values
    test_data = test_data.values
    
    '''train_data = np.loadtxt('spiral_train_data.csv', delimiter=',')
    train_labels=np.loadtxt('spiral_train_label.csv', delimiter=',')
    test_data = np.loadtxt('spiral_test_data.csv', delimiter=',')
    y_test=np.loadtxt('spiral_test_label.csv', delimiter=',')
    num=train_data.shape[0]'''
    
    neuralnetwork = MultiLayerPercept()
    neuralnetwork.ModelTest(train_data, train_labels, test_data)
    
if __name__=="__main__":
    main()