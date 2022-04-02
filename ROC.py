
from sklearn.metrics import roc_curve, auc

#import torch.cuda
import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt

#torch.cuda.set_device(0)

#if GPU is available 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

ga = np.load("TestingGammaNew.npy", allow_pickle=True)
no = np.load("TestingnurtronNew.npy", allow_pickle=True)

ga = torch.Tensor(ga).to(device)
no = torch.Tensor(no).to(device)

#vars
gacount = 0
nocount = 0
BATCH_SIZE = 1
correctCount =0 
wrongCount =0
input_size = 248
wrongCount2 =0
counter1 = 0
counter2 = 0

DROPOUT1 = .2
DROPOUT2 = .2
DROPOUT3 = .2
#RNN structure
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,  num_layers1, num_layers2, num_layers3, num_classes):
        super(RNN, self).__init__()
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.num_layers3 = num_layers3
        self.hidden_size = hidden_size 

        self.drop1 = torch.nn.Dropout(DROPOUT1)  
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers1, batch_first=True , )

        self.drop2 = torch.nn.Dropout(DROPOUT2) 
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers2, batch_first=True , )

        self.drop3 = torch.nn.Dropout(DROPOUT3) 
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers3, batch_first=True , )

        self.fc = nn.Linear(hidden_size, num_classes)                                                
        
    def forward(self, x):
        h01 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size)
        c01 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size)

        h02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size)
        c02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size)

        h03 = torch.zeros(self.num_layers3, x.size(0), self.hidden_size) 
        c03 = torch.zeros(self.num_layers3, x.size(0), self.hidden_size) 
      
        out, _ = self.lstm1(x, (h01,c01)) 
        out = self.drop1(out)

        out, _ = self.lstm2(out, (h02,c02)) 
        out = self.drop2(out)

        out, _ = self.lstm3(out, (h03,c03)) 
        out = self.drop3(out)

        out = out[:, -1, :]
        out = self.fc(out)                
        return out

y_pred = []

#loads RNN weights and so on
model = torch.load("GRUModel.pth")

#ga pulse tester
model.eval()

i = 0
model.to(device)
with torch.no_grad():

    for i in (range(0, len(ga), BATCH_SIZE)):

        batch_X = ga[i:i+BATCH_SIZE]
        batch_X = batch_X.reshape(-1, 1, input_size)
        batch_X.to(device)
        output = model(batch_X)
       
       
        _, pred = torch.max(output, dim=1)

        y_pred.append(_)
        
        for i in pred:
            counter1 = counter1 + 1
            if i !=0:
                wrongCount = wrongCount + 1
               
            
               

        #if counter1 >= 70000:
            #break

#neutron pulse tester
model.eval()
i=0
with torch.no_grad():

    for i in (range(0, len(no), BATCH_SIZE)):

        batch_X = no[i:i+BATCH_SIZE]
        batch_X = batch_X.reshape(-1, 1, input_size)
        output = model(batch_X)

      

        _, pred = torch.max(output, dim=1)

        y_pred.append(_)
       

        for i in pred:
            counter2 = counter2 + 1
            if i ==0:
                wrongCount2 = wrongCount2 + 1
               
   

print(f"Accuracy for No pluse: {wrongCount2 / (counter2) * 100:.4f}%")
#print(f"Accuracy for ga pluse: {wrongCount / 169000 * 100:.4f}%")
print(f"Accuracy for ga pluse: {wrongCount / (counter1) * 100:.4f}%")

print("Wrong count for No pluse: ", wrongCount2)
print("Wrong count for ga pluse: ", wrongCount)
accuracy2 = wrongCount2 
accuracy = wrongCount 

wrongcountoverall = wrongCount2+ wrongCount
#overallacc = (accuracy2+accuracy) /239000
overallacc = (accuracy2+accuracy) /(counter1 + counter2)

print(f"Accuracy overall : {overallacc *100:.4f}%")
print("Wrong count overall: ", wrongcountoverall)

print("How many times it went through the GA loop: ", counter1) #just for checking to make sure the loop is setup right 
print("How many times it went through the NO loop: ", counter2)

print(wrongCount, "Na wrong out of ", counter1)
print(wrongCount2, "Na wrong out of ", counter2)

y_test1 = np.zeros(10000)
y_test2 = np.ones(10000)

y_test = np.concatenate((y_test1, y_test2))



nn_fpr, nn_tpr, nn_thresholds = roc_curve(y_test, y_pred)

print(nn_fpr)
print(nn_tpr)
print(nn_thresholds)





plt.plot(nn_fpr,nn_tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate' )
plt.show()


auc_hold = auc(nn_fpr, nn_tpr)
plt.plot(nn_fpr, nn_tpr, marker='.', label='Neural Network (auc = %0.3f)' % auc_hold)
#print(auc_hold)
plt.show()


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(nn_fpr, nn_tpr, label='Keras (area = {:.3f})'.format(auc_hold))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()
