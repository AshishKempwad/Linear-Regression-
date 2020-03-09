import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score



class Weather:
    def _init_(self):
        self.weights=[]
        
    def train(self,path):
        import numpy as np
        import pandas as pd
        dataset=pd.read_csv(path)
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import r2_score
       
        import random
        x=dataset.iloc[:,:]
        X=x.drop(x.columns[4],axis=1)
        Y=dataset.iloc[:,4]
        Y=pd.DataFrame(Y)
        X1=X.drop(X.columns[[0, 1, 2, 6, 9]], axis = 1) 
#         print(X1)
        weights=[1,0,1,0,1,0]
        weights=np.asarray(weights)
        weights=weights.reshape(len(weights),1)
        scaler=MinMaxScaler()
        X1=scaler.fit_transform(X1)
        X1=pd.DataFrame(X1)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X1,Y, test_size=0.3,random_state=42)
        Bias_array=[1]*(len(X_train))
        X_train.insert(loc=0,column='bias',value=Bias_array)
        Bias_array1=[1]*(len(X_test))
        X_test.insert(loc=0,column='bias',value=Bias_array1)
        y_train=y_train.to_numpy()
        y_test=y_test.to_numpy()
        y_train=y_train.reshape(len(y_train),1)
        y_test=y_test.reshape(len(y_test),1)
        learning_parameter=0.01
        iters=10000
        for i in range(0,iters):
            error=X_train.dot(weights)-y_train
            cost_fun=(X_train.T.dot(error))
            weights=weights-(learning_parameter*cost_fun)*(1/(len(X_train)))
            self.weights=weights
        
        
    def predict(self,path):
        import numpy as np
        import pandas as pd
        dataset=pd.read_csv(path)
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import r2_score
        x=dataset.iloc[:,:]
        X=x.drop(x.columns[4],axis=1)
        Y=dataset.iloc[:,4]
        Y=pd.DataFrame(Y)
        X1=X.drop(X.columns[[0, 1, 2, 6, 9]], axis = 1) 
        scaler=MinMaxScaler()
        X1=scaler.fit_transform(X1)
        X1=pd.DataFrame(X1)
        Bias_array=[1]*(len(X1))
        X1.insert(loc=0,column='bias',value=Bias_array)
        X1=X1.to_numpy()
        new_weights=self.weights
        predict_labels=X1.dot(new_weights)
        print(predict_labels.shape)
        return predict_labels
        
        
        
