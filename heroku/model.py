from xgboost.sklearn import XGBRegressor
from sklearn import linear_model
import numpy as np

class my_model:
    def __init__(self,d):
        self.linear_reg = linear_model.Ridge()
        self.xgb_reg = XGBRegressor(max_depth=7)
        self.d=d
    def fit(self,X,y):
        self.linear_reg.fit(X[:,0].reshape(-1, 1),y)
        self.l_reg_res = self.linear_reg.predict(X[:,0].reshape(-1, 1))
        self.xgb_reg.fit(X[:,1:],y-self.l_reg_res)
        X_nn = np.hstack([X,self.xgb_reg.predict(X[:,1:]).reshape(-1,1),self.l_reg_res.reshape(-1,1)])
        return X_nn
       
    def predict(self,X):
        if isinstance(X[0,-1],str):
            for i in range(X.shape[0]):
                X[i,-1]=self.d[X[i,-1]]
            X = X.astype(np.float64,copy=False)

        X_nn_final = np.hstack([X,self.xgb_reg.predict(X[:,1:]).reshape(-1,1),self.linear_reg.predict(X[:,1].reshape(-1, 1)).reshape(-1,1)])
        return X_nn_final