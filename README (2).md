this project is an Predictive Maintenance of turbofan jet engines and The C-MAPSS software was used to simulate engine degradation. Four separate sets
of operational conditions and fault modes were simulated in four different ways. To
characterize fault progression, record numerous sensor channels. The Prognostics CoE
at NASA Ames provided the data set.
The main goal is to predict the remaining useful life (RUL) of each engine. RUL is
equivalent of number of flights remained for the engine after the last data point in the
test dataset.
The steps involved are :
 1. i have imported all the library such as numpy,pandas,matplot,seaborn.
 ```
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 %matplotlib inline
 ```
 2. converted the excel data to csv format using pandas.
 ```
 train=pd.read_csv('train1.csv' ,header=1)
 ```
 3. then i have performed data cleaning process,removed features having nan values ,outliers.
 ```
 train.dropna()
 ```
 4. removal of feature which is of only a single unique value because practically its impossile if only a single element 
 ```
 unwanted=[]
for i in train.select_dtypes(include=np.number):
    if train[i].nunique()==1:
        unwanted.append(i)
print(unwanted)
 ```
 5. using feature selection choosed only values which are highly correlated
 ```
 def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
 ```
 6. replace outliers
 ``` 
 z_scores = train.apply(lambda x: np.abs((x - x.mean()) / x.std()))

# set a threshold for the z-score
threshold = 3

# identify the outliers
outliers = z_scores > threshold
```
 7. mathematical approach of calculating RUL(dependent feature)
 ```
 train['RUL']=train['life']-train['time']
train.drop(['life'],axis=1,inplace=True)
 ```
 8. performed standardization on the data
 ```
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
 ```
 9. import all the librayries from scikit learn 
 ```
 from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

 ```
 9. predicting the output 
 ```
 linreg=LinearRegression()
linreg.fit(X_train_scaled,y_train)
y_pred=linreg.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred)
```
 10. calculation of accuracy of model using crossvalidation 
 ```
 training_score = cross_val_score(linreg, X_train, y_train, cv=5)
print("Algorithm: ", linreg, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
 ```
 11. pickled the model so that it can be used in future for cloud purpose.
 12. import all the lib in the flask 
 ```
 import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
```
13. assign name for my application 
```
application = Flask(__name__)
app=application
```
14. using GET and POST method to take info from user through the url directly 
```
@app.route('/predictdata',methods=['GET','POST'])
```
15. evaluation of the model 
```
new_data_scaled=standard_scaler.transform([[unit_number,time,operational_setting_1,operational_setting_2,sensor_measurement_3,sensor_measurement_4,sensor_measurement_6,sensor_measurement_7,sensor_measurement_8,sensor_measurement_9,sensor_measurement_11,sensor_measurement_12,sensor_measurement_13,sensor_measurement_15,sensor_measurement_17,sensor_measurement_20,sensor_measurement_21]])
        result=ridge_model.predict(new_data_scaled)
```
16. running flask app in server number of 5000
```
if __name__=="__main__":
    app.run(host="0.0.0.0")
```
 
 

