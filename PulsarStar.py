import pandas as pd
import numpy as np
train_df= pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/project-5/pulsar-star-prediction-train.csv")
test_df= pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/project-5/pulsar-star-prediction-test.csv")

print('train_df', train_df)
print('test_df', test_df)
print("Thue number of rows and columns in training dataset are", train_df.shape)
print("The number of rows and columns in the testing dataset are", test_df.shape)

# Check for the missing values in the 'train_df' DataFrame.
print(train_df.isnull())
print(train_df.isnull().sum())

# Check for the missing values in the 'test_df' DataFrame.
print(test_df.isnull())
print(test_df.isnull().sum())

# Count of the '0' and '1' classes in the 'train_df
print(train_df['target_class'].value_counts())
# Count of the '0' and '1' classes in the 'test_df' DataFrame.
print(test_df['target_class'].value_counts())

#Feature Variables Extraction
x_train= train_df.iloc[:, 1:]
print('x_train', x_train.head())
x_test= test_df.iloc[:, 1:]
print('x_test', x_test.head())

#Target Variables Extraction
y_train= train_df.iloc[:, :1]
print('y_train', y_train.head())
y_test= test_df.iloc[:, :1]
print('y_train', y_test.head())

#XGBoostClassifier
import xgboost as xg
from sklearn.metrics import confusion_matrix,classification_report
model=xg.XGBClassifier() 
model.fit(x_train, y_train)
pred=model.predict(x_test)
print('y_test_pred', pred)
print('confusion matrix', confusion_matrix(y_test,pred))
precision= 444/ (444+ 33)
print("the precision is", precision)
recall = 444/(444+82)
print("The recalll is", recall)
f1_score= 2*((precision*recall)/ (precision+recall))
print("the f1 score is", f1_score)
print(classification_report(y_test, pred))