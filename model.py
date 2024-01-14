import pandas as pd, numpy as np, random
from sklearn.linear_model import LogisticRegression

# load
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# clean
train_df['isMale']=(train_df['Gender']=='Male').astype(int)
test_df['isMale']=(train_df['Gender']=='Male').astype(int)

one_hot_geo = pd.get_dummies(train_df['Geography'], prefix='Geo')
train_df = train_df.join(one_hot_geo)
one_hot_geo = pd.get_dummies(test_df['Geography'], prefix='Geo')
test_df = test_df.join(one_hot_geo)

# logistic regression
estimators=['CreditScore', 'Age', 'Tenure', 'Balance', 
             'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
             'EstimatedSalary','isMale',
             'Geo_France','Geo_Germany', 'Geo_Spain']

X = train_df[estimators]
Y = train_df['Exited']

model = LogisticRegression()
model.fit(X, Y)

# prediction
X_test = test_df[estimators]
y_pred = model.predict_proba(X_test)
prob_exited = y_pred[:,1]
pred = pd.Series(prob_exited, name='Exited').to_frame(name='Exited')

result_df=pd.concat([test_df['id'], pred], axis=1)

# write out
result_df.to_csv('out_pred.csv', index=False)