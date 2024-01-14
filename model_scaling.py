import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# load
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# clean
train_df['isMale'] = (train_df['Gender'] == 'Male').astype(int)
test_df['isMale'] = (test_df['Gender'] == 'Male').astype(int)

one_hot_geo = pd.get_dummies(train_df['Geography'], prefix='Geo')
train_df = train_df.join(one_hot_geo)
one_hot_geo = pd.get_dummies(test_df['Geography'], prefix='Geo')
test_df = test_df.join(one_hot_geo)

# Define estimators
estimators = ['CreditScore', 'Age', 'Tenure', 'Balance', 
              'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
              'EstimatedSalary', 'isMale',
              'Geo_France', 'Geo_Germany', 'Geo_Spain']

# Extract features and target
X_train = train_df[estimators]
Y_train = train_df['Exited']
X_test = test_df[estimators]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
y_pred = model.predict_proba(X_test_scaled)
prob_exited = y_pred[:, 1]
pred = pd.Series(prob_exited, name='Exited').to_frame(name='Exited')

# Prepare the result dataframe
result_df = pd.concat([test_df['id'], pred], axis=1)

# Write out
result_df.to_csv('out_pred_scaled.csv', index=False)