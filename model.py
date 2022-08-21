# Essentials
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
import pickle

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")



# Read in the dataset as a dataframe
housingdata_df = pd.read_csv('data_science_challenge_data.csv')

#Removing outliers from the target feature
#Finding the upper and lower quantiles of the price and removing the ones beyond them. 
upper_limit = housingdata_df['price'].quantile(0.99)
lower_limit = housingdata_df['price'].quantile(0.01)
new_df = housingdata_df[(housingdata_df['price'] <= upper_limit) & (housingdata_df['price'] >= lower_limit)]

#Removing outliers from the size feature
upper_limit_size = new_df['size'].quantile(0.99)
lower_limit_size = new_df['size'].quantile(0.01)
new_df = new_df[(new_df['size'] <= upper_limit_size) & (new_df['size'] >= lower_limit_size)]

new_df_copy = new_df.copy()

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(new_df_copy[['neighbourhood']]).toarray())
enc_df.columns = ['neighbourhood_0','neighbourhood_1','neighbourhood_2']
# merge with main df bridge_df on key values
new_df_copy = pd.merge(
    left=new_df_copy,
    right=enc_df,
    left_index=True,
    right_index=True,
)

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
new_df_copy['building_lbl'] = labelencoder.fit_transform(new_df_copy['building'])

new_df_copy = new_df_copy.drop(['neighbourhood','building'], axis = 1)

new_df_copy['bathrooms'] = new_df_copy['bathrooms'].fillna(new_df_copy['bathrooms'].mode()[0])

#Selecting all the features excluding target variable 'price'
X = new_df_copy.loc[:, new_df_copy.columns != 'price']
#Selecting target variable 'price'
y = new_df_copy.iloc[:,0]

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=10000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X, y)


# Saving model to disk
pickle.dump(model_xgb, open('model.pkl','wb'))

# Saving model to disk
pickle.dump(enc, open('enc.pkl','wb'))
# Saving model to disk
pickle.dump(labelencoder, open('labelencoder.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([['PLY', 980000, 1,2,'Building_163']]))

