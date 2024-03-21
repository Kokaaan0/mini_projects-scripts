import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

data = pd.read_csv('/home/nemanja/Downloads/fetal_health.csv')

columns = ['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability', 'histogram_width',
       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
       'histogram_median', 'histogram_variance', 'histogram_tendency',
       ]

scale_X = StandardScaler()
x = pd.DataFrame(scale_X.fit_transform(data.drop(['fetal_health'],axis=1),),columns=columns)
y = data['fetal_health']


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

random_forest = RandomForestClassifier()
random_forest_tr = random_forest.fit(x_train,y_train)
predictions = random_forest_tr.predict(x_test)


print('Classification report Random Forest')
print(classification_report(y_test, predictions))

print('Confusion Matrix Random Forest')
print(confusion_matrix(y_test, predictions))

xgb_classifier = xgb.XGBClassifier(objective='multi:softprob')

xgb_classifier.fit(x_train,y_train-1)

xgb_pred = xgb_classifier.predict(x_test)
print('Confusion matrix XGB')
print(confusion_matrix(y_test,xgb_pred+1))

print('Classification report XGB')
print(classification_report(y_test,xgb_pred+1))





