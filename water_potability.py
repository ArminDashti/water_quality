import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv('c:/users/armin/desktop/water_potability.csv')

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imp_mean = imp_mean.fit_transform(df)

imputer = KNNImputer(n_neighbors=2, weights="uniform")
df_imputer = imputer.fit_transform(df)

df = pd.DataFrame(df_imp_mean, columns=[df.columns[i] for i in range(10)])
df = df.astype('int64')

y = df['Potability']
x = df.drop(['Potability'], axis=1)
# x = df[['Hardness', 'ph', 'Solids', 'Sulfate', 'Chloramines', 'Organic_carbon', 'Turbidity']]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# x = SelectKBest(chi2, k=7).fit_transform(x, y)


from sklearn.preprocessing import Normalizer

scaler = StandardScaler()
x = scaler.fit_transform(x)

# x = Normalizer().fit_transform(x)




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# pred = clf.predict(X_test)

# accuracy_score(y_test, pred)

from sklearn.svm import SVC
clf = SVC(gamma='scale').fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(y_test, pred)

from sklearn.metrics import f1_score
f1_score(y_test, pred, average='weighted')

# from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(random_state=0, min_samples_split=10).fit(X_train, y_train)
# pred = clf.predict(X_test)
# accuracy_score(y_test, pred)
#%%
