import pandas as pd
from scipy.io import arff
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

data, meta = arff.loadarff('./heart-c.arff')

df = pd.DataFrame(data)

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['cp'] = le.fit_transform(df['cp'])
df['fbs'] = le.fit_transform(df['fbs'])
df['restecg'] = le.fit_transform(df['restecg'])
df['exang'] = le.fit_transform(df['exang'])
df['slope'] = le.fit_transform(df['slope'])
df['thal'] = le.fit_transform(df['thal'])
df['num'] = le.fit_transform(df['num'])

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

X = df_imputed.drop('num', axis=1)  # Drop the target variable
kmeans = KMeans(n_clusters=5, random_state=0)
df_imputed['cluster'] = kmeans.fit_predict(X)

print(df_imputed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_imputed['age'], y=df_imputed['trestbps'], hue=df_imputed['cluster'], palette='Set2', s=100,
                alpha=0.7)
plt.title('KMeans Clustering of Heart Disease Data')
plt.xlabel('Age')
plt.ylabel('Trestbps (Resting Blood Pressure)')
plt.legend(title='Cluster')
plt.show()
