import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

file_path = "heart-c.arff"
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df['cp'] = label_encoder.fit_transform(df['cp'])
df['fbs'] = label_encoder.fit_transform(df['fbs'])
df['restecg'] = label_encoder.fit_transform(df['restecg'])
df['exang'] = label_encoder.fit_transform(df['exang'])
df['slope'] = label_encoder.fit_transform(df['slope'])
df['thal'] = label_encoder.fit_transform(df['thal'])
df['num'] = label_encoder.fit_transform(df['num'])

print("Original Data:")
print(df.head())

print(df.columns)

feature_columns = [df.columns[0], df.columns[3]]
df_selected = df[feature_columns]

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_selected), columns=feature_columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed)

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df_imputed['cluster'] = kmeans.fit_predict(X_scaled)

print("Clustered Data:")
print(df_imputed)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df_imputed.iloc[:, 0],
    y=df_imputed.iloc[:, 1],
    hue=df_imputed['cluster'],
    palette='Set2',
    s=100
)
plt.title('KMeans Clustering (Features pressure and age)')
plt.xlabel('Age')
plt.ylabel('resting blood pressure')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
