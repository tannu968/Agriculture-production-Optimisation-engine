
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact

data=pd.read_csv("E:\optimization agriculture production.csv")

print("shape of the dataset:",data.shape)

data.head()

data.isnull().sum()

data['label'].value_counts()

print("avgrage ratio of natrogen in the soil:{0:2f}".format(data['N'].mean()))
print("avgrage ratio of phosphorous in the soil:{0:2f}".format(data['P'].mean()))
print("avgrage ratio of potassium in the soil:{0:2f}".format(data['K'].mean()))
print("avgrage ratio of temperature in the soil:{0:2f}".format(data['temperature'].mean()))
print("avgrage ratio of humidity in the soil:{0:2f}".format(data['humidity'].mean()))
print("avgrage ratio of PH in the soil:{0:2f}".format(data['ph'].mean()))
print("avgrage ratio of Rainfall in the soil:{0:2f}".format(data['rainfall'].mean()))


# lets check the summary for each of the crop

@interact
def summary(crops=list(data['label'].value_counts().index)):
    x=data[data['label']==crops]
    print("------------------------------------------------------")
    print("statistics for nitrogen")
    print("minimum nitrigen required :",x['N'].min())
    print("Average nitrigen required :",x['N'].mean())
    print("maximum nitrigen required :",x['N'].max())
    print("--------------------------------------------------------")
    print('statistics for phosphorous')
    print("------------------------------------------------------")
    print("statistics for nitrogen")
    print("minimum phosphorous required :",x['P'].min())
    print("Average phosphorous required :",x['P'].mean())
    print("maximum phosphorous required :",x['P'].max())
    print("--------------------------------------------------------")
    print('statistics for potassium')
    print("------------------------------------------------------")
    print("minimum potassium required :",x['K'].min())
    print("Average potassium required :",x['K'].mean())
    print("maximum potassium required :",x['K'].max())
    print("--------------------------------------------------------")
    print('statistics for Temperature')
    print("------------------------------------------------------")
    print("statistics for Temperature")
    print("minimum Temperature required :",x['temperature'].min())
    print("Average Temperature required :",x['temperature'].mean())
    print("maximum Temperature required :",x['temperature'].max())
    print("--------------------------------------------------------")
    print("statistics for humidity")
    print("--------------------------------------------------------")
    print("minimum humidity required :",x['humidity'].min())
    print("Average humidity required :",x['humidity'].mean())
    print("maximum humidity required :",x['humidity'].max())
    print("--------------------------------------------------------")
    print("statistics for PH")
    print("--------------------------------------------------------")
    print("minimum PH required :",x['ph'].min())
    print("Average PH required :",x['ph'].mean())
    print("maximum PH required :",x['ph'].max())
    print("--------------------------------------------------------")
    print("statistics for Rainfall")
    print("--------------------------------------------------------")
    print("minimum Rainfall required :",x['rainfall'].min())
    print("Average Rainfall required :",x['rainfall'].mean())
    print("maximum Rainfall required :",x['rainfall'].max())
    print("--------------------------------------------------------")
    


#lets compare the avg requirement for each crops with avg conditions

@interact
def comare(conditions=['N','P','K','temperature','ph','humidity','rainfall']):
    print('Avgrage Values for',conditions,"is {0:.2f}".format(data[conditions].mean()))
    print("-----------------------------------------------")
    print("Rice :{0:.2f}".format(data[(data['label']=='rice')][conditions].mean()))
    print("Black Gram : {0:.2f}".format(data[data['label']=='blackgram'][conditions].mean()))
    print("Banana : {0:.2f}".format(data[data['label']=='banana'][conditions].mean()))
    print("Jute : {0:.2f}".format(data[data['label']=='jute'][conditions].mean()))
    print("Coconut : {0:.2f}".format(data[data['label']=='coconut'][conditions].mean()))
    print("Apple : {0:.2f}".format(data[data['label']=='apple'][conditions].mean()))
    print("Papaya : {0:.2f}".format(data[data['label']=='papaya'][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(data[data['label']=='muskmelon'][conditions].mean()))
    print("Grapes : {0:.2f}".format(data[data['label']=='grapes'][conditions].mean()))
    print("Watermelon : {0:.2f}".format(data[data['label']=='watermelon'][conditions].mean()))
    print("Kidney Beans : {0:.2f}".format(data[data['label']=='kidneybeans'][conditions].mean()))
    print("Mung Beans : {0:.2f}".format(data[data['label']=='mungbean'][conditions].mean()))
    print("Orange: {0:.2f}".format(data[data['label']=='orange'][conditions].mean()))
    print("Chick Peas : {0:.2f}".format(data[data['label']=='chickpea'][conditions].mean()))
    print("Cotton : {0:.2f}".format(data[data['label']=='cotton'][conditions].mean()))
    print("Maize : {0:.2f}".format(data[data['label']=='maize'][conditions].mean()))
    print("Cotton : {0:.2f}".format(data[data['label']=='cotton'][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[data['label']=='mothbeans'][conditions].mean()))


#lets make this funcation more intuitive

@interact
def compare(conditions=['N','P','K','temerature','ph','humidity','rainfall']):
    print("crops which require greater than average",conditions,'\n')
    print(data[data[conditions]>data[conditions].mean()]['label'].unique())
    print("--------------------------------------------------------")
    print("crop which require less than avgrage",conditions,'\n')
    print(data[data[conditions]<=data[conditions].mean()]['label'].unique())


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming data is a DataFrame that contains 'K', 'temperature', 'rainfall', and 'humidity' columns

plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 3)
sns.histplot(data['K'], kde=True, color='darkblue')
plt.xlabel('Ratio of Potassium', fontsize=12)
plt.grid()

plt.subplot(2, 4, 4)
sns.histplot(data['temperature'], kde=True, color='black')
plt.xlabel('Temperature', fontsize=12)
plt.grid()

plt.subplot(2, 4, 5)
sns.histplot(data['rainfall'], kde=True, color='grey')
plt.xlabel('Rainfall', fontsize=12)
plt.grid()

plt.subplot(2, 4, 6)  # Changed from 5 to 6 to avoid duplication
sns.histplot(data['humidity'], kde=True, color='lightgreen')
plt.xlabel('Humidity', fontsize=12)
plt.grid()

plt.suptitle('Distribution for Agricultural Conditions', fontsize=20)
plt.show()




## lets find out some interesting facts

print("some interesting patterns")
print("---------------------------------")
print("crops which requires very high ratio of nitrogen content in soil:",data[data['N']>120]['label'].unique())
print("crops which requires very high ratio of phosphorous content in soil:",data[data['P']>100]['label'].unique())
print("crops which requires very high ratio of potassium content in soil:",data[data['K']>200]['label'].unique())
print("crops which requires very high ratio of Rainfall :",data[data['rainfall']>200]['label'].unique())
print("crops which requires very high ratio of LOW Temperature :",data[data['temperature']<10]['label'].unique())
print("crops which requires very high ratio of High Temperature :",data[data['temperature']>40]['label'].unique())
print("crops which requires very high ratio of Low Humidity :",data[data['humidity']<20]['label'].unique())
print("crops which requires very high ratio of Low PH :",data[data['ph']<4]['label'].unique())
print("crops which requires very high ratio of high PH :",data[data['ph']>9]['label'].unique())




## lets understand which crops can only be grown in summer season,winter season and rainy season

print("summer crop")
print(data[(data['temperature']>30) & (data['humidity']>50)]['label'].unique())
print("-----------------------------------------------")
print("winter crops")
print(data[(data['temperature']>30) & (data['humidity']>30)]['label'].unique())
print("-----------------------------------------------")
print("Rainy crop")
print(data[(data['rainfall']>30) & (data['humidity']>30)]['label'].unique())




from sklearn.cluster import KMeans

#remove the labels column
x=data.drop(['label'],axis=1)

#selecting all the values of the data
x=x.values

#checking the shape
print(x.shape)


# Assuming x is your dataset and data contains your original data with a 'label' column

# Apply KMeans clustering
km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

# Create a DataFrame for the cluster results
y_means_df = pd.DataFrame(y_means, columns=['cluster'])
z = pd.concat([y_means_df, data['label']], axis=1)

# Check the clusters of each crop
print("Results After Applying K-Means Clustering Analysis\n")
print("Crops in the first cluster:", z[z['cluster'] == 0]['label'].unique())
print("------------------------------------------")
print("Crops in the second cluster:", z[z['cluster'] == 1]['label'].unique())
print("------------------------------------------")
print("Crops in the third cluster:", z[z['cluster'] == 2]['label'].unique())
print("------------------------------------------")
print("Crops in the fourth cluster:", z[z['cluster'] == 3]['label'].unique())




# lets split tha dataset for predictive modeling
y=data['label']
x=data.drop(['label'],axis=1)

print("shape of x:",x.shape)
print("shape of x:",y.shape)




# lets create training and testing sets for validation of results

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print("The shape of x train",x_train.shape)
print("The shape of x test",x_test.shape)
print("The shape of y train",y_train.shape)
print("The shape of y train",y_test.shape)




#lets create a predictive model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


from sklearn.metrics import confusion_matrix

#lets print the confusion matrix first
plt.rcParams['figure.figsize']=(10,10)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap='Wistia')
plt.title('confusion for Logistic Regression',fontsize=15)
plt.show()


#lets print the classification report also
from sklearn.metrics import accuracy_score, classification_report

cr=classification_report(y_test,y_pred)
print(cr)


data.head()


prediction=model.predict((np.array([[90,
                                    40,
                                    40,
                                    20,
                                    80,
                                    7,
                                    200]])))
print('The suggested crop for given climatic condition is :',prediction)

