from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, KFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
#from keras.datasets import mnist
#from numpy import reshape
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset and separating labels

train_data = pd.read_csv("D:\ML\Datasets\emails.csv")
y = train_data['spam']
train_data.drop(labels='spam', axis=1, inplace=True)

# Creating bag of words

vectorizer = CountVectorizer(stop_words=['subject'], max_features=500) 
                             #, strip_accents="unicode")
x = vectorizer.fit_transform(train_data.text)
features = vectorizer.get_feature_names() # to check feature names if required
x = x.toarray()
x.shape

tsne = TSNE(n_components=3, verbose=1, random_state=123, learning_rate=15, perplexity=50, n_iter=1000)
            #, method='exact')
z = tsne.fit_transform(x)

df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
df["comp-3"] = z[:,2]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 2), data=df).set(title="Spam email data T-SNE projection")

X = df[["comp-1", "comp-2", "comp-3"]].to_numpy()

# Specifying model and KFold parameters

model = XGBClassifier(n_estimators=50, scale_pos_weight=3, max_depth=8, learning_rate=0.3, verbosity=0, random_state=1
                      , reg_alpha=5, reg_lambda=0, use_label_encoder=False)
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Making Predictions

predictions = cross_val_predict(model, X, y, cv=cv)

# Printing confusion matrix in the colored format seen below in output

cm = confusion_matrix(y, predictions)
class_names = [0, 1] # names of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Creating heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Printing classification report

print("Classification Report:")
print(classification_report(y, predictions))