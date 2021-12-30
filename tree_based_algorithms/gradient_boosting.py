import matplotlib.pyplot as plt # for data visualization purposes
import pandas as pd
import numpy as np
import seaborn as sns
#%matplotlib inline

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix

# to produce a classification report and a confusion matrix
from sklearn.ensemble import GradientBoostingClassifier

# loading training and testing data
train_data = pd.read_csv(r"D:\ML\Datasets\titanic_train.csv")
test_data = pd.read_csv(r"D:\ML\Datasets\titanic_test.csv")

y_train = train_data["Survived"] # setting the label data ( w h a t s being predicted based on the rest of the features)
train_data.drop(labels="Survived", axis=1, inplace=True ) # making it the index

# join the two datasets
full_data = train_data.append(test_data)

# drop columns that a r e n t relevant to the training process
drop_columns = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
full_data.drop(labels=drop_columns, axis=1, inplace=True)
full_data = pd.get_dummies(full_data, columns=["Sex"]) # converting text data into numerical
full_data.fillna(value=0.0, inplace=True ) # filling empty cells with the value 0.0

# splitting joint dataset into training and testing datasets (with the changes made above)
X_train = full_data.values[0:891] # 0 to 891 rows go into X_train
X_test = full_data.values[891:] # 891 onwards go into X_test
state = 12
test_size = 0.30 # 70% training dataset and 30% testing dataset // found out by 1 - test_size = 1 - 0.30 = 0.70 -> 70%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=state)
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1] # setting learning rates
# comparing c l a s s i f i e r s performance for different learning rates
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth =2, random_state=0) # can specify loss function
    gb_clf.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

# selecting 0.5 as the best learning rate observing accuracy for both training and validation
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5,
                                     max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print ("Confusion Matrix:") # printing confusion matrix using test values of Y and the predictive value of y
print (confusion_matrix(y_val, predictions))

# printing confusion matrix in the colored format seen below in output
cm = confusion_matrix(y_val, predictions)
cm
class_names=[0, 1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu",fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# printing classification report
print("Classification Report:")
print(classification_report(y_val, predictions))