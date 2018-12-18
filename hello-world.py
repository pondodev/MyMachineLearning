from sklearn import tree

# Training data
# Smooth = 1 Bumpy = 0
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# Apple = 0 Orange = 1
labels = [0, 0, 1, 1]

# Train with previously defined training data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Test to see if it can predict what our input is based off training
print(clf.predict([[150, 0]]))