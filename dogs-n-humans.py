from sklearn import tree

# [hair colour, opposable thumbs, cuteness/10]
features = [[False, 10], [True, 4], [False, 9], [True, 2]]
labels = ["Dog", "Human", "Dog", "Human"]

print("-- TRAINING DATA --\n--------")
for i in range(len(features)):
    print("Opposable thumbs: " + str(features[i][0]))
    print("Cuteness on a scale from 0-10: " + str(features[i][1]))
    print("Label: " + labels[i])
    print("--------")
print("-- END TRAINING DATA --\n")

print("Training...")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print("-- RESULT --")
testData = [[True, 1]]
print("Opposable thumbs: " + str(testData[0][0]))
print("Cuteness on a scale from 0-10: " + str(testData[0][1]))
print("Verdict: " + clf.predict(testData)[0])