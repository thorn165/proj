import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


start = time.time()
data = pd.read_csv('ProcessedDataProject.csv',header = 'infer')


start1 = time.time()

# number of previous games to take stats from
k = 25

length = len(data.index) - k

# Number of games it make feature vectors for
# set to length for entire set, takes 7400 seconds
numRow = 1000
Acolumn=['APts','AeFG','ARbd','AAst','APf']
Hcolumn=['HPts','HeFG','HRbd','HAst','HPf']
vector = data[['Winner']]
Awaydf = pd.DataFrame(columns=Acolumn)
Homedf = pd.DataFrame(columns=Hcolumn)
end1 = time.time()



corr = data.drop(['GameID','HID','AID'],axis=1)
corr = corr.corr()
corr = corr.filter(['Winner','HPts','APts'])
corr['CorrSum'] = abs(corr['Winner']) + abs(corr['HPts']) + abs(corr['APts'])

data = data.drop(['HftP','HTrn','HStl','HBlk','AftP','ATrn','AStl','ABlk'],axis = 1)

print()
print("#################################### #########")
print("######## Correlation Matrix ######## ## Sum ##")
print("#################################### #########")
print(corr)



start2 = time.time()
for i in range(0,numRow):
    homeTeam = data.get_value(i,'HID')
    awayTeam = data.get_value(i,'AID')
    
    kHome = data.loc[(data['HID']==homeTeam) & (data['GameID']>(i))] 
    kHome = kHome.head(k)
    kHome.drop(['GameID','HID','AID','APts','AeFG','ARbd','AAst','Winner','APf'],axis=1,inplace=True)
    avgHome = kHome.mean().round(2)
    avgHdf= pd.DataFrame([avgHome])
    Homedf = Homedf.append(avgHdf,ignore_index = True)
    
    kAway = data.loc[(data['AID']==awayTeam) & (data['GameID']>(i))] 
    kAway = kAway.head(k)
    kAway.drop(['GameID','HID','AID','HPts','HeFG','HRbd','HAst','Winner','HPf'],axis=1,inplace=True)
    avgAway = kAway.mean().round(2)
    avgAdf= pd.DataFrame([avgAway])
    Awaydf = Awaydf.append(avgAdf,ignore_index = True)
end2 = time.time()



start3 = time.time()
vector = vector.head(numRow)
vector = vector.join(Awaydf)
vector = vector.join(Homedf)
print()
print("########################################################################")
print("#################### Featrure Vectors For Each Game ####################")
print("########################################################################")
print(vector)

end3 = time.time()
split = 0.7

start4 = time.time()
Y = vector['Winner']
X = vector.drop('Winner',axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split, random_state = 1)





################# Decision Tree ###############################################
maxDepths = [2,3,4,5,6,7,8,8,10]
d_acc = []

index = 0

trainAcc = np.zeros(len(maxDepths))
testAcc = np.zeros(len(maxDepths))

for m in maxDepths:
    clf = tree.DecisionTreeClassifier(max_depth = m)
    clf = clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    d_acc.append(accuracy_score(Y_test, Y_predTest))
    index += 1

tree_acc = [max(d_acc)]

bagfit = plt.figure(1)
plt.plot(maxDepths,trainAcc,'ro--',maxDepths,testAcc,'bv--')
plt.title("DTree")
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
###############################################################################


################# Neighbors ###################################################
numNeighbors = [1, 5, 10,15,20,25,40]
knn_acc = []

index = 0

trainAcc = np.zeros(len(numNeighbors))
testAcc = np.zeros(len(numNeighbors))

for nn in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=nn)
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    knn_acc.append(accuracy_score(Y_test, Y_predTest))
    if accuracy_score(Y_test, Y_predTest) <= max(knn_acc):
        best_n = nn
    index += 1

tree_acc.append(max(knn_acc))

fitting = plt.figure(2)
plt.plot(numNeighbors,trainAcc,'ro--',numNeighbors,testAcc,'bv--')
plt.title("K-Neighbors")
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
###############################################################################


################# Linear ######################################################
numC = [2,5,10,15,25,50]
line_acc= []
index = 0

trainAcc = np.zeros(len(numC))
testAcc = np.zeros(len(numC))

for c in numC:
    clf = linear_model.LogisticRegression(C = c)
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    line_acc.append(accuracy_score(Y_test, Y_predTest))
    if accuracy_score(Y_test, Y_predTest) <= max(line_acc):
        best_c = c
    index += 1

tree_acc.append(max(line_acc))

fitting = plt.figure(3)
plt.plot(numC,trainAcc,'ro--',numC,testAcc,'bv--')
plt.title("Linear")
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Complexity')
plt.ylabel('Accuracy')


clf = linear_model.LogisticRegression(C = best_c)
clf.fit(X_train, Y_train)
Y_predTest = clf.predict(X_test)

scores = cross_val_score(clf, X, Y, cv=5)

print()
print('#####################################################')
print('############ Cross Validation of Linear #############')
print('#####################################################')
print("Linear Scores: ", scores.round(3))
print("Accuracy: %0.2f (+/-) %0.2f)" % (scores.mean(), scores.std() * 2))
print()


###############################################################################


################# SVC #########################################################
svc_acc= []
index = 0

trainAcc = np.zeros(len(numC))
testAcc = np.zeros(len(numC))

for c in numC:
    clf = SVC(C = c,kernel = 'linear')
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    svc_acc.append(accuracy_score(Y_test, Y_predTest))
    if accuracy_score(Y_test, Y_predTest) <= max(svc_acc):
        best_c = c
    index += 1

tree_acc.append(max(svc_acc))

fitting = plt.figure(4)
plt.plot(numC,trainAcc,'ro--',numC,testAcc,'bv--')
plt.title("SVM")
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Complexity')
plt.ylabel('Accuracy')

clf = SVC(C = nn,kernel = 'linear')
clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predTest = clf.predict(X_test)

scores = cross_val_score(clf, X, Y, cv=5)

print()
print('#################################################')
print('############ Cross Validation of SVC ############')
print('#################################################')
print("SVC Scores: ", scores.round(3))
print("Accuracy: %0.2f (+/-) %0.2f)" % (scores.mean(), scores.std() * 2))
print()

###############################################################################


############## Random Forest ##################################################
numEst = [2,5,10,15,20,25,50]
forest_acc = []

index = 0

trainAcc = np.zeros(len(numEst))
testAcc = np.zeros(len(numEst))

for e in numEst:
    clf = ensemble.RandomForestClassifier(n_estimators = e)
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    forest_acc.append(accuracy_score(Y_test, Y_predTest))
    index +=1
    
tree_acc.append(max(forest_acc))

forestfit = plt.figure(5)
plt.plot(numEst,trainAcc,'ro--',numEst,testAcc,'bv--')
plt.title("Random Forest")
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
###############################################################################


############## Bagging with ###################################################
bag_acc =[]
combo = []

index = 0

trainAcc = np.zeros(len(maxDepths))
testAcc = np.zeros(len(maxDepths))

#for m in maxDepths:
#    for n in numEst:
#        t = m,n
#        combo.append(t)
        
for m in maxDepths:
    clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=m),n_estimators = 20)
    clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    bag_acc.append(accuracy_score(Y_test, Y_predTest))
    index += 1

tree_acc.append(max(bag_acc))

bagfit = plt.figure(6)
plt.plot(maxDepths,trainAcc,'ro--',maxDepths,testAcc,'bv--')
plt.title("Bag")
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')

###############################################################################


############## Ada Boost #######################################################
ada_acc = []

index = 0

trainAcc = np.zeros(len(maxDepths))
testAcc = np.zeros(len(maxDepths))
        

for m in maxDepths:
        clf = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=m),n_estimators = 20)
        clf.fit(X_train, Y_train)
        Y_predTrain = clf.predict(X_train)
        Y_predTest = clf.predict(X_test)
        trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
        testAcc[index] = accuracy_score(Y_test, Y_predTest)
        ada_acc.append(accuracy_score(Y_test,Y_predTest))
        index += 1
    
tree_acc.append(max(ada_acc))

bagfit = plt.figure(7)
plt.plot(maxDepths,trainAcc,'ro--',maxDepths,testAcc,'bv--')
plt.title("Ada Boost")
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
###############################################################################


############ Compareing classifiers ###########################################
methods = ['Dtree','KNeigh','Linear','SVM','Forest','Bagging','Ada']

model_compare = plt.figure(8)
plt.title("Model comparison")
plt.bar([1.5,2.5,3.5,4.5,5.5,6.5,7.5], tree_acc)
plt.xticks([1.5,2.5,3.5,4.5,5.5,6.5,7.5], methods)

print()
print('##########################')
print('### Method ## Accuracy ###')
print('##########################')
print("   ",methods[0],":   ",tree_acc[0].round(3))
print("   ",methods[1],":  ",tree_acc[1].round(3))
print("   ",methods[2],":  ",tree_acc[2].round(3))
print("   ",methods[3],":     ",tree_acc[3].round(3))
print("   ",methods[4],":  ",tree_acc[4].round(3))
print("   ",methods[5],": ",tree_acc[5].round(3))
print("   ",methods[6],":     ",tree_acc[6].round(3))
###############################################################################


###################### Cross validation ######################################

clf = KNeighborsClassifier(n_neighbors= best_n)
scores = cross_val_score(clf, X, Y, cv=5)

print()
print('#############################################')
print('###### Cross Validation of K-Neighbors ######')
print('#############################################')
print("NN Scores: ", scores.round(3))
print("Accuracy: %0.2f (+/-) %0.2f)" % (scores.mean(), scores.std() * 2))
print()

##############################################################################

############################# Timings #########################################
end = time.time()
print("Split Value: ",split)
print("K Value:     ",k)
print("Time for first chunk: ", end1-start1)
print("For Loop time:        ", end2-start2)
print("Combining DataFrames: ", end3-start3)
print("Total Completion:     ", end-start)