from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import sparse

def import_data():
    X_train_vec = sparse.load_npz('data\X_train_vec.npz')
    X_test_vec = sparse.load_npz('data\X_test_vec.npz')
    y_train_vec = np.load('data\y_train_vec.npy')
    y_test_vec = np.load('data\y_test_vec.npy')
    return (X_train_vec, y_train_vec, X_test_vec, y_test_vec)


if __name__ == "__main__":
    X_train_vec, y_train_vec, X_test_vec, y_test_vec = import_data()
    

    # define models
    #KNN_classifier=KNeighborsClassifier()
    #LREG_classifier=LogisticRegression(solver='liblinear')
    #DT_classifier = DecisionTreeClassifier()
    #MLP_classifier = MLPClassifier(max_iter=50)
    #SVC_classifier = SVC()
    #NB_classifier = GaussianNB()
    
    clf1 = LogisticRegression(random_state=123)
    clf2 = MLPClassifier(hidden_layer_sizes=(16,),max_iter=19, random_state=13, warm_start=False,learning_rate_init =0.001)
    clf3 = SVC()
    


    eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('MLP', clf2), ('svc', clf3)],
        voting='soft',
        n_jobs=1).fit(X_train_vec, y_train_vec)
    eclf2 = VotingClassifier(estimators=[
        ('lr', clf1), ('MLP', clf2), ('svc', clf3)],
        voting='soft',
        n_jobs=2).fit(X_test_vec, y_test_vec)
    
    pred=eclf2.predict(X_test_vec, y_test_vec)
    
    
    
    print (accuracy_score(pred,X_test_vec, y_test_vec))
    # === KNN:
    #build the parameter grid
    #KNN_grid = [{'n_neighbors': [1,3,5,7,9,11,13,15,17], 'weights':['uniform','distance']}]
    #build a grid search to find the best parameters
    #gridsearchKNN = GridSearchCV(KNN_classifier, KNN_grid, cv=5, verbose=True, n_jobs = -1) 
    #run the grid search
    #gridsearchKNN.fit(X_train_vec,y_train_vec)

    # === DT
    #build the parameter grid
    #DT_grid = [{'max_depth': [3,4,5,6,7,8,9,10,11,12],'criterion':['gini','entropy']}]
    #build a grid search to find the best parameters
    #gridsearchDT  = GridSearchCV(DT_classifier, DT_grid, cv=5, verbose=True, n_jobs = -1)
    #run the grid search
    #gridsearchDT.fit(X_train_vec,y_train_vec)

    # === LREG
    #build the parameter grid
    #LREG_grid = [ {'C':[0.1,0.5,1,1.5,2],'penalty':['l1','l2']}]
    #build a grid search to find the best parameters
    #gridsearchLREG  = GridSearchCV(LREG_classifier, LREG_grid, cv=5, verbose=True, n_jobs = -1)
    #run the grid search
    #gridsearchLREG.fit(X_train_vec,y_train_vec)

    # === MLP
    #build the parameter grid
    #MLP_grid = [ {'learning_rate_init':[0.1, 0.01, 0.001, 0.0001], 'activation':['logistic', 'tanh', 'relu'] }]
    #build a grid search to find the best parameters
    #gridsearchMLP = GridSearchCV(MLP_classifier, MLP_grid, cv=5, verbose=True, n_jobs = -1)
    #run the grid search
    #gridsearchMLP.fit(X_train_vec,y_train_vec)

    # === SVC
    #build the parameter grid
    #SVC_grid = [ {'C':[0.5, 1, 2, 3, 4], 'gamma':['scale', 'auto']}]
    #build a grid search to find the best parameters
    #gridsearchSVC = GridSearchCV(SVC_classifier, SVC_grid, cv=5, verbose=True, n_jobs = -1)
    #run the grid search
    #gridsearchSVC.fit(X_train_vec,y_train_vec)

    # === NB
    #build the parameter grid
    #NB_grid = [ {'var_smoothing':[1e-9, 1e-8, 1e-10]}]
    #build a grid search to find the best parameters
    #gridsearchNB = GridSearchCV(NB_classifier, NB_grid, cv=5, verbose=True, n_jobs = -1)
    #run the grid search
    #gridsearchNB.fit(X_train_vec.toarray(),y_train_vec)


    #print("The KNN best parameters:", gridsearchKNN.best_params_)
    #print("The KNN best score:", gridsearchKNN.best_score_)

    #print("The Decision Tree best parameters:", gridsearchDT.best_params_)
    #print("The DT best score:", gridsearchDT.best_score_)

    #print("The Logistic Regression best parameters:", gridsearchLREG.best_params_)
    #print("The LREG best score:", gridsearchLREG.best_score_)

    #print("The MultiLevel Perceptron best parameters:", gridsearchMLP.best_params_)
    #print("The MLP best score:", gridsearchMLP.best_score_)

    #print("The Support Vector Machine best parameters:", gridsearchSVC.best_params_)
    #print("The SVC best score:", gridsearchSVC.best_score_)

    #print("The Naive BAyes best parameters:", gridsearchNB.best_params_)
    #print("The Naive Bayes best score:", gridsearchNB.best_score_)
    
    


