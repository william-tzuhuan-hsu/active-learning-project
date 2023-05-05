import random
import numpy as np
import xgboost as xg
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans


class Results:
    def __init__(self) -> None:
        self.MSEs=[]
        self.CVs=[]

    def MSE(self):
       MSE=np.vstack(self.MSEs)
       mean=np.mean(MSE,axis=0)
       std=np.std(MSE,axis=0)
       return mean, std
    
    def CV(self):
        CV=np.vstack(self.CVs)
        mean=np.mean(CV,axis=0)
        std=np.std(CV,axis=0)
        return mean, std



# Simulate runs the active learning protocol for n_rounds, to reduce the effects brought by randomness. Within each round of simulation, the three query methods were performed.
# Simulate returns two lists of test errors and cross validation scores for each query method.
def Simulate(df, n_rounds, n_initial, n_toadd,batchsize):
    result_rand=Results()
    result_div=Results()
    result_uns=Results()


    for i in range(n_rounds):
        print("Round:", i)
        random.seed(i)
        X = df.iloc[:,:-1].to_numpy()
        Y = df.iloc[:,-1].to_numpy()


        # get the indicies of the first n_initial observations that were randomly chosen
        index_init = np.random.choice(range(len(X)), size=n_initial, replace=False)
        # save the remaining data points to be sampled from later 
        index_pool = np.setdiff1d(range(len(X)),index_init)

        # assign the data points to the training and test arrays 
        X_train = X[index_init]
        Y_train = Y[index_init]
        X_test = X[index_pool]
        Y_test = Y[index_pool]

        print("Running Random Sampling")
        MSE, CV = ActiveLearn(X_train, Y_train, X_test, Y_test, n_toadd, batchsize, "random")
        result_rand.MSEs.append(MSE)
        if CV != None:
           result_rand.CVs.append(CV)


        print("Running Diversity Sampling")
        MSE, CV = ActiveLearn(X_train, Y_train, X_test, Y_test, n_toadd, batchsize, "Diversity Based Sampling")
        result_div.MSEs.append(MSE)
        if CV != None:
           result_div.CVs.append(CV)

        print("Running Uncertainty Sampling")
        MSE, CV = ActiveLearn(X_train, Y_train, X_test, Y_test, n_toadd, batchsize, "Batch Uncertainty Sampling")
        result_uns.MSEs.append(MSE)
        if CV != None:
           result_uns.CVs.append(CV)

    return result_rand, result_div, result_uns




# ActiveLearn is a function that simulates one round of active learning.
# The inputs of ActiveLearn:
# X_train, y_train, X_test and y_test: X,Y train and test data, should be numpy arrays.
# n_toadd is the number of instances to be observed.
# querystrat is the strategy used for retrieving data points from the pool. The current available options are 'random', 'Diversity Based Sampling' and 'Batch Uncertainty Sampling'.
# batchsize is the number of instances added to training at each round.
# the function returns the test mean squared errors and cross validation scores at each round as lists.

def ActiveLearn(X_train, Y_train, X_test, Y_test, n_toadd, batchsize, querystrat): 
    model= xg.XGBRegressor(objective ='reg:squarederror',n_estimators = 10, seed = 42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    MSEerror=[mean_squared_error(Y_test, y_pred)]

    initial_model_error= cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    CVscores=[np.mean(np.abs(initial_model_error))]
    
    # from the testset each time select (batchsize) points to add to training set, remove the points from the remianing test data, predict the model performace, then teach it again
    for i in range(n_toadd//batchsize):
    
        # random sampling method 
        if querystrat=="random":
            # randomly sample points to be added
            indices = np.random.choice(range(len(X_test)), size=batchsize, replace=False)

        # diversity based sampling method 
        elif querystrat == "Diversity Based Sampling": 
            cluster_num = 3

            # Fit k-means clustering model to the remaining unobserved test data
            kmeans = KMeans(n_clusters=cluster_num,n_init=10).fit(X_test)
            indices = []
            n=batchsize//cluster_num #This is the number of points sampled from each cluster
            for k in range(cluster_num):
                cluster_indices = np.where(kmeans.labels_ == k)[0]
                if k==cluster_num-1:
                    # If batchsize cannot be divided evenly across clusters, add the leftover part to the last cluster
                    n+=batchsize%cluster_num
                random_index = list(np.random.choice(cluster_indices, size=n, replace=False))
                indices+=random_index

        elif querystrat == "Batch Uncertainty Sampling":
            # Get the prediction of each estimator from the XGboost model
            individual_preds=[]
            booster=model.get_booster()
            for tree_ in booster:
                individual_preds.append(tree_.predict(xg.DMatrix(X_test)))
            individual_preds=np.vstack(individual_preds)

            #Compute the variance of predications made for each point in X_test
            pred_var=np.var(individual_preds,axis=0)
            sorted_indices = np.argsort(pred_var)
        
            # Slice the sorted indices to get the last n indices (the points that cause the biggest variance)
            indices = sorted_indices[-(batchsize):]
        else: 
            raise Exception('wrong query strategy')

        # add the points to the training data 
        X_train = np.append(X_train, [X_test[i] for i in indices], axis=0)
        Y_train = np.append(Y_train, [Y_test[i] for i in indices], axis=0)
        # delete the points from the test data
        X_test = np.delete(X_test, indices, axis=0)  
        Y_test = np.delete(Y_test, indices, axis=0)

        model= xg.XGBRegressor(objective ='reg:squarederror',n_estimators = 10, seed = 42)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        MSEerror.append(mean_squared_error(Y_test, y_pred))

        error= cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        CVscores.append(np.mean(np.abs(error)))

    return MSEerror, CVscores