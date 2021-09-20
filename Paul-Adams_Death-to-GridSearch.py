# Paul Adams
# Homework 4: Death to Grid Search
from sklearn.model_selection import KFold, ParameterGrid, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np, time
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import warnings
import pickle

warnings.filterwarnings('ignore')

class DeathToGridSearch:
    '''
    This class runs a random grid search with cross-validation and provides scores,
    plots, and the best models as dicts and callable functions. It also appends the
    best models to the original grid search dict. Current model configuration requires
    X to be numeric.
    '''

    def __init__(self, X, y, random_state):
        self.random_state = random_state
        self.X = X
        self.y = y
        self.accuracy_scorer = make_scorer(accuracy_score)
        self.f1 = make_scorer(f1_score, average='macro')
        self.output_hypers_f1 = {}
        self.output_hypers_acc = {}
        self.clf_hypers = [
            {'model': RandomForestClassifier(random_state = 42),
             'hyperparameters':{'n_estimators': np.linspace(10, 150, dtype='int'),
                                'criterion': ['gini', 'entropy'],
                                'max_depth': np.linspace(10, 100, dtype='int'),
                                'min_samples_split': np.linspace(2, 100, 50, dtype='int'),
                                'min_samples_leaf': np.linspace(2, 100, 50, dtype='int'),
                                'max_features': ['auto', 'sqrt', 'log2']}
            },
            {'model': LogisticRegression(random_state = 42),
             'hyperparameters':{'penalty':['l2'],
                                'C':np.geomspace(0.001, 10, num=30),
                                'solver':['lbfgs','saga']}
            },
            {'model':KNeighborsClassifier(),
             'hyperparameters':{'n_neighbors':np.arange(2, 10, 1),
                                'algorithm':['auto', 'ball_tree', 'kd_tree'],
                                'weights':['uniform','distance'],
                                'metric':['euclidean','minkowski','manhattan']}
            },
            {'model':MLPClassifier(random_state = 42),
             'hyperparameters':{'hidden_layer_sizes':(np.linspace(1, 200, 4, dtype='int'),),
                                'activation':['logistic','tanh','relu'],
                                'solver':['lbfgs', 'sgd', 'adam'],
                                'batch_size':['16','32','64'],
                                'learning_rate':['constant','invscaling'],
                                'learning_rate_init':np.geomspace(0.001, 100, num=300)}
            }
        ]

    def BestF1(self, X=None, y=None, clf_hyper=None, f1=None):
        '''
        This function runs the random search tuned against F1 score and
        reports, plots and saves the output
        '''
        X = self.X
        y = self.y
        f1 = self.f1
        output_hypers_f1 = self.clf_hypers.copy()
        clf_hyperF1 = self.clf_hypers.copy()
        random_state = self.random_state

        models=[]
        scores=[]
        model_namez=[]
        model_scorez=[]
        best_paramz=[]
        best_scorez=[]
        best_mod=[]

        #////////////////////////////////////////////////////#
        # Begin Randomized Grid Search with Cross-Validation #
        #////////////////////////////////////////////////////#

        for models in clf_hyperF1: # for each model in the dictionary of models and hyperparameters
            hyper_grids = ParameterGrid(models['hyperparameters']) # obtain the set of hyperparameters
            rf_clf = models['model'] # and obtain the model (direct function)

            # run randomized search
            random_search_CV_start_time = time.time()
            n_iter_search = 50
            random_search_CV = RandomizedSearchCV(rf_clf, models['hyperparameters'], scoring=f1,
                                         n_iter = n_iter_search, random_state=random_state,
                                         n_jobs=6)

            # seach
            random_search_CV.fit(X, y)

            model_namez.append(random_search_CV.best_estimator_.__class__.__name__)
            model_scorez.append(round(random_search_CV.best_score_,4))
            random_search_CV_end_time = time.time()
            duration = random_search_CV_end_time-random_search_CV_start_time

            best_scorez.append(random_search_CV.best_score_)
            scores.append(random_search_CV.cv_results_)
            best_paramz.append(random_search_CV.best_params_)
            best_mod.append(random_search_CV.best_estimator_)

        max_index = best_scorez.index(max(best_scorez))
        best_mod_f1 = best_mod[best_scorez.index(max(best_scorez))]

        #/////////////////////////////////////////////////////#
        # End of Randomized Grid Search with Cross-Validation #
        #/////////////////////////////////////////////////////#

        #***********************************************************#
        #### Setting up plots for F1 statistic model comparisons ####
        #***********************************************************#

        scores_final = []
        models_final = []
        model_names = []
        legend_names=[]
        scorez_dict = {}

        for idx in scores:
            scores_final.append(idx['mean_test_score'])

        for models in clf_hyperF1:
            models_final.append(models['model'])

        for i in np.arange(len(models_final)):
            model_names.append(models_final[i].__class__.__name__)

        models_and_results = list(zip(model_names, scores_final))

        for i in np.arange(len(models_and_results)):
            legend_names.append(models_and_results[i][0])

        for i in np.arange(len(models_and_results)):
            plt.plot(models_and_results[i][1], linewidth=3)
            plt.title("Model Performance over Random Search Iteration")
            plt.ylabel("Average Cross-Fold F1 Score")
            plt.xlabel("Random Search Iteration")
            plt.grid(linewidth=1, axis='y')
            plt.xlim(0,(n_iter_search-1))

        plt.legend(legend_names, bbox_to_anchor=(1.05, 0.75))
        plt.savefig('Models_by_F1.png', bbox_inches="tight")

        #***********************************************************#
        ###################### End of F1 plots ######################
        #***********************************************************#

        #############################################################################
        ############ From model strings, build functions to save in dict ############
        #############################################################################

        if model_namez[max_index] == 'KNeighborsClassifier':
            top_model = model_namez[max_index] + "()"
        else:
            top_model = model_namez[max_index] + "(random_state = 42)"
            top_model = top_model.replace("'","")

        ref = eval(top_model)
        best_dict = {'model':ref,
                    'hyperparameters':best_paramz[max_index]}

        #############################################################################
        ############# End of converting model strings to function names #############
        #############################################################################

        scorez_dict = dict(zip(model_namez, model_scorez))
        max_value = max(scorez_dict.values())
        max_key = max(scorez_dict, key=scorez_dict.get)

        # Print the best model, F1 score, and model hyperparameters:
        print("Based on the F1 Statistic:\n\nBest Model F1 Score: {}\nBest Model: {}\nBest {}'s Hyperparameters: {}\n\n".format(
            max_value
            , max_key
            , max_key
            , best_paramz[max_index]))
        output_hypers_f1.append(best_dict)

        # To save best model with original models and hyperparameters:
        with open("output_hypers_f1.pickle", "wb") as f:
            pickle.dump(output_hypers_f1, f)

        # to save the BEST model, by F1, as a dict:
        with open("best_f1_model.pickle", "wb") as f:
            pickle.dump(best_dict, f)

        # to save the BEST model, by F1, as a callable function:
        with open("best_mod_f1.pickle", "wb") as f:
            pickle.dump(best_mod_f1, f)

        # Return a plot of the F1 scores and a dictionary of the models and their scores
        return plt, dict(scorez_dict)
    # plot the F1 scores
    plt.show

    def BestAccuracy(self, X=None, y=None, clf_hyper=None, accuracy_scorer=None):
        '''
        This function runs the random search tuned against accuracy and
        reports, plots and saves the output
        '''
        X = self.X
        y = self.y
        accuracy_scorer = self.accuracy_scorer
        output_hypers_acc = self.clf_hypers.copy()
        clf_hyperAcc = self.clf_hypers.copy()
        random_state = self.random_state

        models=[]
        scores=[]
        model_namez=[]
        model_scorez=[]
        best_paramz=[]
        best_scorez=[]
        best_mod=[]

        #////////////////////////////////////////////////////#
        # Begin Randomized Grid Search with Cross-Validation #
        #////////////////////////////////////////////////////#

        for models in clf_hyperAcc:
            hyper_grids = ParameterGrid(models['hyperparameters'])
            rf_clf = models['model']

            # run randomized search
            random_search_CV_start_time = time.time()
            n_iter_search = 20
            random_search_CV = RandomizedSearchCV(rf_clf, models['hyperparameters'], scoring=accuracy_scorer,
                                         n_iter = n_iter_search, random_state=random_state,
                                         n_jobs=6)

            # seach
            random_search_CV.fit(X, y)

            model_namez.append(random_search_CV.best_estimator_.__class__.__name__)
            model_scorez.append(round(random_search_CV.best_score_,4))
            random_search_CV_end_time = time.time()
            duration = random_search_CV_end_time-random_search_CV_start_time

            best_scorez.append(random_search_CV.best_score_)
            scores.append(random_search_CV.cv_results_)
            best_paramz.append(random_search_CV.best_params_)
            best_mod.append(random_search_CV.best_estimator_)

        max_index = best_scorez.index(max(best_scorez))
        best_mod_acc = best_mod[best_scorez.index(max(best_scorez))]

        #/////////////////////////////////////////////////////#
        # End of Randomized Grid Search with Cross-Validation #
        #/////////////////////////////////////////////////////#


        #***********************************************************#
        #### Setting up plots for F1 statistic model comparisons ####
        #***********************************************************#

        scores_final = []
        models_final = []
        model_names = []
        legend_names=[]
        scorez_dict = {}

        for idx in scores:
            scores_final.append(idx['mean_test_score'])

        for models in clf_hyperAcc:
            models_final.append(models['model'])

        for i in np.arange(len(models_final)):
            model_names.append(models_final[i].__class__.__name__)

        models_and_results = list(zip(model_names, scores_final))

        for i in np.arange(len(models_and_results)):
            legend_names.append(models_and_results[i][0])

        for i in np.arange(len(models_and_results)):
            plt.plot(models_and_results[i][1], linewidth=3)
            plt.title("Model Performance over Random Search Iteration")
            plt.ylabel("Average Cross-Fold Accuracy Score")
            plt.xlabel("Random Search Iteration")
            plt.grid(linewidth=1, axis='y')
            plt.xlim(0,(n_iter_search-1))

        plt.legend(legend_names, bbox_to_anchor=(1.05, 0.75))
        plt.savefig('Models_by_Accuracy.png', bbox_inches="tight")

        #***********************************************************#
        ###################### End of F1 plots ######################
        #***********************************************************#

        #############################################################################
        ############ From model strings, build functions to save in dict ############
        #############################################################################

        if model_namez[max_index] == 'KNeighborsClassifier':
            top_model = model_namez[max_index] + "()"
        else:
            top_model = model_namez[max_index] + "(random_state = 42)"
            top_model = top_model.replace("'","")
        # Convert the string value for the best model to an evaluatable value (an sklearn model, in this case)
        ref = eval(top_model)
        best_dict = {'model':ref,
                    'hyperparameters':best_paramz[max_index]}

        #############################################################################
        ############# End of converting model strings to function names #############
        #############################################################################

        scorez_dict = dict(zip(model_namez, model_scorez))
        max_value = max(scorez_dict.values())
        max_key = max(scorez_dict, key=scorez_dict.get)
        print("Based on the Accuracy:\n\nBest Model Accuracy Score: {}\nBest Model: {}\nBest {}'s Hyperparameters: {}\n\n".format(
            max_value
            , max_key
            , max_key
            , best_paramz[max_index]))
        output_hypers_acc.append(best_dict)

        # To save best model with original models and hyperparameters:
        with open("output_hypers_acc.pickle", "wb") as f:
            pickle.dump(output_hypers_acc, f)

        # to save the BEST model, by Accuracy, as a dict:
        with open("best_accuracy_model.pickle", "wb") as f:
            pickle.dump(best_dict, f)

        # to save the BEST model, by Accuracy, as a callable function:
        with open("best_mod_acc.pickle", "wb") as f:
            pickle.dump(best_mod_acc, f)

        return plt, dict(scorez_dict)

    plt.show

    def get_models(self):
        '''
        This function gets the best models, by accuracy and f1,
        and returns them as a callable function
        '''
        with open('best_mod_f1.pickle', 'rb') as f:
            f1_model = pickle.load(f)

        with open('best_mod_acc.pickle', 'rb') as f:
            acc_model = pickle.load(f)

        return f1_model, acc_model

    def get_best_model_dicts(self):
        '''
        This function gets the best models, by accuracy and f1,
        and returns them as a dict with models and hypers
        '''
        with open('best_f1_model.pickle', 'rb') as f:
            f1_model_dict = pickle.load(f)

        with open('best_accuracy_model.pickle', 'rb') as f:
            acc_model_dict = pickle.load(f)

        return f1_model_dict, acc_model_dict

    def get_updated_model_dicts(self):
        '''
        This function gets two versions of the original model dict - one version
        with the best accuracy-scored model appended and one with the best
        f1-scored model appended
        '''
        with open('output_hypers_f1.pickle', 'rb') as f:
            clf_dict_with_updated_f1_model = pickle.load(f)

        with open('output_hypers_acc.pickle', 'rb') as f:
            clf_dict_with_updated_acc_model = pickle.load(f)

        return clf_dict_with_updated_f1_model, clf_dict_with_updated_acc_model


######################################################################################################################
######################################################################################################################
##################                             Using the class and functions                        ##################
######################################################################################################################
######################################################################################################################

# Load the dataset
data = load_wine()
X = data['data']
y = data['target']

# Call the class
gridsdeath = DeathToGridSearch(X, y, random_state=42)

# Invoke the F1 search
gridsdeath.BestF1()

# Invoke the Accuracy search
gridsdeath.BestAccuracy()

# Generate the callable functions
f1_model, acc_model = gridsdeath.get_models()

### Visualize
acc_model
f1_model

'''
The multi-layer perceptron performed the worst. Using the hyperparameter settings the searcher iterated
through, MLP only outperformed on other model - the KNN classifier - once out of 50 random searches. During
most iterations scoring both F1 and Accuracy, the MLP scored NaN, indicating poor randomly selected hyper-
parameters. The MLP would need more training than the other models, undoubtedly. The fact the wine dataset
only has 178 instances across 13 features could be one reason why it performed this way. With more samples,
output could improve when the randomizes search identified compatible hyperparameter settings.

KNN Classifier scored as second-worst by both F1 and Accuracy metrics with neither F1 nor Accuracy improving
beyond 80%. As with the MLP, I suspect this would improve as instance count increases beyond 178 records.

Logistic Regression and the Random Forest Classifier performed very similarly. While the Random Forest
Classifier performed the best over multiple iterations and overall, Logistic Regression had more useful,
non-NaN, hyperparameter combinations at a high performance level. Nonetheless, as the purpose of this
tool is to find the best model and its parameters - and it suceeded in doing so by finding the best of
50 iterations - the best model for this dataset of the four models used, Random Forest Classifier is the best
with max depth of 55, minimum samples per leaf of 4, and log base 2 of the number of features considered when
finding the best split.

Logistic Regression is very good using probability to make classifications, but ultimately, the bootstrap
aggregation of Random Forest provided a model with enough samples to be ideal. Regardless, both Logistic
Regression and Random Forest performed very well by both Accuracy and F1.


'''