import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
import warnings


warnings.filterwarnings("ignore")

mpl.rcParams.update({
    'font.size'           : 16.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'small',})

def load_and_clean_data(filename):
    df = pd.read_csv(filename)
    del df['StandardHours'] # every value is 80
    del df['Over18'] # every value is 'Y'
    del df['EmployeeCount']
    travel = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
    df['BusinessTravel'] = df.BusinessTravel.astype("category", ordered=True, categories=travel).cat.codes
    # df = pd.get_dummies(df, columns=['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'], \
    #                             drop_first=True, prefix=['Dept', 'Edu', 'Gender', 'Role', 'Status', 'OT'])
    df['Department'] = LabelEncoder().fit_transform(df['Department'])
    df['EducationField'] = LabelEncoder().fit_transform(df['EducationField'])
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    df['JobRole'] = LabelEncoder().fit_transform(df['JobRole'])
    df['MaritalStatus'] = LabelEncoder().fit_transform(df['MaritalStatus'])
    df['OverTime'] = LabelEncoder().fit_transform(df['OverTime'])
    return df

def generate_models(X_train, y_train):
    abc = AdaBoostClassifier(n_estimators=10000, learning_rate=.5)
    abc.fit(X_train, y_train)
    gbc = GradientBoostingClassifier(n_estimators=10000, learning_rate=.5, max_features='sqrt', max_depth=2)
    gbc.fit(X_train, y_train)
    svc = SVC(C=100, gamma=0.0001, kernel='rbf', probability=True)
    svc.fit(X_train, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='sgd', alpha=.0005, batch_size=500, learning_rate='constant')
    mlp.fit(X_train, y_train)
    return [abc, gbc, svc, mlp]

def feat_import(n, model, X_train, cols):
    '''Feature importance and Partial Dependency Plots'''
    feature_importances = model.feature_importances_
    top_colidx = np.argsort(feature_importances)[::-1][0:n]
    feature_importances = feature_importances[top_colidx]
    feature_importances = feature_importances / float(feature_importances.max()) #normalize
    feature_names = [cols[idx] for idx in top_colidx]
    print("\nTop {} features and relative importances for {}:".format(str(n), model.__class__.__name__))
    for fn, fi in zip(feature_names, feature_importances):
        print("     {0:<30s} | {1:6.3f}".format(fn, fi))
    name = model.__class__.__name__
    y_ind = np.arange(n-1, -1, -1)
    fig = plt.figure()
    plt.barh(y_ind, feature_importances, height = 0.3, align='center')
    plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
    plt.yticks(y_ind, feature_names)
    plt.xlabel('Relative feature importances')
    plt.ylabel('Features')
    plt.title(name+'\nfeature importance')
    figname = '../plots/feature_importance_bar_plot_'+name+'.png'
    plt.tight_layout()
    plt.savefig(figname, dpi = 300)
    plt.close()

def roc_plot(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    precision = precision_score(y_test, y_predicted)
    accuracy = accuracy_score(y_test, y_predicted)
    recall = recall_score(y_test,y_predicted)
    y_prob = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, label='ROC fold {} (area = {:.2f})'.format(model.__class__.__name__, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Accuracy ={:.2f}\nRecall ={:.2f}\nPrecision ={:.2f}'.format(accuracy,recall, precision))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="best")

def get_predictions(X, EN, name):
    predictions = model.predict(X)
    results = pd.DataFrame(data={'EmployeeNumber': EN, 'Attrition': predictions})
    results = results.replace({1: 'Yes', 0: 'No'})
    results.to_csv('../predictions/AnalyticsChallenge1-Results ('+name+').csv', columns=['EmployeeNumber', 'Attrition'], index=False)


if __name__ == '__main__':

    df = load_and_clean_data('../data/AnalyticsChallenge1-Train.csv')
    del df ['EmployeeNumber']
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    y = df.pop('Attrition').values
    X = df.values
    cols = df.columns.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=123)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    test = load_and_clean_data('../data/AnalyticsChallenge1-Testing.csv')
    EN = test['EmployeeNumber'].values
    del test ['EmployeeNumber']
    XX = scaler.transform(test.values)

    models = generate_models(X_train, y_train)
    feat_import(5, models[0], X_train, cols)
    feat_import(5, models[1], X_train, cols)

    plt.figure(figsize=(10,10))
    for model in models:
        y_pred = model.predict(X_test)
        roc_plot(model, X_test, y_test)
        get_predictions(XX, EN, model.__class__.__name__)
    plt.savefig('../plots/roc_curves.png')
