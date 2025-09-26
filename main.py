from src.preprocess import load_and_clean_data, scale_features
from src.train import split_data
from src.evaluate import evaluate_model
from src.visualize import plot_metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

all_features, all_classes, feature_names = load_and_clean_data('data/mammographic_masses.data.txt')
all_features_scaled = scale_features(all_features, method='standard')
all_features_minmax = scale_features(all_features, method='minmax')

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=1),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=1),
    "SVM Linear": svm.SVC(kernel='linear', C=1.0, probability=True),
    "SVM RBF": svm.SVC(kernel='rbf', C=1.0, probability=True),
    "SVM Sigmoid": svm.SVC(kernel='sigmoid', C=1.0, probability=True),
    "SVM Poly": svm.SVC(kernel='poly', C=1.0, probability=True),
    "k-NN (k=10)": neighbors.KNeighborsClassifier(n_neighbors=10),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

metrics_list = []

for name, clf in models.items():
    print(f"\n================== {name} ==================")
    X = all_features_minmax if name == "Naive Bayes" else all_features_scaled
    X_train, X_test, y_train, y_test = split_data(X, all_classes)
    metrics = evaluate_model(clf, X_train, X_test, y_train, y_test, model_name=name)
    metrics['Model'] = name
    metrics_list.append(metrics)

plot_metrics(metrics_list)
