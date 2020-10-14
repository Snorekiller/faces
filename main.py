import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import mglearn
import seaborn as sns

from funcsigs import signature

from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict

faces = fetch_olivetti_faces(shuffle=True)

tf.disable_v2_behavior()

x, y = faces['data'], faces['target']
# print(f"Data shape: {x.shape}")
# print(f"Label shape: {y.shape}")


for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x[i].reshape(64, 64))

plt.show()

# divide data into training, validation and testing(shuffle)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0,
                                                    test_size=0.3)
# print(f"x_train shape: {x_train.shape}")
# print(f"y_train shape:{y_train.shape}")


# use PCA for diensionality reduction - reconstruct images using a subset of features


pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)

number_of_people = 10
index_range = number_of_people * 10
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.scatter(x_pca[:index_range, 0], x_pca[:index_range, 1], 10, y[:index_range])
plt.xlabel("First Principle Component")
plt.ylabel("Second Principle Component")
plt.title(f"PCA projection of {number_of_people} people")
plt.show()

mglearn.plots.plot_pca_illustration()

# finding optimal number of Principle Components
pca = PCA()
pca.fit(x)
plt.figure(1, figsize=(12, 8))
plt.plot(pca.explained_variance_)
plt.xlabel("Components")
plt.ylabel("Explained Variaces")
plt.show()

# In the figure above it can be seen that 90 and more PCA components represent the same data.
# creating the average face
n_components = 90
pca = PCA(n_components=n_components, whiten=True)
pca.fit(x_train)
plt.title("Average face")
plt.xticks([])
plt.yticks([])
plt.imshow(pca.mean_.reshape((64, 64)), cmap="gray")
plt.show()

# Reconstructing images from subset features
number_of_eigenfaces = len(pca.components_)
# eigen_faces = pca.components_.reshape((number_of_eigenfaces, x.shape[1], x.shape[2]))

cols = 10
rows = int(number_of_eigenfaces / cols)

axes = []
fig = plt.figure(figsize=(15, 15))
for i in range(number_of_eigenfaces):
    axes.append(fig.add_subplot(rows, cols, i + 1))
    plt.title(f"Eigen id:{format(i)}")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pca.components_[i].reshape(faces.images[1].shape), cmap="gray")

plt.suptitle("All Eigen Faces".format(10 * "=", 10 * "="))
plt.show()

# classification results
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

clf = SVC()
clf.fit(x_train_pca, y_train)
y_pred = clf.predict(x_test_pca)
accuracy = format(metrics.accuracy_score(y_test, y_pred))
print(f"accuracy score: {accuracy}")

plt.figure(1, figsize=(12, 8))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
plt.show()
print(metrics.classification_report(y_test, y_pred))

models = []
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("LR", LogisticRegression()))
models.append(("NB", GaussianNB()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("DT", DecisionTreeClassifier()))
models.append(("SVM", SVC()))

for name, model in models:
    clf = model

    clf.fit(x_train_pca, y_train)

    y_pred = clf.predict(x_test_pca)
    print(10 * "=", "{} Result".format(name).upper(), 10 * "=")
    print("Accuracy score:{:0.2f}".format(metrics.accuracy_score(y_test, y_pred)))
    print()

pca = PCA(n_components=n_components, whiten=True)
pca.fit(x)
x_pca = pca.transform(x)
for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    cv_scores = cross_val_score(model, x_pca, y, cv=kfold)
    print("{} mean cross validations score:{:.2f}".format(name, cv_scores.mean()))

lr = LinearDiscriminantAnalysis()
lr.fit(x_train_pca, y_train)
y_pred = lr.predict(x_test_pca)
print()
print("Accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))

cm = metrics.confusion_matrix(y_test, y_pred)

plt.subplots(1, figsize=(12, 12))
sns.heatmap(cm)
plt.show()

print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))

loo_cv = LeaveOneOut()
clf = LogisticRegression()
cv_scores = cross_val_score(clf, x_pca, y, cv=loo_cv)
print("{} Leave One Out cross-validation mean accuracy score:{:.2f}".format(clf.__class__.__name__, cv_scores.mean()))

lr = LogisticRegression(C=1.0, penalty="l2")
lr.fit(x_train_pca, y_train)
print("\nlr score:{:.2f}".format(lr.score(x_test_pca, y_test)))

Target = label_binarize(y, classes=range(40))
print(Target.shape)
print(Target[0])

n_classes = Target.shape[1]

X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(x, Target,
                                                                                                test_size=0.3,
                                                                                                stratify=Target,
                                                                                                random_state=0)
pca = PCA(n_components=n_components, whiten=True)
pca.fit(X_train_multiclass)

X_train_multiclass_pca = pca.transform(X_train_multiclass)
X_test_multiclass_pca = pca.transform(X_test_multiclass)

oneRestClassifier = OneVsRestClassifier(lr)

oneRestClassifier.fit(X_train_multiclass_pca, y_train_multiclass)
y_score = oneRestClassifier.decision_function(X_test_multiclass_pca)

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = metrics.precision_recall_curve(y_test_multiclass[:, i],
                                                                y_score[:, i])
    average_precision[i] = metrics.average_precision_score(y_test_multiclass[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test_multiclass.ravel(),
                                                                        y_score.ravel())
average_precision["micro"] = metrics.average_precision_score(y_test_multiclass, y_score,
                                                             average="micro")
print('\nAverage precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters else {})
plt.figure(1, figsize=(12, 8))
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
plt.show()

lda = []
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit(x_train, y_train).transform(x_train)
X_test_lda = lda.transform(x_test)

lr = []
lr = LogisticRegression(C=1.0, penalty="l2")
lr.fit(X_train_lda, y_train)
y_pred = lr.predict(X_test_lda)

print("Accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))
"""
       # contains a error I cant find a solution for
        
work_flows_std = []
work_flows_std.append(('lda', LinearDiscriminantAnalysis(n_components=n_components)))
work_flows_std.append(('logReg', LogisticRegression(C=1.0, penalty="l2")))
model_std = []
model_std = Pipeline(work_flows_std)
model_std.fit(x_train, y_train)
y_pred = model_std.predict(x_test)
"""
print("Accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))

"""
This is a encoder that encodes four layers and decodes four layer, 
it is time consuming so it has been comented out
"""

"""
# use autoencoder - reconstruct using a compressed representation (code)
x_train_encode = faces["images"]
nb_epochs = 600
batch_size = 50
code_length = 256
# rezise to avoid memory issues(may lose minor visual precision
width = 32
height = 32
graph = tf.Graph()

with graph.as_default():
    encode_input_x1 = tf.placeholder(tf.float32, shape=(None, x_train_encode.shape[1], x_train_encode.shape[2], 1))
    encode_input = tf.image.resize_images(encode_input_x1, (width, height),
                                          method=tf.image.ResizeMethod.BICUBIC)
    # Encoder

    conv_0 = tf.keras.layers.Conv2D(16, (3, 3), (2, 2), padding="same",
                                    activation=tf.nn.relu)(encode_input)
    conv_1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=tf.nn.relu)(conv_0)
    conv_2 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=tf.nn.relu)(conv_1)
    conv_3 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=tf.nn.relu)(conv_2)

    code_input = tf.keras.layers.Flatten()(conv_3)
    code_layer = tf.keras.layers.Dense(units=code_length,
                                       activation=tf.nn.sigmoid)(code_input)
    code_mean = tf.reduce_mean(code_layer, axis=1)
# decoder
with graph.as_default():
    decoder_input = tf.reshape(code_layer, (-1, int(width / 2), int(height / 2), 1))
    convt_0 = tf.keras.layers.Conv2DTranspose(128, (3, 3), (2, 2), padding="same",
                                              activation=tf.nn.relu)(decoder_input)
    convt_1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding="same",
                                              activation=tf.nn.relu)(convt_0)
    convt_2 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding="same",
                                              activation=tf.nn.relu)(convt_1)
    convt_3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding="same",
                                              activation=tf.nn.relu)(convt_2)

    decoded_images = tf.image.resize_images(convt_3, (x_train_encode.shape[1], x_train_encode.shape[2]),
                                            method=tf.image.ResizeMethod.BICUBIC)

with graph.as_default():
    # loss
    loss = tf.nn.l2_loss(convt_3 - encode_input)
    training_step = tf.train.AdamOptimizer(0.001).minimize(loss)

session = tf.InteractiveSession(graph=graph)
tf.global_variables_initializer().run()

for e in range(nb_epochs):
    np.random.shuffle(x_train_encode)

    total_loss = 0.0
    code_means = []

    for i in range(0, x_train_encode.shape[0] - batch_size, batch_size):
        Xt = np.expand_dims(x_train_encode[i:i + batch_size, :, :],
                           axis=3).astype(np.float32)

        _, n_loss, c_mean = session.run([training_step, loss, code_mean],
                                        feed_dict={encode_input_x1: Xt})

        total_loss += n_loss
        code_means.append(c_mean)
    print(f"Epoch{e + 1}) Average loss per sample: {total_loss / float(x_train_encode.shape[0])} "
      f"(Code mean; {np.mean(code_means)})")
"""
# cross-validation - SGD, BGD and MBGD
