from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import h5py

sns.set()

def load_data(path):
    hf = h5py.File(path, 'r')

    data = np.array(hf.get('data'))
    labels = np.array(hf.get('labels'))

    hf.close()
    return data, labels   

def plot_confusion_matrix(y_true, y_pred):
    classes = ['A','N','O','~']
    labels = np.arange(4)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)

    ## compute f1 score
    f1_scores = list()
    for i, l in enumerate(classes):
        f1 = 2*cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        f1_scores.append(f1)
        print("class %s"%l + ", F1:%s"%f1)
    print("Mean F1 score: %s" % np.mean(f1_scores))

    fig, ax = plt.subplots(figsize=(6,6))
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Greens', cbar=False)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    fig.tight_layout()

    return fig

def plot_loss(history):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(history['loss'], label='Training loss')
    ax.plot(history['val_loss'], label='Validation loss')
    ax.set_title('Training loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    return fig
    
def main(ags):
    # load model
    from models import toy_model, resnet
    model = resnet()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # load data
    data, labels = load_data(args.h5_path)

    # create train and validation set (80, 20)
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.20, random_state=42)

    # train the model
    fit_history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5, batch_size=32, verbose=1
    )

    # save the weights
    model.save_weights(args.model_path)

    # plot confusion matrix
    y_pred = model.predict(x_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm_figure = plot_confusion_matrix(y_pred, y_true)
    cm_figure.savefig(args.plots_path+"cm.pdf")
    
    # plot loss
    loss_figure = plot_loss(fit_history.history)
    loss_figure.savefig(args.plots_path+"loss.pdf")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--h5-path',
        type=str,
        help='path to hdf5 dataset'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='path for the model weights'
    )
    parser.add_argument(
        '--plots-path',
        type=str,
        help='path to save plots'
    )

    args = parser.parse_args()

    main(args)