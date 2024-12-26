

# Move script from test_notebook into here
# google colab notebook should just call this script
# but can also be called from command line


import matplotlib.pyplot as plt

from model import Trainer


def main(epochs= 5, device=None, optimizer=None):

    trainer = Trainer.standard_init()

    train_loss, train_acc, eval_loss, eval_acc = trainer.train_model(64, epochs)

    plot_metric(train_loss, marker="x")
    plot_metric(eval_loss)
    plt.ylabel("Loss")
    plt.show()
    plot_metric(train_acc, marker="x")
    plot_metric(eval_acc)
    plt.ylabel("Accuracy")
    plt.show()


def plot_metric(metric, **kwargs):
    plt.plot(range(len(metric)), metric, **kwargs)
    plt.xlabel("Epochs")
    plt.grid(True)


if __name__ == "__main__":

    # parse command line args

    main(...)

"""
Aufgaben:
- implement centralized baseline
- simulate parallel training of local methods
- Optimizers implementieren: LARS, LAMB, AdamW, SGDM

"""
