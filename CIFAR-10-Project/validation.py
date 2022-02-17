import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import pandas as pd

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def get_class_accuracy(confusion):
    acc_list = []
    for label, row in enumerate(confusion):
        n_correct = row[label]
        n_samples = row.sum()
        acc_class = n_correct/n_samples
        rep = n_samples/torch.sum(confusion)
        acc_class, rep = acc_class.item(), rep.item()
        acc_list.append(acc_class)
    return acc_list

def get_total_accuracy(confusion):
    n_correct = torch.sum(torch.diag(confusion))
    n_samples = torch.sum(confusion)
    acc = n_correct/n_samples
    acc = acc.item()
    return acc 


def animate(i):
    try:
        dataframe = pd.read_csv("./CIFAR-10-Project/validation_info.csv")
    except:
        print("Error loading file... there may be no data yet")
    else:
        batch_size = dataframe["batch_size"].tolist()[-1]
        learning_rate = dataframe["learning_rate"].tolist()[-1]
        
        y_acc_class = []
        for class_ in classes: # create slots to represent each class, we will put the accuracy of each class per epoch in these slots
            y_acc_class.append([])
        
        y_acc_total = []
        loss = dataframe["loss"].tolist()[-1]

        for confusion in dataframe["confusion"]: # for each confusion matrix on each epoch calculate wanted parameters
            confusion = torch.tensor(pd.eval(confusion).tolist()) # convert confusion matrix to tensor
            acc_list = get_class_accuracy(confusion)
            for slot, acc in zip(y_acc_class, acc_list):
                slot.append(acc) # append class acuracy in appropriate slot
            acc_total = get_total_accuracy(confusion)
            y_acc_total.append(acc_total)
        
        for class_, slot in zip(classes, y_acc_class):
            dataframe[f"{class_} accuracy"] = slot
        dataframe["total accuracy"] = y_acc_total

        plot_list = ["epoch", "total accuracy"]
        style_list = ["--"]
        for class_ in classes:
            plot_list.append(f"{class_} accuracy")
            style_list.append("-")

        plt.Axes.clear(ax[0])
        dataframe[plot_list].plot(x="epoch", kind="line", ax=ax[0], style=style_list)
        plt.Axes.clear(ax[1])
        dataframe.plot(x="epoch", y=["loss"], kind="line", ax=ax[1])
        text = "\n".join((f"batch size = {batch_size}", 
                                f"learning_rate = {learning_rate}", 
                                f"accuraccy = {y_acc_total[-1]:.3f}", 
                                f"loss = {loss:.3f}"))
        ax[0].text(1.01, 1, text, va="top", transform=ax[0].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        plt.legend()

fig, ax = plt.subplots(2,1)
ani = FuncAnimation(plt.gcf(), animate, interval=5*1000)
plt.show()