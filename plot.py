#Basavesh Try
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid

def plot(x,y):
    pred_1 = x.flatten()
    pred_2 = y.flatten()
    labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    x = np.arange(len(labels))  # label positions
    width = 0.35  # width of bars

    # Create bar chart
    fig, ax = plt.subplots()
    ax.bar(x - width/2, pred_1, width, label='Prediction 1')
    ax.bar(x + width/2, pred_2, width, label='Prediction 2')

    # Add labels and title
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

     # Generate a unique filename
    unique_id = uuid.uuid4().hex[:8]
    filename = f'prediction_comparison_{unique_id}.png'
    plt.savefig(filename)
    plt.close()
    print(f'Graph saved at: {os.path.abspath(filename)}')

    return filename  # Optional: return file path for later use
