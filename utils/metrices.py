import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def compute_and_plot_metrics(all_labels, all_preds, class_names, model_name="Model"):
    # Generate classification report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(f"\n--- Detailed Classification Report for {model_name} ---")
    print(report)
    
    # Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.show()
    
    return cm, report