from pennylane import numpy as pp
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

def plot_performance(history, n_epochs, nfolds):
    """Generate comprehensive performance plots from training history"""

    # Convert to numpy arrays for plotting
    all_train_losses = pp.array([i["train_losses"] for i in history["fold_metrics"]])
    all_val_losses = pp.array([i["val_losses"] for i in history["fold_metrics"]])
    all_train_accs = pp.array([i["train_accs"] for i in history["fold_metrics"]])
    all_val_accs = pp.array([i["val_accs"] for i in history["fold_metrics"]])

    # Calculate mean and std across folds
    mean_train_loss = pp.mean(all_train_losses, axis=0)
    std_train_loss = pp.std(all_train_losses, axis=0)
    mean_val_loss = pp.mean(all_val_losses, axis=0)
    std_val_loss = pp.std(all_val_losses, axis=0)

    mean_train_acc = pp.mean(all_train_accs, axis=0)
    std_train_acc = pp.std(all_train_accs, axis=0)
    mean_val_acc = pp.mean(all_val_accs, axis=0)
    std_val_acc = pp.std(all_val_accs, axis=0)


    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Training and Validation Loss (averaged over folds)
    ax1 = plt.subplot(2, 4, 1)
    epochs = range(1, n_epochs + 1)
    ax1.plot(epochs, mean_train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, color='b')
    ax1.plot(epochs, mean_val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, color='r')
    ax1.set_xlim(-1,min(n_epochs,pp.where(mean_val_loss==0)[0][0])+1)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss\n(Mean ± Std over 5 Folds)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Training and Validation Accuracy (averaged over folds)
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(epochs, mean_train_acc, 'b-', label='Train Acc', linewidth=2)
    ax2.fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha=0.3, color='b')
    ax2.plot(epochs, mean_val_acc, 'r-', label='Val Acc', linewidth=2)
    ax2.fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, alpha=0.3, color='r')
    ax2.set_xlim(-1,min(n_epochs,pp.where(mean_val_loss==0)[0][0])+1)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy\n(Mean ± Std over 5 Folds)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Individual fold losses
    ax3 = plt.subplot(2, 4, 3)
    for i in range(nfolds):
        ax3.plot(epochs, all_val_losses[i], label=f'Fold {i+1}', alpha=0.7)
    ax3.set_xlim(-1,min(n_epochs,pp.where(mean_val_loss==0)[0][0])+1)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation Loss', fontsize=12)
    ax3.set_title('Validation Loss per Fold', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Individual fold accuracies
    ax4 = plt.subplot(2, 4, 4)
    for i in range(nfolds):
        ax4.plot(epochs, all_val_accs[i], label=f'Fold {i+1}', alpha=0.7)
    ax4.set_xlim(-1,min(n_epochs,pp.where(mean_val_loss==0)[0][0])+1)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title('Validation Accuracy per Fold', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Confusion Matrix (averaged over folds)
    ax5 = plt.subplot(2, 4, 5)
    all_y_true = pp.concatenate([i["y_true"] for i in history["fold_metrics"]])
    all_y_pred = pp.concatenate([i["y_pred"] for i in history["fold_metrics"]])
    cm = confusion_matrix(all_y_true, all_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=True)
    ax5.set_xlabel('Predicted Label', fontsize=12)
    ax5.set_ylabel('True Label', fontsize=12)
    ax5.set_title('Confusion Matrix\n(All Folds Combined)', fontsize=13, fontweight='bold')

    # 6. ROC Curve
    ax6 = plt.subplot(2, 4, 6)
    all_y_prob = pp.concatenate([i["y_prob"] for i in history["fold_metrics"]])
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    roc_auc = auc(fpr, tpr)
    ax6.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax6.set_xlabel('False Positive Rate', fontsize=12)
    ax6.set_ylabel('True Positive Rate', fontsize=12)
    ax6.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax6.legend(loc="lower right")
    ax6.grid(True, alpha=0.3)

    # 7. Precision-Recall Curve
    ax7 = plt.subplot(2, 4, 7)
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
    pr_auc = auc(recall, precision)
    ax7.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax7.set_xlabel('Recall', fontsize=12)
    ax7.set_ylabel('Precision', fontsize=12)
    ax7.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax7.legend(loc="lower left")
    ax7.grid(True, alpha=0.3)

    # 8. Final Accuracy Bar Chart per Fold
    ax8 = plt.subplot(2, 4, 8)
    fold_accs = pp.array([i["final_acc"] for i in history["fold_metrics"]])
    fold_nums = [f"Fold {i+1}" for i in range(nfolds)]
    bars = ax8.bar(fold_nums, fold_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax8.axhline(y=pp.mean(fold_accs), color='r', linestyle='--', linewidth=2, label=f'Mean: {pp.mean(fold_accs):.3f}')
    ax8.set_ylabel('Accuracy', fontsize=12)
    ax8.set_title('Final Validation Accuracy per Fold', fontsize=13, fontweight='bold')
    ax8.set_ylim([0, 1])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    # plt.savefig('quantum_classifier_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print(f"\n{'='*50}")
    print("SUMMARY STATISTICS")
    print(f"{'='*50}")
    print(f"Mean Validation Accuracy: {pp.mean(fold_accs):.4f} ± {pp.std(fold_accs):.4f}")
    print(f"Best Fold Accuracy: {max(fold_accs):.4f}")
    print(f"Worst Fold Accuracy: {min(fold_accs):.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    print(f"{'='*50}")