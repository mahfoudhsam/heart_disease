import numpy as np
import matplotlib.pyplot as plt
import os


def sigmoid(x):
    # Safely clamp inputs between -500 and 500 to prevent exponential overflow warnings
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


def plot_loss_curve(losses):
    plt.plot(losses)
    plt.title("loss curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def plot_linear_svm_margins(model, X, y, save_dir="../figures", filename="linear_svm_margins.png"):
   
    os.makedirs(save_dir, exist_ok=True)
    
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    # Render scattered points (Red for positive risk, Blue for baseline)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='#ff4d4d', marker="o", edgecolors='k', label="Disease (Yes)")
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color='#4d79ff', marker="o", edgecolors='k', label="Healthy (No)")

    # Establish uniform line limits
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # Core separator plane lines
    x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
    x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

    # Negative margin boundary
    x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, -1)

    # Positive margin boundary
    x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, 1)

    # Render geometry layouts
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--", linewidth=2, label="Decision Hyperplane")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k:", label="Negative Margin (-1)")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k-.", label="Positive Margin (+1)")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 2, x1_max + 2])
    
    ax.set_title("Geometric Profile: Linear SVM Max-Margin Separation Plane", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Success] Linear SVM Margin map exported to: {output_path}")