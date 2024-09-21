import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare_data(file_path):
    df = pd.read_csv(file_path, sep=';')

    df.dropna(inplace=True)
    data = df.drop(columns=['Diagnosis'])

    diagnosis = df['Diagnosis']

    return data.values, diagnosis


# Реализация PCA
def pca_method(data, n_components):
    cov_matrix = np.cov(data.T)
    V, PC = np.linalg.eig(cov_matrix)
    sort_index = np.argsort(-1 * V)
    PC = PC[:, sort_index]
    reduced_data = np.dot(PC.T[0:n_components], data.T)

    full_info = V.sum()
    V = V[sort_index][0:n_components]
    new_info = V.sum()
    print(100 - new_info / full_info * 100)
    return reduced_data.T


# Визуализация данных
def plot_pca(reduced_data, diagnosis, n_components):
    unique_classes = diagnosis.unique()

    if n_components == 2:
        for label in unique_classes:
            class_indices = diagnosis == label
            x_values = reduced_data[class_indices, 0]
            y_values = reduced_data[class_indices, 1]

            plt.scatter(x_values, y_values, label=label)

        plt.title('PCA Visualization (2 Components)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for label in unique_classes:
            class_indices = diagnosis == label
            x_values = reduced_data[class_indices, 0]
            y_values = reduced_data[class_indices, 1]
            z_values = reduced_data[class_indices, 2]

            ax.scatter(x_values, y_values, z_values, label=label)

        ax.set_title('PCA Visualization (3 Components)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend()
        plt.show()


file_path = 'S:/Univercity/4kurs/IAD/Lab1/Exasens.csv'

data, diagnosis = prepare_data(file_path)

n_components = 3
reduced_data = pca_method(data, n_components)

plot_pca(reduced_data, diagnosis, n_components)
