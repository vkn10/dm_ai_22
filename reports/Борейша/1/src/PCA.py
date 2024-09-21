import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main():
    dataset = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, usecols=range(7))
    classes = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, usecols=[7])

    dataset_centered = dataset.apply(lambda x: x - x.mean(), axis=0)
    cov_matrix = np.cov(dataset_centered.T)
    eig_val, eig_vec = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_indices]
    eig_vec = eig_vec[:, sorted_indices]

    V_2 = eig_vec[:, :2]
    reduced_info_2 = np.sum(eig_val[:2])
    full_info = np.sum(eig_val)
    loss_2 = 100 * (1 - reduced_info_2 / full_info)
    print(f'Потери для 2 компонент: {loss_2:.2f}%')

    V_3 = eig_vec[:, :3]
    reduced_info_3 = np.sum(eig_val[:3])
    loss_3 = 100 * (1 - reduced_info_3 / full_info)
    print(f'Потери для 3 компонент: {loss_3:.2f}%')

    # Отображение 2D
    reduced_2d = dataset_centered.dot(V_2)
    plt.scatter(reduced_2d.iloc[:, 0], reduced_2d.iloc[:, 1], c=classes, cmap='plasma')
    plt.xlabel('1-я главная компонента')
    plt.ylabel('2-я главная компонента')
    plt.title('2D собственная реализация')
    plt.show()

    # Отображение 3D
    reduced_3d = dataset_centered.dot(V_3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(reduced_3d.iloc[:, 0], reduced_3d.iloc[:, 1], reduced_3d.iloc[:, 2], c=classes, cmap='plasma')
    ax.set_xlabel('1-я главная компонента')
    ax.set_ylabel('2-я главная компонента')
    ax.set_zlabel('3-я главная компонента')
    plt.title('3D собственная реализация')
    plt.show()

    pca_2 = PCA(n_components=2)
    reduced_data_2 = pca_2.fit_transform(dataset_centered)

    pca_3 = PCA(n_components=3)
    reduced_data_3 = pca_3.fit_transform(dataset_centered)
    
    # Отображение 2D
    plt.scatter(reduced_data_2[:, 0], reduced_data_2[:, 1], c=classes, cmap='viridis')
    plt.xlabel('1-я главная компонента')
    plt.ylabel('2-я главная компонента')
    plt.title('2D sklearn')
    plt.show()

    # Отображение 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(reduced_data_3[:, 0], reduced_data_3[:, 1], reduced_data_3[:, 2], c=classes, cmap='viridis')
    ax.set_xlabel('1-я главная компонента')
    ax.set_ylabel('2-я главная компонента')
    ax.set_zlabel('3-я главная компонента')
    plt.title('3D sklearn')
    plt.show()


if __name__ == "__main__":
    main()
