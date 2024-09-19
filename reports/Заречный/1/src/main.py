import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCAM:

    input_size: int
    components: np.array
    dataset: np.array
    eig_val: np.array
    target: np.array
    sort: np.ndarray

    def __init__(self):
        self.load_dataset()
        self.get_main_components()

    def load_dataset(self):
        data = np.genfromtxt('Wholesale customers data.csv', delimiter=',',
                             dtype=None, skip_header=True)
        self.target = data[:, 1:2]
        self.dataset = np.concatenate((data[:, 0:1], data[:, 2:]), axis=1)
        self.input_size = data.shape[1] - 1


    def get_main_components(self):

        math_expectation = np.sum(self.dataset, axis=0) / self.dataset.shape[0]
        dataset = self.dataset - math_expectation

        cov_matrix = np.cov(dataset.T)


        eig_val, eig_vect = np.linalg.eig(cov_matrix)
        sort = np.argsort(-1 * eig_val)
        self.eig_val = eig_val[sort]
        self.components = eig_vect[:, sort]

    def encode(self, data: np.array, compr_size: int):
        full = np.sum(self.eig_val)
        compressed = np.sum(self.eig_val[0:compr_size])
        print(f'Потери - {100 * (1 - compressed / full)} %')
        return np.dot(data, self.components[:, :compr_size])


def main():
    pca = PCAM()
    pca2 = PCA(n_components=2)
    pca3 = PCA(n_components=3)




    # самописный pca, 2 компоненты
    compressed_data = pca.encode(pca.dataset, 2)
    for i in range(3):
        mask = (pca.target.T[0]== (i+1))
        points = compressed_data[mask]
        plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()


    # самописный pca, 3 компоненты
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    compressed_data = pca.encode(pca.dataset, 3)
    for i in range(3):
        mask = (pca.target.T[0]== (i+1))


        points = compressed_data[mask]
        plt.plot(points[:, 0], points[:, 1], 'o')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()

    # sklearn pca, 2 компоненты
    compressed_data = pca2.fit_transform(pca.dataset)
    for i in range(3):
        mask = (pca.target.T[0]== (i+1))

        points = compressed_data[mask]
        plt.plot(points[:, 0], points[:, 1], 'o')

    plt.show()


    # sklearn pca, 3 компоненты
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    compressed_data = pca3.fit_transform(pca.dataset, 3)
    for i in range(3):
        mask = (pca.target.T[0] == (i + 1))
        points = compressed_data[mask]
        plt.plot(points[:, 0], points[:, 1], 'o')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()


if __name__ == "__main__":
    main()
