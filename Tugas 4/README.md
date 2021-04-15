Helmi Azhar
1184013
D4 Teknik Informatika 3C
Sistem Pakar

from sys import stdout
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

class FuzzyCMeans:
    def __init__(self, n_clusters=4, fuzziness=2, epsilon=0.0000000000001):

        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.epsilon = epsilon

    def fit(self, X):

        self.data = X

        # Membuat matriks partisi fuzzy dengan nilai acak dari distribusi seragam
        self.fuzzy_matrix = np.random.randint(low=100, high=200, size=(self.data.shape[0], self.n_clusters))

        # Modifikasi matriks partisi fuzzy sehingga setiap baris berjumlah 1
        self.fuzzy_matrix = self.fuzzy_matrix/self.fuzzy_matrix.sum(axis=1, keepdims=True)

        # Matriks sentroid awal acak
        self.centroids = np.random.randint(low=-4, high=4, size=(self.n_clusters, self.data.shape[1]))

        # Hitung kesalahan SSE
        fuzzy_matrix_powered = np.power(self.fuzzy_matrix, self.fuzziness)
        self.sse_error = 0.
        for j in range(self.n_clusters):
            for i in range(self.data.shape[0]):
                self.sse_error += fuzzy_matrix_powered[i][j] * np.power(np.linalg.norm(self.data[i]-self.centroids[j]), 2)

        fig = plt.figure()

        iteration_count = 1

        while True:

            # Cetak nomor iterasi untuk kejelasan kemajuan
            stdout.write("\rIteration {}...".format(iteration_count))
            stdout.flush()
            iteration_count = iteration_count + 1


            # Hitung setiap elemen fuzzy matrix yang diberdayakan untuk ketidak jelasan
            fuzzy_matrix_powered = np.power(self.fuzzy_matrix, self.fuzziness)

            # Bagilah setiap kolom fuzzy matrix powered dengan jumlah kolom
            fuzzy_matrix_powered = fuzzy_matrix_powered/fuzzy_matrix_powered.sum(axis=0, keepdims=True)

            # Hitung sentroid baru (C = (W^p/sum(W^p))T * X)
            new_centroids = np.matmul(fuzzy_matrix_powered.T, self.data)


          

            # Perbarui / Hitung matriks fuzzy baru
            new_fuzzy_matrix = np.zeros(shape=(self.data.shape[0], self.n_clusters))
            for i in range(new_fuzzy_matrix.shape[0]):
                for j in range(new_fuzzy_matrix.shape[1]):
                    new_fuzzy_matrix[i][j] = 1./np.linalg.norm(self.data[i]-new_centroids[j])

            new_fuzzy_matrix = np.power(new_fuzzy_matrix, 2./(self.fuzziness-1))

            new_fuzzy_matrix = new_fuzzy_matrix/new_fuzzy_matrix.sum(axis=1, keepdims=True)


           

            # Hitung kesalahan baru, SSE
            new_fuzzy_matrix_powered = np.power(new_fuzzy_matrix, self.fuzziness)
            new_sse_error = 0.
            for j in range(self.n_clusters):
                for i in range(self.data.shape[0]):
                    new_sse_error += new_fuzzy_matrix_powered[i][j] * np.power(np.linalg.norm(self.data[i]-new_centroids[j]), 2)

            # Jika perubahan dalam kesalahan SSE <epsilon, putus
            if (self.sse_error - new_sse_error) < self.epsilon:
                break

            # Lain, perbarui matriks sentroid dan matriks fuzzy
            self.centroids = new_centroids.copy()
            self.fuzzy_matrix = new_fuzzy_matrix.copy()
            self.sse_error = new_sse_error

            

            max_membership_indices = np.argmax(self.fuzzy_matrix, axis=1)

            # Plot data dengan warna berbeda berdasarkan cluster
            for i in range(self.fuzzy_matrix.shape[0]):

                max_membership_idx = max_membership_indices[i]

                if (max_membership_idx == 0):
                    plt.scatter(self.data[i][0], self.data[i][1], c='blue', marker='o')
                elif (max_membership_idx == 1):
                    plt.scatter(self.data[i][0], self.data[i][1], c='green', marker='o')
                elif (max_membership_idx == 2):
                    plt.scatter(self.data[i][0], self.data[i][1], c='orange', marker='o')
                elif (max_membership_idx == 3):
                    plt.scatter(self.data[i][0], self.data[i][1], c='cyan', marker='o')

            # Plot datanya
            
            plt.scatter(self.centroids[:,0], self.centroids[:,1], c='red', marker='+')
            plt.pause(1)
            plt.close()


if __name__ == "__main__":

    # Parse argumen baris perintah
    parser = argparse.ArgumentParser(description='Fuzzy c-means clustering')
    parser.add_argument('--n_clusters', '-k', type=int, default=4, help='No. of clusters')
    parser.add_argument('--fuzziness', '-m', type=int, default=2, help='Fuzziness parameter')
    parser.add_argument('--n_samples', '-n', type=int, default=500, help='No. of samples to generate')
    parser.add_argument('--epsilon', '-e', type=float, default=0.0000000000001, help='Stopping threshold')
    args = parser.parse_args()

    # Pemeriksaan kasus khusus untuk ketidakjelasan = 1
    if (args.fuzziness == 1):
        print('Fuzziness value of 1 leads to divide by zero error')
        exit(1)

    # Hasilkan data di sekitar 4 pusat menggunakan make_blobs
    centers = [[-2,-2], [2,2], [2,-2], [-2,2]]
    
    X, _ = make_blobs(n_samples=args.n_samples, centers=centers, cluster_std=0.6)

    # Buat objek kelas FuzzyCMeans
    fcm = FuzzyCMeans(n_clusters=args.n_clusters, fuzziness=args.fuzziness, epsilon=args.epsilon)

    # Paskan data X
    fcm.fit(X)

    # Cetak sentroid
    centroids = fcm.centroids
    print('\nFinal Centroids = ')
    print(centroids)

    # Plot datanya

    max_membership_indices = np.argmax(fcm.fuzzy_matrix, axis=1)

    # Plot data dengan warna berbeda berdasarkan cluster
    for i in range(fcm.fuzzy_matrix.shape[0]):
        
        max_membership_idx = max_membership_indices[i]

        if (max_membership_idx == 0):
            plt.scatter(X[i][0], X[i][1], c='blue', marker='o')
        elif (max_membership_idx == 1):
            plt.scatter(X[i][0], X[i][1], c='green', marker='o')
        elif (max_membership_idx == 2):
            plt.scatter(X[i][0], X[i][1], c='orange', marker='o')
        elif (max_membership_idx == 3):
            plt.scatter(X[i][0], X[i][1], c='cyan', marker='o')
            
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='+')
    plt.show()