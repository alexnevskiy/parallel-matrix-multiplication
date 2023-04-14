import argparse
import os
import numpy
import scipy.stats
import math
import sys
import matplotlib.pyplot as plt


def statistics(file_path):
    sample_seq = []
    sample_pthread = []
    sample_mpi = []
    algorithm_seq = False
    algorithm_pthread = False
    algorithm_mpi = False

    with open(file_path, "r") as file:
        threads_count = int(file.readline().split()[1])
        matrix_size = int(file.readline().split()[2])
        experiments_count = int(file.readline().split()[3])
        for line in file.readlines():
            if line.strip() == "algorithm sequentially":
                algorithm_seq = True
                continue
            elif line.strip() == "algorithm pthread":
                algorithm_pthread = True
                continue
            elif line.strip() == "algorithm mpi":
                algorithm_mpi = True
                continue

            if algorithm_seq:
                sample_seq = list(map(int, line.strip().split()))
                algorithm_seq = False
            elif algorithm_pthread:
                sample_pthread = list(map(int, line.strip().split()))
                algorithm_pthread = False
            elif algorithm_mpi:
                sample_mpi = list(map(int, line.strip().split()))
                algorithm_mpi = False

    print("==============================================================")
    print("threads: %d, matrix size: %d, number of experiments: %d" % (threads_count, matrix_size, experiments_count))
    print("==============================================================")

    mean_seq = 0
    mean_pthread = 0
    mean_mpi = 0
    algorithms = ["sequentially", "pthread", "mpi"]
    for algorithm in algorithms:
        sample = []
        if algorithm == "sequentially":
            sample = sample_seq
        elif algorithm == "pthread":
            sample = sample_pthread
        elif algorithm == "mpi":
            sample = sample_mpi

        length = len(sample)

        mean = numpy.mean(sample)
        summary = numpy.sum(sample)

        dispersion = 0
        for element in sample:
            dispersion += (element - mean) ** 2

        dispersion /= length - 1

        radius = scipy.stats.t.ppf((1 + 0.9) / 2, length - 1) * scipy.stats.sem(sample) / math.sqrt(length)

        print("===== %s =====" % algorithm)
        print("Count: %d\nTotal: %f sec\nMean: %f μs\nDispersion: %f\nRadius: %f μs\nInterval: [%f μs, %f μs]\n" % (
            length, summary / 1000000.0, mean, dispersion, radius, mean - radius, mean + radius))

        if algorithm == "sequentially":
            mean_seq = mean
        elif algorithm == "pthread":
            mean_pthread = mean
        elif algorithm == "mpi":
            mean_mpi = mean

    return mean_seq, mean_pthread, mean_mpi, threads_count, matrix_size, experiments_count


def plot_matrix_size(dir_path):
    mean_seq_list = []
    mean_pthread_list = []
    mean_mpi_list = []
    matrix_size_list = []

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            mean_seq, mean_pthread, mean_mpi, threads_count, matrix_size, experiments_count = statistics(file_path)
            mean_seq_list.append(mean_seq)
            mean_pthread_list.append(mean_pthread)
            mean_mpi_list.append(mean_mpi)
            matrix_size_list.append(matrix_size)

    sort_index = numpy.argsort(matrix_size_list)
    matrix_size_list[:] = [matrix_size_list[i] for i in sort_index]
    mean_seq_list[:] = [mean_seq_list[i] for i in sort_index]
    mean_pthread_list[:] = [mean_pthread_list[i] for i in sort_index]
    mean_mpi_list[:] = [mean_mpi_list[i] for i in sort_index]
    plt.plot(matrix_size_list, mean_seq_list, linestyle='-', marker='o', label='sequentially')
    plt.plot(matrix_size_list, mean_pthread_list, linestyle='-', marker='o', label='pthread')
    plt.plot(matrix_size_list, mean_mpi_list, linestyle='-', marker='o', label='mpi')
    plt.xlabel('Matrix size')
    plt.ylabel('Mean execution time, μs')
    plt.legend()
    plt.title('Comparison of execution time relative to the size of the matrix')
    plt.show()

    boost_pthread = []
    boost_mpi = []
    for i in range(len(mean_seq_list)):
        boost_pthread.append(mean_seq_list[i] / mean_pthread_list[i])
        boost_mpi.append(mean_seq_list[i] / mean_mpi_list[i])

    plt.plot(matrix_size_list, boost_pthread, linestyle='-', marker='o', label='pthread boost')
    plt.plot(matrix_size_list, boost_mpi, linestyle='-', marker='o', label='mpi boost')
    plt.xlabel('Matrix size')
    plt.ylabel('Performance boost')
    plt.legend()
    plt.title('Comparison of performance boost relative to the size of the matrix')
    plt.show()


def plot_thread_boost(dir_path):
    mean_seq_list = []
    mean_pthread_list = []
    mean_mpi_list = []
    threads_count_list = []

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            mean_seq, mean_pthread, mean_mpi, threads_count, matrix_size, experiments_count = statistics(file_path)
            mean_seq_list.append(mean_seq)
            mean_pthread_list.append(mean_pthread)
            mean_mpi_list.append(mean_mpi)
            threads_count_list.append(threads_count)

    sort_index = numpy.argsort(threads_count_list)
    threads_count_list[:] = [threads_count_list[i] for i in sort_index]
    mean_seq_list[:] = [mean_seq_list[i] for i in sort_index]
    mean_pthread_list[:] = [mean_pthread_list[i] for i in sort_index]
    mean_mpi_list[:] = [mean_mpi_list[i] for i in sort_index]
    plt.plot(threads_count_list, mean_seq_list, linestyle='-', marker='o', label='sequentially')
    plt.plot(threads_count_list, mean_pthread_list, linestyle='-', marker='o', label='pthread')
    plt.plot(threads_count_list, mean_mpi_list, linestyle='-', marker='o', label='mpi')
    plt.xlabel('Threads count')
    plt.ylabel('Mean execution time, μs')
    plt.legend()
    plt.title('Comparison of execution time relative to the number of threads')
    plt.show()

    boost_pthread = []
    boost_mpi = []
    for i in range(len(mean_seq_list)):
        boost_pthread.append(mean_seq_list[i] / mean_pthread_list[i])
        boost_mpi.append(mean_seq_list[i] / mean_mpi_list[i])

    plt.plot(threads_count_list, boost_pthread, linestyle='-', marker='o', label='pthread boost')
    plt.plot(threads_count_list, boost_mpi, linestyle='-', marker='o', label='mpi boost')
    plt.xlabel('Threads count')
    plt.ylabel('Performance boost')
    plt.legend()
    plt.title('Comparison of performance boost relative to the number of threads')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Comparison of parallelization implementations')
    parser.add_argument('--matrix-size-dir', dest='matrix_size_dir', type=str,
                        help="Path to directory with statistics files to compare matrix sizes")
    parser.add_argument('--thread-boost-dir', dest='thread_boost_dir', type=str,
                        help="Path to directory with statistics files to compare performance boost")
    args = parser.parse_args()

    plot_matrix_size(args.matrix_size_dir)
    plot_thread_boost(args.thread_boost_dir)


if __name__ == '__main__':
    main()
