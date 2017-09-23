import pickle
from sklearn.decomposition import PCA
import scipy
import sys

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def byte_to_string(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(byte_to_string, data.items()))
    if isinstance(data, tuple):  return map(byte_to_string, data)
    return data

def prepareData(K, D, N, PATH_TO_DATA):
    file_content = unpickle(PATH_TO_DATA)
    decoded_file = byte_to_string(file_content)
    file_data = decoded_file['data']
    data_list = []

    for img in file_data[:1000]:
        data = []
        for i in range( 32 * 32 ):
            tup = (int(0.299 * img[i]) + int(0.587 * img[i + 1024]) + int(0.114 * img[i + 2048]))
            data.append(tup)
        data_list.append(data)

    test_set= data_list[:N]
    train_set = data_list[N: 1000]
    pca = PCA(n_components = D, svd_solver="full")
    pca.fit(train_set)
    transform_train = pca.transform(train_set)
    transform_test = pca.transform(test_set)
    labels = []
    test_labels=decoded_file['labels'][:N]
    train_labels = decoded_file['labels'][N:1000]

    for img in transform_test:
        dist_list = []
        li = train_labels
        for im in transform_train:
            dist_list.append(scipy.spatial.distance.euclidean(img, im))

        Z = [x for _, x in sorted(zip(dist_list, li))]

        label_count = {}
        max = 0
        for i in range(K):
            if Z[i] in label_count:
                label_count[Z[i]] += 1
            else:
                label_count[Z[i]] = 1
            if label_count[Z[i]] > max:
                max = label_count[Z[i]]
                lab = Z[i]
        labels.append(lab)

    output_file = open('5077950171.txt', 'w')
    for i in range(len(test_labels)):
        ans = str(labels[i])+" "+ str(test_labels[i])+"\n"
        output_file.write(ans)

prepareData(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

