#https://sebastianraschka.com/Articles/2014_kernel_pca.html#scikit-rbf-kernel-pca
import csv
import numpy
#def read_veal_data(filename="./vela_puppis_with_population_split.csv"):
def read_veal_data(filename="/home/spz/Data/Gaia_Vela/vela_puppis_with_population_split.csv"):

    def get_index_for_string(row, str):
        for xi in range(len(row)):
            if str == row[xi]:
                return xi 

    ra = []
    dec = []
    parallax = []
    x = []
    y = []
    z = []
    distance = []
    vr = []
    pmra = []
    mg = []
    pmdec = []
    pop = []
    vela = {}
    data = []
    labels = []
    with open(filename,'r') as csvfile:
        vela_data = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in vela_data:
            if i==0:
                ira = get_index_for_string(row, "ra")
                idec = get_index_for_string(row, "dec")
                ipar = get_index_for_string(row, "parallax")
                ix = get_index_for_string(row, "x")
                iy = get_index_for_string(row, "y")
                iz = get_index_for_string(row, "z")
                img = get_index_for_string(row, "g")
                ip = get_index_for_string(row, "population")
                ivr = get_index_for_string(row, "radial_velocity")
                ipmra = get_index_for_string(row, "pmra")
                ipmdec = get_index_for_string(row, "pmdec")
            else:
                parallax.append(float(row[ipar]))
                ra.append(float(row[ira]))
                dec.append(float(row[idec]))
                x.append(float(row[ix]))
                y.append(float(row[iy]))
                z.append(float(row[iy]))
                mg.append(float(row[img]))
                distance.append(numpy.sqrt(x[-1]**2+y[-1]**2+z[-1]**2))
                pmra.append(float(row[ipmra]))
                pmdec.append(float(row[ipmdec]))
                pop.append(float(row[ip]))
                labels.append(pop[-1])
                if len(row[ivr])>0:
                    vr.append(float(row[ivr]))
                else:
                    if len(vr)>0:
                        vr.append(vr[-1])
                    else:
                        vr.append(0.0)
                data.append([ra[-1], dec[-1], parallax[-1], pmra[-1], pmdec[-1], vr[-1], labels[-1]])
            i+=1
    names = ["ra", "dec", "parallax", "pmra", "pmdec", "vr", "label"]
    return numpy.asarray(data), numpy.asarray(names)

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def stepwise_kpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc

def main(gamma):
    vela_data, vela_names = read_veal_data()

    final_vela_data = vela_data
    print(final_vela_data.shape)

    import pandas as pd
    vela_dataset = pd.DataFrame(final_vela_data)

    vela_dataset.columns = vela_names
    print(vela_dataset.head())
    print(vela_dataset.tail())

    vela_dataset['label'].replace(1, 'Vela1',inplace=True)
    vela_dataset['label'].replace(2, 'Vela2',inplace=True)
    vela_dataset['label'].replace(3, 'Vela3',inplace=True)
    vela_dataset['label'].replace(4, 'Vela4',inplace=True)
    vela_dataset['label'].replace(5, 'Vela5',inplace=True)

    print(vela_dataset.head())
    print(vela_dataset.tail())

    #Data seems ready. Now process..
    ## https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python
    from sklearn.preprocessing import StandardScaler
    x = vela_dataset.loc[:, vela_names[:-1]].values
    print(x)
    x =StandardScaler().fit_transform(x) # normalizing the features
    print(x.shape)

    print("mean=",numpy.mean(x))
    print("std=", numpy.std(x))

    feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
    normalised_vela = pd.DataFrame(x,columns=feat_cols)
    print(normalised_vela.tail())

    # now apply the PCA by transforming the 7dim data into 2 dimensions
    from sklearn.decomposition import PCA
    #pca_vela = PCA(n_components=2)
    #principalComponents_vela = pca_vela.fit_transform(x)

    principalComponents_vela = stepwise_kpca(normalised_vela, gamma=gamma, n_components=2)

    # Now create a DataFrame that will have the principal component values for all the samples.

    principal_vela_Df = pd.DataFrame(data = principalComponents_vela,
                                     columns = ['principal component 1',
                                                'principal component 2'])
    print(principal_vela_Df.tail())

    #print('Explained variation per principal component: {}'.format(pca_vela.explained_variance_ratio_))
    #print('Amount of informaiton lost: {}'.format(1-pca_vela.explained_variance_ratio_.sum()))

    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title("Principal Component Analysis of the Vela Dataset",fontsize=20)
    targets = ['Vela1', 'Vela2', 'Vela3', 'Vela4', 'Vela5']
    colors = ['r', 'g', 'k', 'b', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = vela_dataset['label'] == target
        plt.scatter(principal_vela_Df.loc[indicesToKeep, 'principal component 1'],
                    principal_vela_Df.loc[indicesToKeep, 'principal component 2'],
                    c = color, s = 50)
    plt.legend(targets,prop={'size': 15})
    plt.savefig("fig_kpca_vela_g{}.pdf".format(gamma))
    plt.show()

def new_option_parser():
    from optparse import OptionParser
    result = OptionParser()
    result.add_option("-g", dest="gamma", type="float",
                      default = 1.5,
                      help="input gamma parameter [%default]")
    return result

if __name__ in ("__main__","__plot__"):
    o, arguments  = new_option_parser().parse_args()
    main(o.gamma)

