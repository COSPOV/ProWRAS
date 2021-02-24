r"""
ProWRAS:
Generates sample data points for inbalanced data sets.
"""

# List of public functions.
__all__ = ["ProWRAS_gen"]

# See Style guide: https://www.python.org/dev/peps/pep-0008/
# Importing libraries

import warnings
import time

import numpy as np
from numpy.random import randint, normal, seed
from sklearn.neighbors import NearestNeighbors


def minority_class(data, labels):
    """Returns all data points that are in the minority class.

    data: numpy array of data points.

    labels: numpy array of labels. (== 1: minority class, != 1: other class)
    """
    return data[np.where(labels == 1)]


def majority_class(data, labels):
    """Returns all data points that are not in the minority class.

    data: Numpy array of data points.

    labels: Numpy array of labels. (== 1: minority class, != 1: other class)
    """
    return data[np.where(labels != 1)]


def random_subset(data, size):
    """Return a subset of the array data. The returned data points are selected
    by random.

    data: Numpy array of data points.

    size: Number of values in the returned array.
    """
    return data[randint(len(data), size=size)]


def random_item(data):
    """Returns a random item in the given array.

    data: Numpy array of data points.
    """
    return data[randint(len(data))]


def concatArray(listOfListOfThings):
    """Changes the type of the given array to Numpy array.
    Concatenates all aubarrays.
    
    listOfListOfThings: Array of arrays.

    Example:
    [[1,2,3],[4,5,6,7],[8,9]] will become array([1,2,3,4,5,6,7,8,9])

    [[[1,2],[3,4]],[[5,6],[7,8]]] will become array([[1,2],[3,4],[5,6],[7,8]])
    """
    return np.concatenate(np.array(listOfListOfThings))


def random_vector_with_sum_one(size):
    """Rerurns an array of the given size. The array has random non negative
    values. The sum of all Values is one.

    size: Number of the expected values in the array.
    """
    w = [0]
    sum_w = 0
    while sum_w == 0:
        w = randint(100, size=size)
        sum_w = sum(w)

    return np.array(w / sum_w)


# Defining ProWRAS function

class ProWRASHelper:
    """
    A collection of helpfull functions for the ProWRAS algorithm.
    """

    def __init__(
        self,
        convex_nbd,
        shadow,
        sigma,
        n_jobs,
        num_feats):
        """Constructor of the helper class. Checks some values for validity.
        Sets the parameters for the helping functions.

        convex_nbd: Number of points in the neighbourhoods. If a cluster has
            more than this amount of points. It will be divided in
            neighbourhoods of this size.

        shadow: Number of shadow point to create for each point in cluster.

        sigma: Used to generate random shadow samples with normal deviation.
        
        n_jobs: Maximal count of processor cores, used during the calculation.
        
        num_feats: Number of features for each sample.
        """
        assert convex_nbd >= 1
        assert shadow >= 1
        assert n_jobs >= 1
        assert num_feats >= 1

        self.convex_nbd_size = int(convex_nbd)
        self.shadow_size = int(shadow)
        self.sigma = sigma
        self.n_jobs = int(n_jobs)
        self.num_feats = int(num_feats)
        self.isDebugEnabled = False

    def random_vector(self, point):
        """Returns a random vector with normal deviation
        around the given point.

        point: Data point (Numpy array of float values.)

        returns: Numpy array of float values.
        """
        return point + [normal(0, self.sigma) for k in range(len(point))]

    def shadow_for_point(self, point):
        """Creates random shadow points for a given point.

        point: Data point (Numpy array of float values.)

        returns: Numpy array of points.
        """
        return [
            self.random_vector(point)
            for _c in range(self.shadow_size)
            ]

    def split_into_neighbourhoods(self, cluster):
        """If the given cluster has more points than the given convex_nbd_size,
        it will be divided into neighbourhood of size convex_nbd_size.

        It will be returned an array of arrays of indices.
        
        cluster: Numpy array of data points.
        """
        if len(cluster) > self.convex_nbd_size:
            self.debug('local')
            return self.neb_grps(cluster)

        self.debug('global')
        return np.array([np.array(range(len(cluster)))])

    # Defining ProWRAS function

    def neb_grps(self, data):
        """
        Function calculating nearest convex_nbd neighbours
        (among input data points), for every input data point.

        data: Numpy array of data points.
        """
        nbrs = NearestNeighbors(
                n_neighbors=self.convex_nbd_size,
                n_jobs=self.n_jobs)
        nbrs = nbrs.fit(data)
        _distances, indices = nbrs.kneighbors(data)
        return np.asarray(indices)

    def partition_info(self, data, labels, max_levels, n_neighbors, theta):
        """Divides the given array of data points into weighted layers.

        data: Numpy array of data points.

        labels: numpy array of labels. (1: minority class, != 1: other class)

        max_levels: Maximal number of returned layers.

        n_neighbours: Number of neighbours in a layer from the minority class
            for points in majority class.

        theta: Scaling factor for the weights.
        """

        def proximity_level(level):
            """Calculates the weight for a layer level."""
            return np.exp(-theta * (level - 1))

        # Step 2
        P = np.where(labels == 1)[0]
        data_maj = majority_class(data, labels)

        Ps = []
        weights = []

        # Step 3
        for i in range(1, max_levels):
            if len(P) == 0:
                break
            # Step 3 a
            n_neighbors = min([len(P), n_neighbors])
            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            nn.fit(data[P])
            _distances, indices = nn.kneighbors(data_maj)

            # Step 3 b
            P_i = np.unique(np.hstack(indices))

            # Step 3 c - proximity levels are encoded in the Ps list index
            Ps.append(P[P_i])
            weights.append(proximity_level(i))

            # Step 3 d
            P = np.delete(P, P_i)

        if len(P) > 0:
            # Step 4
            Ps.append(P)

            # Step 5
            weights.append(proximity_level(i - 1))

        # Step 6
        weights = np.array(weights)

        # weights is the probability distribution of sampling in the
        # clusters identified
        weights = weights / np.sum(weights)
        return (np.array(Ps), weights)

    def limit_neighbourhood(self, neighbourhood):
        """Limits a neighbourhood to convex_nbd_size amount of values."""
        if len(neighbourhood) > self.convex_nbd_size:
            return neighbourhood[:self.convex_nbd_size]
        return neighbourhood

    def generate_points(self, cluster, num_of_points_to_create, num_convcomb):
        """
        Function creating samples for one minority data point neighbourhood.

        cluster: Numpy array of data points.

        num_of_points_to_create: Size of the returned array with new created
            points.

        num_convcomb: Number of points used for creating one new point.
        """

        assert num_convcomb >= 1

        generated_data = []

        isLowVariance = True
        num_convcomb = int(num_convcomb)
        if num_convcomb < self.num_feats:
            isLowVariance = False
            num_convcomb = 2
            self.debug('high')
        else:
            self.debug('low')

        neb_list = self.split_into_neighbourhoods(cluster)

        for _i in range(int(num_of_points_to_create)):
            random_neighbourhood = cluster[random_item(neb_list)]

            if isLowVariance:
                data_shadow = concatArray([
                    self.shadow_for_point(x)
                    for x in self.limit_neighbourhood(random_neighbourhood)
                    ])
            else:
                data_shadow = random_neighbourhood

            shadowPoints = random_subset(data_shadow, num_convcomb)
            aff_w = random_vector_with_sum_one(num_convcomb)

            generated_data.append(np.dot(aff_w, shadowPoints))

        return np.array(generated_data)

    def debug(self, message):
        """Prints a message, if debug is enabled."""
        if self.isDebugEnabled:
            print(message)

    def enableDebug(self):
        """Prints a message, if debug is enabled."""
        self.isDebugEnabled = True

def ProWRAS_gen(
        data,
        labels,
        max_levels,
        convex_nbd,
        n_neighbors,
        max_concov,
        num_samples_to_generate,
        theta,
        shadow,
        sigma,
        n_jobs,
        enableDebug=False):
    """Calculates shadow points for a minority class of data points.

    data: numpy array of data points.

    labels: numpy array of labels. (== 1: minority class, != 1: other classes)

    max_levels: Maximal number of layers, used in the algorithm.

    convex_nbd: Number of points in the neighbourhoods. If a layer has more
        than this amount of points. It will be divided in neighbourhoods of
        this size.

    n_neighbours: Number of neighbours in a layer from the minority class
        for points in majority class. This parameter is used for generating
        the layers.

    max_concov:

    num_of_samples_to_generate: Number of new generated points in the minority
        class of data points.

    theta: Scaling factor for the weights of the layers.

    shadow: Number of shadow point to create for each point.

    sigma: Used to generate random shadow samples with normal deviation.
    
    n_jobs: Maximal count of processor cores, used during the calculation.
    """

    warnings.filterwarnings("ignore")
    seed(int(time.time() * 1000) & 0x0fffffff)

    features_1_trn = minority_class(data, labels)
    features_0_trn = majority_class(data, labels)

    num_feats = data.shape[1]

    helper = ProWRASHelper(convex_nbd, shadow, sigma, n_jobs, num_feats)
    if enableDebug:
        helper.enableDebug()

    clusters, weights = helper.partition_info(
        data, labels, max_levels, n_neighbors, theta)

    num_samples_each_cluster = np.ceil(num_samples_to_generate * weights)
    num_convcomb_each_cluster = np.ceil((weights / max(weights)) * max_concov)

    sample_params = zip(
        clusters,
        num_samples_each_cluster,
        num_convcomb_each_cluster
        )

    synth_samples = concatArray([
        helper.generate_points(data[cluster], num_samples, num_convcomb)
        for (cluster, num_samples, num_convcomb) in sample_params
        ])

    prowras_train = np.concatenate((
        synth_samples,
        features_1_trn,
        features_0_trn
        ))

    minority_class_size = len(synth_samples) + len(features_1_trn)
    prowras_labels = np.concatenate((
        np.ones(minority_class_size),
        np.zeros(len(features_0_trn))
        ))

    return(prowras_train, prowras_labels)
