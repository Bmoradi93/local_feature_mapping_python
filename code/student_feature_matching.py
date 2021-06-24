import numpy as np
from scipy import spatial
import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE 
    #############################################################################
    d_all = []
    feat_1_match_index = []
    feat_2_match_index = []
    feat_match = []
    mathing_confidence = []

    for i in range(features1.shape[0]):
        for j in range(features2.shape[0]):
            local_distance = np.linalg.norm(features1[i]-features2[j])
            d_all.append(local_distance)
        first_min = min(d_all)
        min_1_index = d_all.index(first_min)
        del d_all[min_1_index]
        second_min = min(d_all)
        min_2_index = d_all.index(second_min)
        nearest_neighbor_distance_ratio = first_min /second_min
        match_index = min_1_index
        feature_index_1 = i
        feat_1_match_index.append(i)
        feat_2_match_index.append(match_index)
        mathing_confidence.append(nearest_neighbor_distance_ratio)
        feat_match.append([nearest_neighbor_distance_ratio, feature_index_1, match_index])
        d_all = []
    match_list = []
    confidence_list = []
    for mt in sorted(feat_match)[:100]:
            match_list.append([mt[1], mt[2]])
            confidence_list.append(mt[0])

    matches = np.array(match_list)
    confidences = np.array(confidence_list)

    matches = matches.astype(int)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences

    raise NotImplementedError('`match_features` function in ' +
        '`student_feature_matching.py` needs to be implemented')
