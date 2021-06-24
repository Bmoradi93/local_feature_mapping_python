import cv2
import numpy as np

def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    # The code starts from this point.
    # I will be following the Szeliski algorithm in page 190:

    # (1) Compute the horizontal and vertical derivatives of the 
    # image I x and I y by convolving the original image 
    # with derivatives of Gaussians (Section 3.2.3).

    gaussian_filter_size = 3
    Gausian_standard_diviation = 0.5
    gaussian_low_pass_filter = cv2.getGaussianKernel(gaussian_filter_size,Gausian_standard_diviation)
    img_guassian_filtered = cv2.filter2D(image,-1,gaussian_low_pass_filter)
    img_grad_y, img_grad_x = np.gradient(img_guassian_filtered)

    # (2) Compute the three images corresponding to the outer products of these gradients.
    # (The matrix A is symmetric, so only three entries are needed.)
    img_grad_x_2 = img_grad_x**2
    img_grad_x_y = img_grad_x*img_grad_y
    img_grad_y_2 = img_grad_y**2

    # (3) Convolve each of these images with a larger Gaussian.
    larger_gaussian_filter_size = 7
    Gausian_standard_diviation = 0.5
    gaussian_larger_filter = cv2.getGaussianKernel(larger_gaussian_filter_size,Gausian_standard_diviation)
    img_grad_x_2_filtered = cv2.filter2D(img_grad_x_2, -1, gaussian_larger_filter)
    img_grad_x_y_filtered = cv2.filter2D(img_grad_x_y, -1, gaussian_larger_filter)
    img_grad_y_2_filtered = cv2.filter2D(img_grad_y_2, -1, gaussian_larger_filter)

    factor_alpha = 0.06

    det_A = img_grad_x_2_filtered * img_grad_y_2_filtered - img_grad_x_y_filtered **2
    trace_A = img_grad_x_2_filtered + img_grad_y_2_filtered
    Forstner_Harris = det_A -factor_alpha * trace_A ** 2

    # (4) Compute a scalar interest measure using one of the formulas
    h, w = Forstner_Harris.shape
    interest_points = []
    fh_index = []
    for f1 in range(0, h):
        fh_index = []
        for f2 in range(0, w):
                fh_index = [Forstner_Harris[f1,f2]]
                fh_index.append(f2)
                fh_index.append(f1)
                interest_points.append(fh_index)

    # (5) Find local maxima above a certain threshold and report them as detected feature
    # point locations.

    interest_points_sort_list = sorted(interest_points, reverse = True)
    number_of_desired_interest_points = 1000
    interest_points_sort_list = interest_points_sort_list[0:number_of_desired_interest_points]
    interest_point_neighborhood_list = []

    for i in range(0, number_of_desired_interest_points):
        local_nibrhd = float('inf')
        first_location = interest_points_sort_list[i]
        for j in range(0, i):
            second_location = interest_points_sort_list[j]
            euclidean_distance = (first_location[1]-second_location[1])**2 +(first_location[2]-second_location[2])**2
            local_nibrhd = min(euclidean_distance, local_nibrhd)
        interest_point_neighborhood_list.append([np.sqrt(local_nibrhd), first_location[0], first_location[1], first_location[2]])
    x_list = []
    y_list = []
    for interest_point_neighborhood in sorted(interest_point_neighborhood_list, reverse = True):
        x_list.append(interest_point_neighborhood[2])
        y_list.append(interest_point_neighborhood[3])  
    x = np.array(x_list)
    y = np.array(y_list)

    return x,y, confidences, scales, orientations

    # raise NotImplementedError('`get_interest_points` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose Forstner_Harris value is significantly (10%)              #
    # greater than that of all of its neighbors within a interest_point_neighborhood_list r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of interest_point_neighborhood_list r pixels. One way to do so is to sort all          #
    # points by the Forstner_Harris strength, from large to small Forstner_Harris.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any interest_point_neighborhood_list. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater Forstner_Harris strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # interest_point_neighborhood_list within which the current point is a local maximum. We              #
    # call this the suppression interest_point_neighborhood_list of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    # raise NotImplementedError('adaptive non-maximal suppression in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  ##  return x,y, confidences, scales, orientations

