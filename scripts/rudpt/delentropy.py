#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Trying to reproduce: https://arxiv.org/abs/1609.01117

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import rosbag
import seaborn as sns
from tqdm import tqdm
import time
import argparse
from pathlib import Path
np.float = np.float64
np.int = np.int64

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def calculate_delentropy(image, gamma = 1.0):
    # Using a 2x2 difference kernel [[-1,+1],[-1,+1]] results in artifacts!
    # In tests the deldensity seemed to follow a diagonal because of the
    # assymetry introduced by the backward/forward difference
    # the central difference correspond to a convolution kernel of
    # [[-1,0,1],[-1,0,1],[-1,0,1]] and its transposed, produces a symmetric
    # deldensity for random noise.
    # see paper eq. (4)
    fx = ( image[:,2:] - image[:,:-2] )[1:-1,:]
    fy = ( image[2:,:] - image[:-2,:] )[:,1:-1]

    diff_range = np.max( [ np.abs( fx.min() ), np.abs( fx.max() ), np.abs( fy.min() ), np.abs( fy.max() ) ] )
    if diff_range >= 200 and diff_range <= 255  : diff_range = 255

    # see paper eq. (17)
    # The bin edges must be integers, that's why the number of bins and range depends on each other
    nBins = 2*diff_range+1
    
    # Centering the bins is necessary because else all value will lie on
    # the bin edges thereby leading to assymetric artifacts
    dbin = 0.5
    r = diff_range + dbin
    del_density, xedges, yedges = np.histogram2d( fx.flatten(), fy.flatten(), bins = nBins, range = [ [-r,r], [-r,r] ] )
    assert( xedges[1] - xedges[0] == 1.0 )
    assert( yedges[1] - yedges[0] == 1.0 )

    # Normalization for entropy calculation. np.sum( H ) should be ( imageWidth-1 )*( imageHeight-1 )
    # The -1 stems from the lost pixels when calculating the gradients with non-periodic boundary conditions
    #assert( np.product( np.array( image.shape ) - 1 ) == np.sum( del_density ) )
    del_density = del_density / np.sum( del_density ) # see paper eq. (17)
    del_density = del_density.T
    
    # "The entropy is a sum of terms of the form p log(p). When p=0 you instead use the limiting value (as p approaches 0 from above), which is 0."
    # The 0.5 factor is discussed in the paper chapter "4.3 Papoulis generalized sampling halves the delentropy"
    del_entropy = - 0.5 * np.sum( del_density[ del_density.nonzero() ] * np.log2( del_density[ del_density.nonzero() ] ) ) # see paper eq. (16)
    
    # gamma enhancements and inversion for better viewing pleasure
    del_density = np.max(del_density) - del_density
    del_density = ( del_density / np.max( del_density ) )**gamma * np.max( del_density )
    
    del_dict = {
        "del_entropy": del_entropy,
        "del_density": del_density,
        "fx": fx,
        "fy": fy,
        "xedges": xedges,
        "yedges": yedges,
        "diff_range": diff_range
    }
    
    return del_dict

def delentropy_of_bag(bag_file, image_topic, gamma = 1.0):
    # Iterate through each message on the specified topic
    bridge = CvBridge()
    with rosbag.Bag(bag_file, 'r') as bag:
        frame_count = bag.get_message_count(topic_filters=[image_topic])
        
        del_entropies = np.zeros(frame_count)
        del_densities = np.zeros((frame_count, 511, 511))
        for i, (_, msg, _) in tqdm(enumerate(bag.read_messages(topics=[image_topic])), total=frame_count, desc="Calculating delentropy"):
            # Convert ROS image message to OpenCV format
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

            # Calculate the delentropy for each frame of the video
            del_dict = calculate_delentropy(frame, gamma)
            del_entropies[i] = del_dict["del_entropy"]
            del_densities[i] = del_dict["del_density"]

    return del_entropies, del_densities

if __name__ == "__main__":
    # High delentropy: Gradient values are uniformly distributed, indicating high randomness and little structure.
    # Low delentropy: Gradient values are concentrated (or zero), indicating a smooth, featureless image.
    # Intermediate delentropy: Provides the best conditions for extracting meaningful features
    
    parser = argparse.ArgumentParser(
        description='''Compute delentropy of the input. Either image (.png, .jpg) or video (.bag).''')
    parser.add_argument(
        'input_file', type=str,
        help="File path to the image/video.")
    parser.add_argument(
        '--image_topic', type=str, default="/camera/image_raw",
        help="Name of the image topic in the bag.")
    parser.add_argument(
        '--gamma', type=float, default=1.0,
        help="Gamma enhancement factor.")
    parser.add_argument(
        '--save', dest='save', action='store_true')
    parser.set_defaults(save=False)
    args = parser.parse_args()
    
    if args.input_file.endswith((".png", ".jpg")):
        # Calculate the delentropy of an image
        image = cv.imread(args.input_file, cv.IMREAD_GRAYSCALE)
        del_dict = calculate_delentropy(image, args.gamma)

        fig = plt.figure(figsize = (10,8))
        ax = [
            fig.add_subplot( 221, title = "Original image (Delentropy: {:.2f}".format(del_dict["del_entropy"]) + ")"),
            fig.add_subplot( 222, title = "x gradient of image"),
            fig.add_subplot( 223, title = "y gradient of image"),
            fig.add_subplot( 224, title = "Histogram of gradient (Deldensity)")
        ]

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[1].imshow(del_dict["fx"], cmap=plt.cm.gray, vmin = -del_dict["diff_range"], vmax = del_dict["diff_range"])
        ax[2].imshow(del_dict["fy"], cmap=plt.cm.gray, vmin = -del_dict["diff_range"], vmax = del_dict["diff_range"])
        ax[3].imshow(del_dict["del_density"]  , cmap=plt.cm.gray, vmin = 0, interpolation='nearest', origin='lower',
                extent = [del_dict["xedges"][0], del_dict["xedges"][-1], del_dict["yedges"][0], del_dict["yedges"][-1]])

        fig.tight_layout()
        
    elif args.input_file.endswith(".bag"):
        # Calculate the delentropy of a video
        del_entropies, del_densities = delentropy_of_bag(args.input_file, args.image_topic, args.gamma)
        
        # Create violin plot of delentropy
        fig = plt.figure(figsize = (4,4))
        ax1 = fig.add_subplot(111, title = "Delentropy of " + Path(args.input_file).stem)
        sns.violinplot(x=np.full(del_entropies.shape[0], Path(args.input_file).stem), y=del_entropies, ax=ax1)
        ax1.set_ylim((0.0, np.max(del_entropies)+0.5))
        
    else:
        raise ValueError("Input file must be an image (.png, .jpg) or video (.bag).")

    if args.save:
        save_path = Path(__file__).parent / f"assets/{Path(args.input_file).stem}-delentropy.png"
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
