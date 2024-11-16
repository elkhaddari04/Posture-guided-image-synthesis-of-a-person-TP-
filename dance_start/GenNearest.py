import numpy as np
import cv2
import os
import pickle
import sys
import math
from sklearn.neighbors import NearestNeighbors


class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """

    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt
        self.cached_images = []
        print("Caching images...")

        # Cache all images at initialization
        for i in range(len(self.videoSkeletonTarget.ske)):
            img = self.videoSkeletonTarget.readImage(i)
            img = img.astype(np.float32) / 255.0
            self.cached_images.append(img)

        print(f"Cached {len(self.cached_images)} images")

    def generate(self, ske):
        """ generator of image from skeleton """
        if ske is None:
            return self.cached_images[0]

        # Calculate distances using original method
        distances = []
        for target_ske in self.videoSkeletonTarget.ske:
            distance = ske.distance(target_ske)
            distances.append(distance)

        # Find index of closest skeleton
        closest_idx = np.argmin(distances)

        # Return cached normalized image
        return self.cached_images[closest_idx]