import sys
import cv2
import numpy as np
import math

# Upload the image in the same directory and then run the code
# If image not found you may get an error
# Reading Image


# This class holds x and y coordinates of a point
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

# This class holds all information for a single Region
class Region:
    regionCount = 0     # common variable to all instances, used for indexing regions
    global image
    global threshold
    global out_img
    global mask_img

    def __init__(self, seed: Point, color):
        Region.regionCount += 1
        self.regionId = Region.regionCount
        self.seed = seed
        self.color = color
        self.points = []
        self.points.append(self.seed)
        self.tmp_points = self.points
        self.done = False

    def grow(self):
        # Stop when we have no points left to process
        if len(self.tmp_points) <= 0:
            self.done = True
            return

        # Take the first point (Breadth-first search)
        center = self.tmp_points.pop(0)

        for coord in get8Connexity(center.getX(), center.getY(), image.shape):
            # R : src[coord[0], coord[1], 2]
            # G : src[coord[0], coord[1], 1]
            # B : src[coord[0], coord[1], 0]
            # print(self.seed.x, self.seed.y, coord.x, coord.y)
            deltaR = pow(int(image[self.seed.getY(), self.seed.getX(), 2]) - int(image[coord.getY(), coord.getX(), 2]), 2)
            deltaG = pow(int(image[self.seed.getY(), self.seed.getX(), 1]) - int(image[coord.getY(), coord.getX(), 1]), 2)
            deltaB = pow(int(image[self.seed.getY(), self.seed.getX(), 0]) - int(image[coord.getY(), coord.getX(), 0]), 2)

            delta = deltaR + deltaG + deltaB
            delta = math.sqrt(delta)
            delta = delta / 3

            # We use a mask of the original image to mark pixels already visited
            if mask_img[coord.getY(), coord.getX()] == 0 and delta < threshold:
                mask_img[coord.getY(), coord.getX()] = 255      # mark the pixel in the mask image
                out_img[coord.getY(), coord.getX(), 2] = self.color[2]
                out_img[coord.getY(), coord.getX(), 1] = self.color[1]
                out_img[coord.getY(), coord.getX(), 0] = self.color[0]
                self.tmp_points.append(coord)   # hold current points we have to visit
                self.points.append(coord)   # hold all the points of the region

    def getSeed(self) -> Point:
        return self.seed


# This function returns an array containing the 8-connexity of a point 
def get8Connexity(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    # bottom left
    outx = min(max(x-1, 0), maxx)
    outy = min(max(y+1, 0), maxy)
    out.append(Point(outx, outy))

    # bottom center
    outx = x
    outy = min(max(y+1, 0), maxy)
    out.append(Point(outx, outy))

    # bottom right
    outx = min(max(x+1, 0), maxx)
    outy = min(max(y+1, 0), maxy)
    out.append(Point(outx, outy))

    # right
    outx = min(max(x+1, 0), maxx)
    outy = y
    out.append(Point(outx, outy))

    # top right
    outx = min(max(x+1, 0), maxx)
    outy = min(max(y-1, 0), maxy)
    out.append(Point(outx, outy))

    # top center
    outx = x
    outy = min(max(y-1, 0), maxy)
    out.append(Point(outx, outy))

    # top left
    outx = min(max(x-1, 0), maxx)
    outy = min(max(y-1, 0), maxy)
    out.append(Point(outx, outy))

    # left
    outx = min(max(x-1, 0), maxx)
    outy = y
    out.append(Point(outx, outy))

    return out


# Main function for region growing algorithm
# Returns the result image
def region_growing_without_priority(img_src, seeds):
    global out_img
    global mask_img
    global colors

    out_img = np.zeros_like(img_src)
    mask_img = np.zeros_like(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY))

    # -- Initialize regions with seeds and colors
    regions = []
    for idx, seed in enumerate(seeds):
        regions.append(Region(seed, colors[idx]))

    print("[ REGIONS INITIALIZED ] ")
    for idx, region in enumerate(regions):
        print("-- Region n°", idx, " region.seed: ", region.getSeed().getX(), region.getSeed().getY())

    # -- While all the regions have not done their growing, then continue to grow
    while(any(region.done == False for region in regions)):
        for region in regions:
            region.grow()

    print("[ REGION GROWING DONE ]")

    return out_img


def region_growing_with_priority(src, seeds, threshold):
    global colors

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    out_img = np.zeros_like(gray)
    out_img_color = np.zeros_like(src)

    id_region = 0

    for seed in seeds:

        points_stack = []
        points_stack.append(seed)

        while(len(points_stack) > 0):

            center = points_stack.pop(0)

            for coord in get8Connexity(center.getX(), center.getY(), src.shape):

                # R : src[coord[0], coord[1], 2]
                # G : src[coord[0], coord[1], 1]
                # B : src[coord[0], coord[1], 0]
                deltaR = pow(int(src[seed.getY(), seed.getX(), 2]) - int(src[coord.getY(), coord.getX(), 2]), 2)
                deltaG = pow(int(src[seed.getY(), seed.getX(), 1]) - int(src[coord.getY(), coord.getX(), 1]), 2)
                deltaB = pow(int(src[seed.getY(), seed.getX(), 0]) - int(src[coord.getY(), coord.getX(), 0]), 2)

                delta = deltaR + deltaG + deltaB
                delta = math.sqrt(delta)
                delta = delta / 3

                if out_img[coord.getY(), coord.getX()] == 0 and delta < threshold:
                    out_img[coord.getY(), coord.getX()] = 255

                    if id_region > len(colors)-1:
                        colors = list(np.random.choice(range(256), size=3))
                        out_img_color[coord.getY(), coord.getX(), 2] = colors[2]
                        out_img_color[coord.getY(), coord.getX(), 1] = colors[1]
                        out_img_color[coord.getY(), coord.getX(), 0] = colors[0]
                    else:
                        out_img_color[coord.getY(), coord.getX(), 2] = colors[id_region][2]
                        out_img_color[coord.getY(), coord.getX(), 1] = colors[id_region][1]
                        out_img_color[coord.getY(), coord.getX(), 0] = colors[id_region][0]
                        
                    points_stack.append(coord)

        id_region += 1

    return out_img_color


def on_mouse(event, x, y, flags, params):
    global seeds

    if event == cv2.EVENT_LBUTTONDOWN:
        print('Clicked on: ' + str(x) + ', ' + str(y), image[y, x])
        seeds.append(Point(x,y))


def main():
    global seeds
    global image
    global colors
    global threshold

    colors = [(255, 87, 34), (255, 193, 7), (205, 220, 57), (76, 175, 80), (0, 188, 212), (33, 150, 243), (121, 85, 72), (130, 119, 23), (0, 102, 102), (255, 204, 204)]

    ################################### Image Dame ##################################
    # Positions germes
    # main gauche, x: 118, y:205
    # tete, x:201 y:72
    # main droite, x:295 y:207
    # torse, x:200 y:234
    # jambe gauche, x:187 y:412
    # jambe droite, x:224 y:407
    # cou, x: 199 y: 137
    # cheveux, x: 206, y: 54
    image = cv2.imread('data/personne.jpeg')
    threshold = 20
    if len(sys.argv) >= 2:
        image = cv2.imread(sys.argv[1])
    if len(sys.argv) >= 3:
        threshold = int(sys.argv[2])
    image = cv2.GaussianBlur(image, (5, 5), 0)

    cv2.imshow('Original Image', image)
    cv2.setMouseCallback('Original Image', on_mouse, 0,)

    seeds = []

    print("Cliquez sur l'image pour poser vos germes ou appuyer sur une touche pour continuer.")
    cv2.waitKey()

    if len(seeds) == 0:
        seeds = [Point(118, 205), Point(201, 72), Point(295, 207), Point(206, 54), Point(200, 234), Point(187, 412), Point(224, 407), Point(199, 137)]


    out_img_without_priority = region_growing_without_priority(image, seeds)
    out_img_with_priority = region_growing_with_priority(image, seeds, threshold)

    cv2.imshow('Region Growing with priority', out_img_with_priority)
    cv2.imshow('Region Growing without priority', out_img_without_priority)


    ################################### Image Dessin Anime ##################################

    # # Points bonhomme dessin animé
    # # Tete (259,61)
    # # Bras gauche (191, 246)
    # # Torse (257, 183)
    # # Bras droit (328, 244)
    # # Jambe gauche (223, 382)
    # # Jambe droite (290, 380)

    # image = cv2.imread('data/personne.png')

    # threshold = 20

    # cv2.imshow('Original Image', image)
    # cv2.setMouseCallback('Original Image', on_mouse, 0,)

    # seeds = []

    # print("Cliquez sur l'image pour poser vos germes ou appuyer sur une touche pour continuer.")
    # cv2.waitKey()

    # if len(seeds) == 0:
    #     seeds = [Point(259, 61), Point(191, 246), Point(257, 183), Point(328, 244), Point(223, 382), Point(290, 380)]


    # out_img_without_priority = region_growing_without_priority(image, seeds)
    # out_img_with_priority = region_growing_with_priority(image, seeds, threshold)

    # cv2.imshow('Region Growing with priority', out_img_with_priority)
    # cv2.imshow('Region Growing without priority', out_img_without_priority)


    print("\nPress ESC to close all windows.")
    k = cv2.waitKey(0) & 0xFF
    if k == 27:  # close on ESC key
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
