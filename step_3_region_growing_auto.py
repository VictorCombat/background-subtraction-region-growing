import sys
import cv2
import numpy as np
import math
import random

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
    global image2
    global threshold
    global out_img
    global mask_img

    def __init__(self, seed: Point, color, area):
        Region.regionCount += 1
        self.regionId = Region.regionCount
        self.seed = seed
        self.color = color
        self.points = []
        self.points.append(self.seed)
        self.tmp_points = self.points
        self.done = False

        self.area = area

    def grow(self):
        # Stop when we have no points left to process
        if len(self.tmp_points) <= 0:
            self.done = True
            return

        # Take the first point (Breadth-first search)
        center = self.tmp_points.pop(0)

        for coord in get8Connexity(center.getX(), center.getY(), self.area, image2.shape):
            # R : src[coord[0], coord[1], 2]
            # G : src[coord[0], coord[1], 1]
            # B : src[coord[0], coord[1], 0]
            # print(self.seed.x, self.seed.y, coord.x, coord.y)
            deltaR = pow(int(image2[self.seed.getY(), self.seed.getX(), 2]) - int(image2[coord.getY(), coord.getX(), 2]), 2)
            deltaG = pow(int(image2[self.seed.getY(), self.seed.getX(), 1]) - int(image2[coord.getY(), coord.getX(), 1]), 2)
            deltaB = pow(int(image2[self.seed.getY(), self.seed.getX(), 0]) - int(image2[coord.getY(), coord.getX(), 0]), 2)

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


# This function returns an array containing the 8-connexity of a point  limited by an area
def get8Connexity(x, y, area, shape):
    out = []
    maxx = min(area[0] + area[3]-1, shape[1]-1)
    maxy = min(area[1] + area[2]-1, shape[0]-1)
    minx = area[0]
    miny = area[1]

    # bottom left
    outx = min(max(x-1, minx), maxx)
    outy = min(max(y+1, miny), maxy)
    out.append(Point(outx, outy))

    # bottom center
    outx = x
    outy = min(max(y+1, miny), maxy)
    out.append(Point(outx, outy))

    # bottom right
    outx = min(max(x+1, minx), maxx)
    outy = min(max(y+1, miny), maxy)
    out.append(Point(outx, outy))

    # right
    outx = min(max(x+1, minx), maxx)
    outy = y
    out.append(Point(outx, outy))

    # top right
    outx = min(max(x+1, minx), maxx)
    outy = min(max(y-1, miny), maxy)
    out.append(Point(outx, outy))

    # top center
    outx = x
    outy = min(max(y-1, miny), maxy)
    out.append(Point(outx, outy))

    # top left
    outx = min(max(x-1, minx), maxx)
    outy = min(max(y-1, miny), maxy)
    out.append(Point(outx, outy))

    # left
    outx = min(max(x-1, minx), maxx)
    outy = y
    out.append(Point(outx, outy))

    return out

# Main function for region growing algorithm
# Returns the result image
def region_growing(img_src, seeds):
    global regions
    global out_img
    global mask_img

    out_img = np.zeros_like(img_src)
    mask_img = np.zeros_like(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY))

    # -- While all the regions have not done their growing, then continue to grow
    while(any(region.done == False for region in regions)):
        for region in regions:
            region.grow()

    return out_img

# -- Initialize regions with seeds and colors
def init_seeds(min_x, min_y, max_x, max_y, grid_size):
    global regions
    global image2
    global select_frame_rgb
    global frame_mask

    seeds = []

    for x in range(min_x, max_x, grid_size):
        cv2.line(select_frame_rgb, (x, min_y), (x, max_y), (255, 255, 0), 1, 1)
        for y in range(min_y, max_y, grid_size):
            cv2.line(select_frame_rgb, (min_x, y), (max_x, y), (255, 255, 0), 1, 1)

            white_pixels = []
            for yy in range(y, min(y + grid_size - 1, max_y - 1)):
                for xx in range(x, min(x + grid_size - 1, max_x - 1)):
                    if frame_mask[yy, xx] == 255:
                        white_pixels.append((yy, xx))

            if len(white_pixels) != 0:
                idx_x = random.randint(0, len(white_pixels)-1)
                idx_y = random.randint(0, len(white_pixels)-1)
                seed_x = white_pixels[idx_x][1]
                seed_y = white_pixels[idx_x][0]
                seed = Point(seed_x, seed_y)
                seeds.append(seed)
                color = list(np.random.choice(range(256), size=3)) 
                regions.append(Region(seed, color, (x, y, x + grid_size - 1, y + grid_size - 1)))

                cv2.circle(select_frame_rgb, (seed_x, seed_y), radius=2, color=(0, 255, 0), thickness=-1)

    return seeds

# This function will compute the envelope of the mask's frame
# It returns an array of 4 coordinates : min_x, min_y, max_x, max_y
def compute_envelope():
    height, width = frame_mask.shape
    min_x = width + 1
    min_y = height + 1
    max_x = 0
    max_y = 0
    for x in range(0, width -1):
        for y in range(0, height -1):
            if frame_mask[y,x] == 255:
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

    # Draw bbox
    cv2.line(select_frame_rgb, (min_x, min_y), (min_x, max_y), (255, 0, 0), 1, 1)
    cv2.line(select_frame_rgb, (min_x, min_y), (max_x, min_y), (255, 0, 0), 1, 1)
    cv2.line(select_frame_rgb, (max_x, min_y), (max_x, max_y), (255, 0, 0), 1, 1)
    cv2.line(select_frame_rgb, (min_x, max_y), (max_x, max_y), (255, 0, 0), 1, 1)

    return (min_x, min_y, max_x, max_y)


def extract_silhouette(cap, cap_vierge, bg_img):
    global select_frame_rgb
    global frame_mask

    # Randomly select 25 frames
    frameIds = cap_vierge.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap_vierge.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap_vierge.read()
        frames.append(frame)

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    # Reset frame number to 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Convert background to grayscale
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    # Loop over all frames
    frame_nb = 5
    frame_idx = 0
    ret = True
    frame_mask = None
    select_frame_rgb = None
    bg_img_resize = None
    while(ret):
        # Read frame
        ret, frame_rgb = cap.read()
        if ret == False:
            break

        if frame_idx < frame_nb:
            # Resize the current frame to match the frame we want to compare with
            v_width = medianFrame.shape[1]
            v_height = medianFrame.shape[0]
            vierge_size = (v_width, v_height)
            frame_rgb = cv2.resize(frame_rgb, vierge_size)

            # Convert current frame to grayscale
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference of current frame and 
            # the median frame
            dframe = cv2.absdiff(frame, grayMedianFrame)
            # Treshold to binarize
            th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

            # resize the background image to match the mask size
            width = dframe.shape[1]
            height = dframe.shape[0]
            dsize = (width, height)
            bg_img_resize = cv2.resize(bg_img, dsize)

            # Get all black pixels from the mask 
            bg = np.where(dframe == 0)
            # Change all corresponding pixels from the current frame to the background pixels
            frame_rgb[bg] = bg_img_resize[bg]

            select_frame_rgb = frame_rgb.copy()

            frame_mask = dframe.copy()

        frame_idx += 1


def main():
    global image2
    global regions
    global threshold
    global seeds

    threshold = 20
    grid_size = 50
    if len(sys.argv) >= 2:
        threshold = int(sys.argv[1])
    if len(sys.argv) >= 3:
        grid_size = int(sys.argv[2])

    cap = cv2.VideoCapture('data/test1.avi')
    cap_vierge = cv2.VideoCapture('data/vierge.avi')
    bg_img = cv2.imread('data/bg_image01.png')

    extract_silhouette(cap, cap_vierge, bg_img)

    # Release video object
    cap.release()

    image2 = select_frame_rgb.copy()
    regions = []
    envelope = compute_envelope()
    seeds = init_seeds(envelope[0], envelope[1], envelope[2], envelope[3], grid_size)

    bg = np.where(frame_mask == 0)
    image2[bg] = (0, 255, 0) # Replace background with green color to get better results with region growing algorithm

    output = region_growing(image2, seeds)

    cv2.imshow("Selected Frame", select_frame_rgb)
    cv2.imshow("Image2", image2)
    cv2.imshow("Output", output)

    k = cv2.waitKey(0) & 0xFF
    if k == 27:  # close on ESC key
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()