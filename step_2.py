import numpy as np
import cv2

# Open Video
cap = cv2.VideoCapture('data/test2.avi')
cap_vierge = cv2.VideoCapture('data/vierge.avi')
bg_img = cv2.imread('data/bg_image01.png')

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

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while(ret):

  # Read frame
  ret, frame_rgb = cap.read()
  if ret == False:
    break

  # Resize the current frame to match the frame we want to compare with
  v_width = medianFrame.shape[1]
  v_height = medianFrame.shape[0]
  vierge_size = (v_width, v_height)
  frame_rgb = cv2.resize(frame_rgb, vierge_size)

  # Convert current frame to grayscale
  frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
  
  #frame = cv2.GaussianBlur(frame,(5,5),0)

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

  # Display image
  cv2.imshow('frame', frame_rgb)
  cv2.waitKey(20)

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
