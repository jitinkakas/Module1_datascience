#Project: Grayscale Image Manipulator
#Objective: Understand that images are just matrices — manipulate pixel values using NumPy operations.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# STEP 1: Load a  Image
img=mpimg.imread("img1.jpg")  # Replace with your image path
print(type(img))
print(img.shape)
print(img.dtype,"\n","\n")

# To show aan image
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Make it grayscale
img_gray=np.mean(img,axis=2) # Average across color channels
print("shape of grayscale image:",img_gray.shape,"\n")

plt.imshow(img_gray,cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()


# Increasing brightness
bright=img+50
bright_img=np.clip(bright,0,255) #It is used to limit the values in an array to a specified range. In this case, it ensures that the pixel values do not exceed 255 (the maximum for 8-bit images) or go below 0.

plt.imshow(bright_img.astype(np.uint8))
plt.title("bright image")
plt.axis('off')
plt.show()

# Decreasing brightness
dark=img-50
dark_img=np.clip(dark,0,255) #It is used to limit the values in an array to a specified range. In this case, it ensures that the pixel values do not exceed 255 (the maximum for 8-bit images) or go below 0.

plt.imshow(dark_img.astype(np.uint8))
plt.title("dark image")
plt.axis('off')
plt.show()

# horizontal flip
H_flip=img[:,::-1] # img[:, ::-1] == First part ":" means: take ALL rows of image &&& Second part "::-1" means: take ALL columns of image but in reverse order. So, it flips the image horizontally.

plt.imshow(H_flip)
plt.title("Horizontal fliped image")
plt.axis('off')
plt.show()

# vertical flip
V_flip=img[::-1,:] # img[::-1, :] == First part "::-1" means: take ALL rows of image but in reverse order &&& Second part is ":": take ALL columns of image. So, it flips the image vertically.

plt.imshow(V_flip)
plt.title("Vertical fliped image")
plt.axis('off')
plt.show()

# crop an image
crop_img=img[100:400,50:100] # img[100:400, 50:100] == First part "100:400" means: take rows from index 100 to 399 &&& Second part "50:100" means: take columns from index 50 to 99. So, it crops the image to the specified region.

plt.imshow(crop_img)
plt.title("cropped image")
plt.axis('off')
plt.show()

# Display the original and manipulated images side by side
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(img) # Display the original image
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(img_gray,cmap="gray") # Display the grayscale image with a colormap of "gray" to ensure it is shown in shades of gray
plt.title("Gray")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(bright.astype(np.uint8)) # Display the brightened image, converting it to uint8 to ensure pixel values are in the correct range for display
plt.title("Bright")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(dark.astype(np.uint8)) # Display the darkened image, converting it to uint8 to ensure pixel values are in the correct range for display
plt.title("Dark")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(H_flip) # Display the horizontally flipped image
plt.title("H Flip")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(crop_img) # Display the cropped image
plt.title("Crop")
plt.axis("off")

plt.tight_layout()
plt.show() 

#Bonus: Try applying a simple filter (like a blur or edge detection) using convolution operations on the image matrix.

blur = (
    img[:-2, :-2] + img[1:-1, :-2] + img[2:, :-2] + ## Top-Left      Top       Top-Right
    img[:-2, 1:-1] + img[1:-1, 1:-1] + img[2:, 1:-1] + ## Left          Center    Right
      img[:-2, 2:] + img[1:-1, 2:] + img[2:, 2:] ## Bottom-Left   Bottom    Bottom-Right
) / 9

plt.imshow(blur.astype(np.uint8))
plt.title("Blur Image")
plt.axis("off")
plt.show()