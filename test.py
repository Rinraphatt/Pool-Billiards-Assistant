from PIL import Image, ImageTk
import tkinter as tk

root = tk.Tk()

img1 = tk.PhotoImage(file="iMac - 18.png")
img2 = tk.PhotoImage(file="iMac - 17.png")

# Get the size of the PhotoImage objects using the width() and height() methods
img1_size = (img1.width(), img1.height())
img2_size = (img2.width(), img2.height())

# Create PIL Image objects from the PhotoImage objects using the ImageTk.getimage() method
img1_pil = ImageTk.getimage(img1)
img2_pil = ImageTk.getimage(img2)

# Resize the images using the resize() method
img1_pil = img1_pil.resize((400, 400))
img2_pil = img2_pil.resize((400, 400))

# Create a copy of img2_pil and set the alpha channel to a lower value
img2_pil_transparent = img2_pil.copy()
alpha = 100  # set the alpha value here (0-255)
img2_pil_transparent.putalpha(alpha)

# Composite the images using alpha_composite()
result_pil = Image.alpha_composite(img1_pil, img2_pil_transparent)

# Create a PhotoImage object from the PIL Image object
result = ImageTk.PhotoImage(result_pil)

# Create a Label widget and display the composite image on it
label = tk.Label(root, image=result)
label.pack()

root.mainloop()
