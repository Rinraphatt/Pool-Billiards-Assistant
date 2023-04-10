# import tkinter as tk
# from PIL import Image, ImageTk

# root = tk.Tk()

# # Load the first image as a PhotoImage object
# img1 = tk.PhotoImage(file="iMac - 18.png")
# img2 = tk.PhotoImage(file="iMac - 17.png")

# img1_size = (img1.width(), img1.height())
# img2_size = (img2.width(), img2.height())

# # Convert both PhotoImage objects to PIL Image objects
# img1_pil = ImageTk.getimage(img1)
# img2_pil = ImageTk.getimage(img2)

# # Create a new RGBA image with the same size as the first image
# new_img_pil = Image.new("RGBA", img1_size, (0, 0, 0, 0))

# # Composite the second image on the new image
# new_img_pil = Image.alpha_composite(new_img_pil, img2_pil)

# # Convert the new image to a transparent PhotoImage object
# new_img = ImageTk.PhotoImage(new_img_pil)

# # Create a Label widget and display the new image on it
# label = tk.Label(root, image=new_img)
# label.pack()

# root.mainloop()

import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()

# Load the first image as a PhotoImage object
img1 = ImageTk.PhotoImage(file="iMac - 18.png")

# Load the second image as a PhotoImage object
img2 = ImageTk.PhotoImage(file="iMac - 17.png")

img1.data = img1.tk.call("data", img1.cget("name"), "-format", "rgba")
img2.data = img2.tk.call("data", img1.cget("name"), "-format", "rgba")

# Convert both PhotoImage objects to PIL Image objects
img1_pil = Image.frombytes("RGBA", (img1.width(), img1.height()), img1.data)
img2_pil = Image.frombytes("RGBA", (img2.width(), img2.height()), img2.data)

# Create a new RGBA image with the same size as the first image
new_img_pil = Image.new("RGBA", img1_pil.size, (0, 0, 0, 0))

# Composite the second image on the new image
new_img_pil = Image.alpha_composite(new_img_pil, img2_pil)

# Convert the new image to a transparent PhotoImage object
new_img = ImageTk.PhotoImage(new_img_pil)

# Create a Label widget and display the new image on it
label = tk.Label(root, image=new_img)
label.pack()

root.mainloop()

