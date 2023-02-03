import tkinter as tk               
from tkinter import Tk, Canvas, Frame, BOTH
from tkinter import *
from tkinter import font
from typing import Sized
from PIL import Image, ImageTk
from tkinter.font import Font

  
class Sampleapp(tk.Tk):  
  
    def __init__(self, *args, **kwargs):  
          
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("1920x1080")
        container = tk.Frame(self)  
        container.pack(side="top", fill="both", expand = True)  
        container.grid_rowconfigure(0, weight=1)  
        container.grid_columnconfigure(0, weight=1)  
        self.frames = {}  
  
        for F in (StartPage, PageOne):  
  
            frame = F(container, self)  
            self.frames[F] = frame  
            frame.grid(row=0, column=0, sticky="nsew")  

        self.show_frame(StartPage)  
  
    def show_frame(self, cont):  
        frame = self.frames[cont]  
        frame.tkraise()  
  
          
class StartPage(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self,parent)
        self.bg = PhotoImage(file="Bg-Firstpage.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)
        button = Button(self, text="Visit Page 1",font=self.Myfont(40),  
                            command=lambda: controller.show_frame(PageOne))    
        button.pack()

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont
  
  
class PageOne(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent)  
        label = tk.Label(self, text="Page One!!!", font=self.Myfont(40))  
        label.pack(pady=10,padx=10)  
  
        button1 = tk.Button(self, text="Back to Home",  
                            command=lambda: controller.show_frame(StartPage))  
        button1.pack()  

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont
  
         
app = Sampleapp()  
app.title("Pool-Billiard")
app.mainloop()  