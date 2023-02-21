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
  
        for F in (StartPage, PageOne ,TrainMode, Pool8Mode):  
  
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
        self.bgbutton = PhotoImage(file="BG-Button.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)
        button = Button(self,text="Start",bg="#C24E4E",fg="white", bd=0,activebackground="#C24E4E",font=self.Myfont(40),   
                            command=lambda: controller.show_frame(PageOne))  
        button.place(x=810,y=880,width=300,height=100)
        my_text = Entry(self, justify=CENTER,bg="#FFFFFF",bd=0, font=self.Myfont(50))
        my_text.insert(0, "Welcome")
        my_text.pack(padx=140, pady=140)
        my_tex1 = Entry(self, justify=CENTER,bg="#FFFFFF",bd=0, font=self.Myfont(50))
        my_tex1.insert(0, "Pool Billiard Assistant")
        my_tex1.pack(padx=0, pady=0)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont
    
  
class PageOne(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="iMac - 14.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        button2 = Button(self,text="Training",bg="#FFFFFF",fg="Black", bd=0,activebackground="#FFFFFF",font=self.Myfont(90),   
                            command=lambda: controller.show_frame(TrainMode))  
        button2.pack(pady=140,padx=140)
        button3 = Button(self,text="8-Pool",bg="Black",fg="#FFFFFF", bd=0,activebackground="Black",font=self.Myfont(90),   
                            command=lambda: controller.show_frame(Pool8Mode))  
        button3.pack(pady=110,padx=110)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class TrainMode(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="bgTraingmode.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        button = Button(self,text="Training",bg="#FFFFFF",fg="Black", bd=0,activebackground="#FFFFFF",font=self.Myfont(40),   
                            command=lambda: controller.show_frame(PageOne))  
        button.place(x=530,y=880,width=300,height=100)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class Pool8Mode(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="bgpool8.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        button = Button(self,text="Training",bg="#FFFFFF",fg="Black", bd=0,activebackground="#FFFFFF",font=self.Myfont(40),   
                            command=lambda: controller.show_frame(PageOne))  
        button.place(x=530,y=880,width=300,height=100)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont
         
app = Sampleapp()  
app.title("Pool-Billiard")
app.mainloop()  