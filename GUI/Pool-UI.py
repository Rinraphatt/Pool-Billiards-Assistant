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
        self.geometry("1920x880")
        container = tk.Frame(self)  
        container.pack(side="top", fill="both", expand = True)  
        container.grid_rowconfigure(0, weight=1)  
        container.grid_columnconfigure(0, weight=1)  
        self.frames = {}  
  
        for F in (StartPage, PageOne ,TrainMode, Pool8Mode, ModeBasic, ModeAmature, Result8Pool, Stage, Play8Pool):  
  
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
        button.place(x=825,y=717,width=280,height=90)
        lbl1 = Label(self, text="Welcome", bg="White" ,fg="Black", font=self.Myfont(50), anchor=CENTER)
        lbl1.pack(padx=80, pady=80)
        lbl2 = Label(self, text="Pool Billiard Assistant", bg="White" ,fg="Black", font=self.Myfont(50), anchor=CENTER)
        lbl2.pack(padx=0, pady=0)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont
    
  
class PageOne(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="iMac - 14.png")
        self.btnback = PhotoImage(file="left-arrow - Copy.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0,y=0)

        buttontrain = Button(self,text="Training",bg="#FFFFFF",fg="Black", bd=0,activebackground="#FFFFFF",font=self.Myfont(90),   
                            command=lambda: controller.show_frame(TrainMode))  
        buttontrain.pack(pady=130,padx=130)
        button8pool = Button(self,text="8-Pool",bg="Black",fg="#FFFFFF", bd=0,activebackground="Black",activeforeground="#FFFFFF",font=self.Myfont(90),   
                            command=lambda: controller.show_frame(Pool8Mode))  
        button8pool.pack(pady=10,padx=10)

        buttonback = Button(self,image=self.btnback,bg="#FFFFFF",bd=0,activebackground="#FFFFFF",
                            command=lambda: controller.show_frame(StartPage))
        buttonback.place(x=20,y=20,width=100,height=100)


    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class TrainMode(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="bgTraingmode.png")
        self.btnback = PhotoImage(file="left-arrow.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        # button
        btn_basic = Button(self,text="Basic", fg="#FFFFFF",bg="Black",bd=0,activebackground="#FFFFFF",font=self.Myfont(40),
                                command=lambda: controller.show_frame(ModeBasic))
        btn_basic.place(x=262,y=420,width=405,height=320)
        btn_Amature = Button(self,text="Amature", fg="#FFFFFF",bg="Black",bd=0,activebackground="#FFFFFF",font=self.Myfont(40),
                                command=lambda: controller.show_frame(ModeAmature))
        btn_Amature.place(x=764,y=420,width=405,height=320)
        btn_Custom = Button(self,text="Custom", fg="#FFFFFF",bg="Black",bd=0,activebackground="#FFFFFF",font=self.Myfont(40),)
        btn_Custom.place(x=1265,y=420,width=405,height=320)

        buttonback = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black",
                            command=lambda: controller.show_frame(StartPage))
        buttonback.place(x=20,y=20,width=100,height=100)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class Pool8Mode(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="bgpool8.png")
        self.btnback = PhotoImage(file="left-arrow.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        button2 = Button(self,text="Single Player",bg="#E22424",fg="#FFFFFF", bd=0,activebackground="#E22424",font=self.Myfont(50),
                            command=lambda: controller.show_frame(Play8Pool))  
        button2.place(x=273,y=680,width=620,height=130)
        button3 = Button(self,text="Multi Player",bg="#E22424",fg="#FFFFFF", bd=0,activebackground="#E22424",font=self.Myfont(50))  
        button3.place(x=1040,y=680,width=620,height=130)

        buttonback = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black",
                            command=lambda: controller.show_frame(StartPage))
        buttonback.place(x=20,y=20,width=100,height=100)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class ModeBasic(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent)
        self.bg = PhotoImage(file="basicmode.png")
        self.btnback = PhotoImage(file="left-arrow.png")
        self.btnnext = PhotoImage(file="arrow-forward.png")
        self.stage1 = PhotoImage(file="iMac - 18.png")
        self.stage2 = PhotoImage(file="iMac - 23.png")
        self.stage3 = PhotoImage(file="iMac - 24.png")

        self.num = [1,2,3]
        self.stage = 0
        self.state = False
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)
        buttonback = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black",
                            command=lambda: [controller.show_frame(StartPage),self.setnum()])
        buttonback.place(x=20,y=20,width=100,height=100)
        
        buttonnext = Button(self,image=self.btnnext,bg="Black",bd=0,activebackground="Black")
        buttonnext.bind("<Button-1>", self.addnum)
        buttonnext.place(x=1820,y=550,width=70,height=70)

        buttonbackstage = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black")
        buttonbackstage.bind("<Button-1>", self.deletenum)
        buttonbackstage.place(x=50,y=550,width=70,height=70)

        btnnum1 = Button(self,text=self.num[0],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[0]))
        btnnum1.place(x=262,y=435,width=405,height=310)
        btnnum2 = Button(self,text=self.num[1],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[1]))
        btnnum2.place(x=764,y=435,width=405,height=310)
        btnnum3 = Button(self,text=self.num[2],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[2]))
        btnnum3.place(x=1265,y=435,width=405,height=310)

    def showbtn(self):
        btnnum1 = Button(self,text=self.num[0],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[0]))
        btnnum1.place(x=262,y=435,width=405,height=310)
        btnnum2 = Button(self,text=self.num[1],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[1]))
        btnnum2.place(x=764,y=435,width=405,height=310)
        btnnum3 = Button(self,text=self.num[2],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[2]))
        btnnum3.place(x=1265,y=435,width=405,height=310)
 
    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

    def addnum(self,event):
        for i in range(len(self.num)):
            self.num[i]+= 3
            print(self.num)
        self.showbtn()

    def deletenum(self,event):
        for i in range(len(self.num)):
            if self.num[i] > 3 :
                self.num[i]-= 3
                print(self.num)
                self.showbtn()
    
    def close(self,stage):
        stage.destroy()
        self.showbtn()
        buttonnext = Button(self,image=self.btnnext,bg="Black",bd=0,activebackground="Black")
        buttonnext.bind("<Button-1>", self.addnum)
        buttonnext.place(x=1820,y=550,width=70,height=70)
        print(self.num)
    
    def setnum(self):
        self.num = [1,2,3]
        self.showbtn()
        print("set")

    def display(self,stagenum):
        self.stage = stagenum
        print(self.stage)
        if self.stage == 1 :
            self.img1_size = (self.stage1.width(), self.stage1.height())
            self.img2_size = (self.stage2.width(), self.stage2.height())

            self.img1_pil = ImageTk.getimage(self.stage1)
            self.img2_pil = ImageTk.getimage(self.stage2)

            self.img1_pil = self.img1_pil.resize((1920, 1080))
            self.img2_pil = self.img2_pil.resize((1920, 1080))

            self.img2_pil_transparent = self.img2_pil.copy()
            self.alpha = 100  # set the alpha value here (0-255)
            self.img2_pil_transparent.putalpha(self.alpha)

            self.result_pil = Image.alpha_composite(self.img1_pil, self.img2_pil_transparent)
            self.result = ImageTk.PhotoImage(self.result_pil)

            stageShow1 = Label(self, image=self.result)
            stageShow1.place(x=0, y=0)
            buttonfinish = Button(self,image=self.btnnext,bg="Black",bd=0,activebackground="Black"
                        ,command=lambda *args: self.close(stageShow1))
            buttonfinish.place(x=1820,y=550,width=70,height=70)
            
        elif self.stage == 2 :
            stageShow2 = Label(self, image=self.stage2)
            stageShow2.place(x=0, y=0)
            buttonfinish = Button(self,image=self.btnnext,bg="Black",bd=0,activebackground="Black"
                        ,command=lambda *args: self.close(stageShow2))
            buttonfinish.place(x=1820,y=550,width=70,height=70)
        elif self.stage == 3 :
            stageShow3 = Label(self, image=self.stage3)
            stageShow3.place(x=0, y=0)
            buttonfinish = Button(self,image=self.btnnext,bg="Black",bd=0,activebackground="Black"
                        ,command=lambda *args: self.close(stageShow3))
            buttonfinish.place(x=1820,y=550,width=70,height=70)


class ModeAmature(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="mode amature.png")
        self.btnback = PhotoImage(file="left-arrow.png")
        self.btnnext = PhotoImage(file="arrow-forward.png")
        self.num = [1,2,3]
        self.stage = 0
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        buttonback = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black",
                            command=lambda: controller.show_frame(StartPage))
        buttonback.place(x=20,y=20,width=100,height=100)

        buttonnext = Button(self,image=self.btnnext,bg="Black",bd=0,activebackground="Black")
        buttonnext.bind("<Button-1>", self.addnum)
        buttonnext.place(x=1820,y=550,width=70,height=70)

        
        btnnum1 = Button(self,text=self.num[0],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[0]))
        btnnum1.place(x=262,y=435,width=405,height=310)
        btnnum2 = Button(self,text=self.num[1],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[1]))
        btnnum2.place(x=764,y=435,width=405,height=310)
        btnnum3 = Button(self,text=self.num[2],bg="black",fg="#FFFFFF", bd=0,activeforeground="#E22424",activebackground="Black",font=self.Myfont(110)
                        , command=lambda *args: self.display(self.num[2]))
        btnnum3.place(x=1265,y=435,width=405,height=310)


    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont
    
    def addnum(self,event):
        for i in range(len(self.num)):
            self.num[i]+= 3
    
    # def display(self,stagenum):
    #     self.stage = stagenum
    #     print(self.stage)
    #     if self.stage == 1 :
    #         stage1 = Label(self, image=self.stage1)
    #         stage1.place(x=0, y=0)
    #     elif self.stage == 2 :
    #         stage2 = Label(self, image=self.stage2)
    #         stage2.place(x=0, y=0)
    #     elif self.stage == 3 :
    #         stage3 = Label(self, image=self.stage3)
    #         stage3.place(x=0, y=0)

class ModeCreative(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class Stage(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="howto.png")
        self.btnback = PhotoImage(file="left-arrow.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        buttonback = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black",
                            command=lambda: controller.show_frame(StartPage))
        buttonback.place(x=60,y=70,width=80,height=80)

    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class Result8Pool(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.bg = PhotoImage(file="result.png")
        self.btnback = PhotoImage(file="left-arrow.png")
        bg = Label(self, image=self.bg)
        bg.place(x=0, y=0)

        buttonback = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black",
                            command=lambda: controller.show_frame(StartPage))
        buttonback.place(x=20,y=20,width=100,height=100)
        
    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

class Play8Pool(tk.Frame):  
  
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent) 
        self.btnback = PhotoImage(file="left-arrow.png")
        self.btnnext = PhotoImage(file="arrow-forward.png")

        buttonback = Button(self,image=self.btnback,bg="Black",bd=0,activebackground="Black",
                            command=lambda: controller.show_frame(StartPage))
        buttonback.place(x=20,y=20,width=100,height=100)

        buttonnext = Button(self,image=self.btnnext,bg="Black",bd=0,activebackground="Black",
                            command=lambda: controller.show_frame(Result8Pool))
        buttonnext.place(x=1820,y=550,width=70,height=70)
        
    def Myfont(self, sizefont):
        self.myfont = Font(family="Londrina Solid", size=sizefont)
        return self.myfont

app = Sampleapp()  
app.title("Pool-Billiard")
app.mainloop()  