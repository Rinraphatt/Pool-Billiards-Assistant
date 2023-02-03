from tkinter import *
from tkinter import ttk
import tkinter.messagebox 
import tkinter.colorchooser 
import tkinter.filedialog 

# Initiat Window
root = Tk()
root.title("Calculator")

# Functions
def addTextInput(input):
    global content
    content = content+str(input)
    textInput.set(content)

def clearTextInput():
    global content
    content = "0"
    textInput.set(content)

def calculate():
    global content
    content = str(eval(content))
    textInput.set(content)

# Create widgets
content = ""
textInput = StringVar(value="0")
display = Entry(font=("arial", 30, "bold"), fg="white", bg="green", textvariable=textInput, justify="right")
display.grid(columnspan=4)

pixelVirtual = tkinter.PhotoImage(width=1, height=1)
buttonWidth = 75
buttonHeight = 75

botton7 = Button(font=("arial", 30, "bold"), fg="black", text="7", command=lambda:addTextInput(7), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=1, column=0)
botton8 = Button(font=("arial", 30, "bold"), fg="black", text="8", command=lambda:addTextInput(8), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=1, column=1)
botton9 = Button(font=("arial", 30, "bold"), fg="black", text="9", command=lambda:addTextInput(9), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=1, column=2)
bottonc = Button(font=("arial", 30, "bold"), fg="black", text="c", command=lambda:clearTextInput(), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=1, column=3)

botton4 = Button(font=("arial", 30, "bold"), fg="black", text="4", command=lambda:addTextInput(4), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=2, column=0)
botton5 = Button(font=("arial", 30, "bold"), fg="black", text="5", command=lambda:addTextInput(5), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=2, column=1)
botton6 = Button(font=("arial", 30, "bold"), fg="black", text="6", command=lambda:addTextInput(6), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=2, column=2)
bottonDivide = Button(font=("arial", 30, "bold"), fg="black", text="/", command=lambda:addTextInput("/"), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=2, column=3)

botton1 = Button(font=("arial", 30, "bold"), fg="black", text="1", command=lambda:addTextInput(1), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=3, column=0)
botton2 = Button(font=("arial", 30, "bold"), fg="black", text="2", command=lambda:addTextInput(2), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=3, column=1)
botton3 = Button(font=("arial", 30, "bold"), fg="black", text="3", command=lambda:addTextInput(3), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=3, column=2)
bottonMultiply = Button(font=("arial", 30, "bold"), fg="black", text="x", command=lambda:addTextInput("*"), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=3, column=3)

bottonFrontBracket = Button(font=("arial", 30, "bold"), fg="black", text="(", command=lambda:addTextInput("("), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=4, column=0)
botton0 = Button(font=("arial", 30, "bold"), fg="black", text="0", command=lambda:addTextInput(0), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=4, column=1)
bottonBackBracket = Button(font=("arial", 30, "bold"), fg="black", text=")", command=lambda:addTextInput(")"), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=4, column=2)
bottonMinus = Button(font=("arial", 30, "bold"), fg="black", text="-", command=lambda:addTextInput("-"), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=4, column=3)

bottonEqual = Button(font=("arial", 30, "bold"), fg="black", text="=", command=lambda:calculate(), image=pixelVirtual, width=str(buttonWidth*2+8)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=5, column=0, columnspan=2)
bottonDot = Button(font=("arial", 30, "bold"), fg="black", text=".", command=lambda:addTextInput("."), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=5, column=2)
bottonPlus = Button(font=("arial", 30, "bold"), fg="black", text="+", command=lambda:addTextInput("+"), image=pixelVirtual, width=str(buttonWidth)+"px", height=str(buttonHeight)+"px", compound="c").grid(row=5, column=3)

# Display Window
root.mainloop()