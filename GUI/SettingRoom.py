from tkinter import *
from tkinter import filedialog as fd

def mouseClick(event):
    print('mouseClick')
    # x = event.x
    # y = event.y
    
    # global mode
    # global entities
    # global selected
    # onSelect = False

    # for i in range(len(entities)):
    #     # click on exist ball
    #     if x >= entities[i].coordinates[0]-(entities[i].width/2) and x <= entities[i].coordinates[0]+(entities[i].width/2) and y >= entities[i].coordinates[1]-(entities[i].height/2) and y <= entities[i].coordinates[1]+(entities[i].height/2):    
    #         selected = i
    #         onSelect = True

    # # Select Exist Entity
    # if onSelect == True:
    #     print("select " + str(selected+1))
    # else:
    #     # Click on void
    #     if mode == 'create':
    #         if dropdownClicked.get() == 'White Ball':
    #             entity = WhiteBall(x, y)
    #             entities.append(entity)
    #         if dropdownClicked.get() == 'Target Ball':
    #             entity = GreenBall(x, y)
    #             entities.append(entity)
    #         if dropdownClicked.get() == 'Obstacle Ball':
    #             entity = RedBall(x, y)
    #             entities.append(entity)
    #         if dropdownClicked.get() == 'Target Zone':
    #             entity = TargetZone(x, y)
    #             entities.append(entity)
    #     else:
    #         print(f"move ball{selected} to X = {x}, Y = {y}")
    #         print(entities[selected].coordinates[0])

    #         # Update Visaul of entity
    #         if x < entities[selected].coordinates[0]:
    #             canvas.move(entitiesVisual[selected], -(entities[selected].coordinates[0]-x), 0)
    #             canvas.move(entitiesTextVisual[selected], -(entities[selected].coordinates[0]-x), 0)
    #         elif x > entities[selected].coordinates[0]:
    #             canvas.move(entitiesVisual[selected], x - entities[selected].coordinates[0], 0)
    #             canvas.move(entitiesTextVisual[selected], x - entities[selected].coordinates[0], 0)

    #         if y < entities[selected].coordinates[1]:
    #             canvas.move(entitiesVisual[selected], 0, -(entities[selected].coordinates[1]-y))
    #             canvas.move(entitiesTextVisual[selected], 0, -(entities[selected].coordinates[1]-y))
    #         elif y > entities[selected].coordinates[1]:
    #             canvas.move(entitiesVisual[selected], 0, y - entities[selected].coordinates[1])
    #             canvas.move(entitiesTextVisual[selected], 0, y - entities[selected].coordinates[1])

    #         # Update new coordinate of entity
    #         entities[selected].coordinates[0] = x
    #         entities[selected].coordinates[1] = y

def saveSetting():
    print('saveSetting')
    saveStrings = []
    global rgbColors, grayScaleValues

    # Save All RGB Values
    for i in range(len(rgbColors)):
        saveRGB = ""
        saveRGB += str(rgbColors[i][2]) + " "
        saveRGB += str(rgbColors[i][1]) + " "
        saveRGB += str(rgbColors[i][0]) + " "
        saveRGB += "\n"
        saveStrings.append(saveRGB)

    # Save All Gray Scale Values
    for i in range(len(grayScaleValues)):
        saveGrayScale = ""
        saveGrayScale += str(grayScaleValues[i][0]) + " "
        saveGrayScale += str(grayScaleValues[i][1]) + " "
        saveGrayScale += "\n"
        saveStrings.append(saveGrayScale)

    print(saveStrings)

    file1 = open('../Setting.txt', "w")
    file1.writelines(saveStrings)
    file1.close

def loadSetting():
    print("loadSetting")

    global rgbColors, grayScaleValues

    file1 = open('../Setting.txt', "r+")
    # print(file1.read())
    loadStrings = file1.readlines()
    file1.close

    print(loadStrings)

    # Load All RGB Values
    for i in range(len(rgbColors)):
        loadRGB = loadStrings[i].split()
            
        rgbColors[i][2] = loadRGB[2]
        rgbColors[i][1] = loadRGB[1]
        rgbColors[i][0] = loadRGB[0]

    # Load All Gray Scale Values
    for i in range(len(grayScaleValues)):
        loadGrayScaleValues = loadStrings[i+9].split()
            
        grayScaleValues[i][0] = loadGrayScaleValues[0]
        grayScaleValues[i][1] = loadGrayScaleValues[1]

    print(loadStrings)
    

def ballOptionChanged(event):
    global options
    print(dropdownClicked.get())
    print(options.index(dropdownClicked.get()))
    i = 0
    if dropdownClicked.get() == 'White Ball':
        i = 8
    else:
        i = (options.index(dropdownClicked.get())) % 8
        
    redValue.set(rgbColors[i][2])
    greenValue.set(rgbColors[i][1])
    blueValue.set(rgbColors[i][0])

    j = 0
    if (options.index(dropdownClicked.get())) >= 8:
        j = 1
    grayScaleValue.set(grayScaleValues[i][j])


def updateSetting():
    global redValue, greenValue, blueValue, grayScaleValue
    global redValueText, greenValueText, blueValueText, grayScaleValueText

    global options
    print(dropdownClicked.get())
    print(options.index(dropdownClicked.get()))
    i = 0
    if dropdownClicked.get() == 'White Ball':
        i = 8
    else:
        i = (options.index(dropdownClicked.get())) % 8
        
    rgbColors[i][2] = redValue.get()
    rgbColors[i][1] = greenValue.get()
    rgbColors[i][0] = blueValue.get()

    j = 0
    if (options.index(dropdownClicked.get())) >= 8:
        j = 1

    grayScaleValues[i][j] = grayScaleValue.get()

    print('Update')
    print(redValue.get(), greenValue.get(), blueValue.get(), grayScaleValue.get())

window = Tk()
window.resizable(False, False)

# Variables
CANVAS_WIDTH = 1500
CANVAS_HEIGHT = 750

rgbColors = [
    [17.0, 163.0, 212.0], #Yellow
    [92.0, 53.0, 37.0], #Blue
    [54.0, 83.0, 197.0], #Red
    [129.0, 72.0, 107.0], #Purple
    [50.0, 136.0, 238.0], #Orange
    [106.0, 140.0, 47.0], #Green
    [55.0, 77.0, 132.0], #Crimson
    [55.0, 55.0, 55.0], #Black
    [200.0, 200.0, 200.0] #White
]

grayScaleValues = [
    [167.0, 170.0],
    [88.0, 121.0],
    [133.0, 153.0],
    [113.0, 147.0],
    [173.0, 187.0],
    [134.0, 150.0],
    [123.0, 139.0],
    [98.0, 0.0],
    [0.0, 225.0],
]

loadSetting()

options = [
    'Ball 1', 'Ball 2', 'Ball 3', 'Ball 4',
    'Ball 5', 'Ball 6', 'Ball 7', 'Ball 8',
    'Ball 9', 'Ball 10', 'Ball 11', 'Ball 12',
    'Ball 13', 'Ball 14', 'Ball 15', 'White Ball',
]

dropdownClicked = StringVar()
dropdownClicked.set('Ball 1')

redValue = StringVar()
greenValue = StringVar()
blueValue = StringVar()
grayScaleValue = StringVar()

redValue.set(rgbColors[0][2])
greenValue.set(rgbColors[0][1])
blueValue.set(rgbColors[0][0])
grayScaleValue.set(grayScaleValues[0][0])

window.update()

canvas = Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
canvas.grid(row=0, column=0, sticky=W, rowspan=10)

label = Label(window, text='Setting', font=('consolas', 40))
label.grid(row=0, column=1, sticky=W, columnspan=3)

dropdownList = OptionMenu(window, dropdownClicked, *options, command=ballOptionChanged)
dropdownList.grid(row=1, column=1, sticky=W, columnspan=5)

redValueLabel = Label(window, text='Red Value', font=('consolas', 10))
redValueLabel.grid(row=2, column=1, sticky=W)
redValueText = Entry(window, textvariable=redValue, bg='light cyan')
redValueText.grid(row=2, column=2, sticky=W)

greenValueLabel = Label(window, text='Green Value', font=('consolas', 10))
greenValueLabel.grid(row=3, column=1, sticky=W)
greenValueText = Entry(window, textvariable=greenValue, bg='light cyan')
greenValueText.grid(row=3, column=2, sticky=W)

blueValueLabel = Label(window, text='Blue Value', font=('consolas', 10))
blueValueLabel.grid(row=4, column=1, sticky=W)
blueValueText = Entry(window, textvariable=blueValue, bg='light cyan')
blueValueText.grid(row=4, column=2, sticky=W)

grayScaleValueLabel = Label(window, text='Gray Scale Value', font=('consolas', 10))
grayScaleValueLabel.grid(row=5, column=1, sticky=W)
grayScaleValueText = Entry(window, textvariable=grayScaleValue, bg='light cyan')
grayScaleValueText.grid(row=5, column=2, sticky=W)

Button(text="OK", command=updateSetting).grid(row=6, column=1, sticky=W, columnspan=2)
Button(text="Cancle", command=loadSetting).grid(row=6, column=2, sticky=W, columnspan=2)

Button(text="Save", command=saveSetting).grid(row=7, column=1, sticky=W, columnspan=2)

canvas.bind('<Button-1>', mouseClick)
# canvas.bind('<B1-Motion>', ball.move)

window.mainloop()