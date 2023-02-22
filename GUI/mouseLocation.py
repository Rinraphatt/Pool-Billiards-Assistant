from tkinter import *

class Hole:
    def __init__(self, x, y) -> None:
        xPos = x
        yPos = y

        self.coordinates = [xPos, yPos]
        self.width = 50
        self.height = 50

        hole = canvas.create_oval(xPos-(self.width/2), yPos-(self.height/2), xPos+(self.width/2), yPos+(self.height/2), fill=HOLE_DEACTIVE_COLOR, tag="Hole")
        holesVisual.append(hole)

class WhiteBall:
    def __init__(self, x, y) -> None:
        xPos = x
        yPos = y

        self.coordinates = [xPos, yPos]
        self.width = 50
        self.height = 50

        ball = canvas.create_oval(xPos-(self.width/2), yPos-(self.height/2), xPos+(self.width/2), yPos+(self.height/2), fill=WHITEBALL_COLOR, outline=WHITEBALL_BORDER_COLOR, width=3, tag="whiteBall")
        entitiesVisual.append(ball)

        entityText = canvas.create_text(xPos, yPos-(self.height/2+20), text="White Ball", font=('consolas', 13), fill=WHITEBALL_BORDER_COLOR, tags="whiteBallText")
        entitiesTextVisual.append(entityText)

class GreenBall:
    def __init__(self, x, y) -> None:
        xPos = x
        yPos = y

        self.coordinates = [xPos, yPos]
        self.width = 50
        self.height = 50

        ball = canvas.create_oval(xPos-(self.width/2), yPos-(self.height/2), xPos+(self.width/2), yPos+(self.height/2), fill=TARGETBALL_COLOR, outline=TARGETBALL_BORDER_COLOR, width=3, tag="targetBall")
        entitiesVisual.append(ball)

        entityText = canvas.create_text(xPos, yPos-(self.height/2+20), text="Target Ball", font=('consolas', 13), fill=TARGETBALL_BORDER_COLOR, tags="targetBallText")
        entitiesTextVisual.append(entityText)

class RedBall:
    def __init__(self, x, y) -> None:
        xPos = x
        yPos = y

        self.coordinates = [xPos, yPos]
        self.width = 50
        self.height = 50

        ball = canvas.create_oval(xPos-(self.width/2), yPos-(self.height/2), xPos+(self.width/2), yPos+(self.height/2), fill=OBSTACLEBALL_COLOR, outline=OBSTACLEBALL_BORDER_COLOR, width=3, tag="obstacleBall")
        entitiesVisual.append(ball)

        entityText = canvas.create_text(xPos, yPos-(self.height/2+20), text="Obstacle Ball", font=('consolas', 13), fill=OBSTACLEBALL_BORDER_COLOR, tags="ObstacleBallText")
        entitiesTextVisual.append(entityText)

class TargetZone:
    def __init__(self, x, y) -> None:
        xPos = x
        yPos = y

        self.coordinates = [xPos, yPos]
        self.width = 200
        self.height = 200

        targetZone = canvas.create_oval(xPos-(self.width/2), yPos-(self.height/2), xPos+(self.width/2), yPos+(self.height/2), fill=TARGETZONE_COLOR, outline=TARGETZONE_BORDER_COLOR, width=3, tag="targetZone")
        entitiesVisual.append(targetZone)

        entityText = canvas.create_text(xPos, yPos-(self.height/2+20), text="Target Zone", font=('consolas', 13), fill=TARGETZONE_BORDER_COLOR, tags="targetZoneText")
        entitiesTextVisual.append(entityText)

def mouseClick(event):
    x = event.x
    y = event.y
    
    global mode
    global entities
    global selected
    onSelect = False

    for i in range(len(entities)):
        # click on exist ball
        if x >= entities[i].coordinates[0]-(entities[i].width/2) and x <= entities[i].coordinates[0]+(entities[i].width/2) and y >= entities[i].coordinates[1]-(entities[i].height/2) and y <= entities[i].coordinates[1]+(entities[i].height/2):    
            selected = i
            onSelect = True

    # Select Exist Entity
    if onSelect == True:
        print("select " + str(selected+1))
    else:
        # Click on void
        if mode == 'create':
            if dropdownClicked.get() == 'White Ball':
                entity = WhiteBall(x, y)
                entities.append(entity)
            if dropdownClicked.get() == 'Target Ball':
                entity = GreenBall(x, y)
                entities.append(entity)
            if dropdownClicked.get() == 'Obstacle Ball':
                entity = RedBall(x, y)
                entities.append(entity)
            if dropdownClicked.get() == 'Target Zone':
                entity = TargetZone(x, y)
                entities.append(entity)
        else:
            print(f"move ball{selected} to X = {x}, Y = {y}")
            print(entities[selected].coordinates[0])

            # Update Visaul of entity
            if x < entities[selected].coordinates[0]:
                canvas.move(entitiesVisual[selected], -(entities[selected].coordinates[0]-x), 0)
                canvas.move(entitiesTextVisual[selected], -(entities[selected].coordinates[0]-x), 0)
            elif x > entities[selected].coordinates[0]:
                canvas.move(entitiesVisual[selected], x - entities[selected].coordinates[0], 0)
                canvas.move(entitiesTextVisual[selected], x - entities[selected].coordinates[0], 0)

            if y < entities[selected].coordinates[1]:
                canvas.move(entitiesVisual[selected], 0, -(entities[selected].coordinates[1]-y))
                canvas.move(entitiesTextVisual[selected], 0, -(entities[selected].coordinates[1]-y))
            elif y > entities[selected].coordinates[1]:
                canvas.move(entitiesVisual[selected], 0, y - entities[selected].coordinates[1])
                canvas.move(entitiesTextVisual[selected], 0, y - entities[selected].coordinates[1])

            # Update new coordinate of entity
            entities[selected].coordinates[0] = x
            entities[selected].coordinates[1] = y

def changeDirection(newDirection):
    if mode == 'edit':
        if newDirection == 'left':
            # Update Visaul of entity
            canvas.move(entitiesVisual[selected], -MOVE_SCALE, 0)
            canvas.move(entitiesTextVisual[selected], -MOVE_SCALE, 0)

            # Update new coordinate of entity
            entities[selected].coordinates[0] -= MOVE_SCALE
        elif newDirection == 'right':
            # Update Visaul of entity
            canvas.move(entitiesVisual[selected], MOVE_SCALE, 0)
            canvas.move(entitiesTextVisual[selected], MOVE_SCALE, 0)

            # Update new coordinate of entity
            entities[selected].coordinates[0] += MOVE_SCALE
        elif newDirection == 'up':
            # Update Visaul of entity
            canvas.move(entitiesVisual[selected], 0, -MOVE_SCALE)
            canvas.move(entitiesTextVisual[selected], 0, -MOVE_SCALE)

            # Update new coordinate of entity
            entities[selected].coordinates[1] -= MOVE_SCALE
        elif newDirection == 'down':
            # Update Visaul of entity
            canvas.move(entitiesVisual[selected], 0, MOVE_SCALE)
            canvas.move(entitiesTextVisual[selected], 0, MOVE_SCALE)

            # Update new coordinate of entity
            entities[selected].coordinates[1] += MOVE_SCALE

def toggleMode():
    global mode
    if mode == 'create': mode = 'edit'
    elif mode == 'edit': mode = 'create' 
    label.config(text="Mode:{}".format(mode))

def deleteEntity():
    global mode
    if mode == 'edit':
        del entities[selected]
        canvas.delete(entitiesVisual[selected])
        del entitiesVisual[selected]

        canvas.delete(entitiesTextVisual[selected])
        del entitiesTextVisual[selected]

def updateHoles():
    choiceHole1 = targetHole1.get()
    choiceHole2 = targetHole2.get()
    choiceHole3 = targetHole3.get()
    choiceHole4 = targetHole4.get()
    choiceHole5 = targetHole5.get()
    choiceHole6 = targetHole6.get()

    if choiceHole1 == 1: canvas.itemconfig(holesVisual[0], fill=HOLE_ACTIVE_COLOR)
    else: canvas.itemconfig(holesVisual[0], fill=HOLE_DEACTIVE_COLOR)

    if choiceHole2 == 1: canvas.itemconfig(holesVisual[1], fill=HOLE_ACTIVE_COLOR)
    else: canvas.itemconfig(holesVisual[1], fill=HOLE_DEACTIVE_COLOR)

    if choiceHole3 == 1: canvas.itemconfig(holesVisual[2], fill=HOLE_ACTIVE_COLOR)
    else: canvas.itemconfig(holesVisual[2], fill=HOLE_DEACTIVE_COLOR)

    if choiceHole4 == 1: canvas.itemconfig(holesVisual[3], fill=HOLE_ACTIVE_COLOR)
    else: canvas.itemconfig(holesVisual[3], fill=HOLE_DEACTIVE_COLOR)

    if choiceHole5 == 1: canvas.itemconfig(holesVisual[4], fill=HOLE_ACTIVE_COLOR)
    else: canvas.itemconfig(holesVisual[4], fill=HOLE_DEACTIVE_COLOR)

    if choiceHole6 == 1: canvas.itemconfig(holesVisual[5], fill=HOLE_ACTIVE_COLOR)
    else: canvas.itemconfig(holesVisual[5], fill=HOLE_DEACTIVE_COLOR)

def resetMap():
    selected = -1

    targetHole1.set(0)
    targetHole2.set(0)
    targetHole3.set(0)
    targetHole4.set(0)
    targetHole5.set(0)
    targetHole6.set(0)
    updateHoles()

    for i in reversed(range(len(entities))):
        del entities[i]
        canvas.delete(entitiesVisual[i])
        del entitiesVisual[i]

        canvas.delete(entitiesTextVisual[i])
        del entitiesTextVisual[i]

window = Tk()
window.resizable(False, False)

# Variables
CANVAS_WIDTH = 1568
CANVAS_HEIGHT = 784
MOVE_SCALE = 2
HOLE_DEACTIVE_COLOR = '#787878'
HOLE_ACTIVE_COLOR = '#1AFF00'
WHITEBALL_COLOR = '#DDDDDD'
WHITEBALL_BORDER_COLOR = '#FFFFFF'
TARGETBALL_COLOR = '#FFEC47'
TARGETBALL_BORDER_COLOR = '#D08B03'
OBSTACLEBALL_COLOR = '#FF6C6C'
OBSTACLEBALL_BORDER_COLOR = '#FF0F0F'
TARGETZONE_COLOR = '#FFE7C3'
TARGETZONE_BORDER_COLOR = '#D4A764'

selected = -1
holes = []
holesVisual = []
entities = []
entitiesVisual = []
entitiesTextVisual = []
mode = 'create'

targetHole1 = IntVar()
targetHole2 = IntVar()
targetHole3 = IntVar()
targetHole4 = IntVar()
targetHole5 = IntVar()
targetHole6 = IntVar()

options = [
    'White Ball',
    'Target Ball',
    'Obstacle Ball',
    'Target Zone'
]

dropdownClicked = StringVar()
dropdownClicked.set('White Ball')

window.update()

label = Label(window, text="Mode: {}".format(mode), font=('consolas', 40))
label.grid(row=0, column=0, sticky=W, columnspan=10)

dropdownList = OptionMenu(window, dropdownClicked, *options)
dropdownList.grid(row=1, column=0, sticky=W, columnspan=5)

Button(text="Reset", command=resetMap).grid(row=1, column=1, sticky=W, columnspan=3)

canvas = Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
canvas.grid(row=2, column=0, sticky=W, columnspan=20)

hole1 = Hole(0, 0)
hole2 = Hole(int((CANVAS_WIDTH/2)), 0)
hole3 = Hole(int((CANVAS_WIDTH)), 0)
hole4 = Hole(0, int((CANVAS_HEIGHT)))
hole5 = Hole(int((CANVAS_WIDTH/2)), int((CANVAS_HEIGHT)))
hole6 = Hole(int((CANVAS_WIDTH)), int((CANVAS_HEIGHT)))

holes.append(hole1)
holes.append(hole2)
holes.append(hole3)
holes.append(hole4)
holes.append(hole5)
holes.append(hole6)

Checkbutton(text="Hole 1", variable=targetHole1).grid(row=3, column=0, sticky=W)
Checkbutton(text="Hole 2", variable=targetHole2).grid(row=3, column=1, sticky=W)
Checkbutton(text="Hole 3", variable=targetHole3).grid(row=3, column=2, sticky=W)
Checkbutton(text="Hole 4", variable=targetHole4).grid(row=3, column=3, sticky=W)
Checkbutton(text="Hole 5", variable=targetHole5).grid(row=3, column=4, sticky=W)
Checkbutton(text="Hole 6", variable=targetHole6).grid(row=3, column=5, sticky=W)

Button(text="Update Holes", command=updateHoles).grid(row=4, column=0, sticky=W, columnspan=3)

canvas.bind('<Button-1>', mouseClick)
# canvas.bind('<B1-Motion>', ball.move)
window.bind('<Left>', lambda event: changeDirection('left'))
window.bind('<Right>', lambda event: changeDirection('right'))
window.bind('<Up>', lambda event: changeDirection('up'))
window.bind('<Down>', lambda event: changeDirection('down'))

window.bind('t', lambda event: toggleMode())
window.bind('d', lambda event: deleteEntity())
window.bind('r', lambda event: resetMap())

window.mainloop()