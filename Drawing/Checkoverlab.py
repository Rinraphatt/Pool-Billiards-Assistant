import math

def checkCollision(x1, y1, x2, y2, x, y, radius):
    
    # Calculate the coefficients a, b, and c of the line equation.
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    
    # Calculate the distance between the center of the circle and the line.
    dist = abs(a*x + b*y + c) / math.sqrt(a*a + b*b)
    
    # Check the relationship between the line and the circle.
    if dist == radius:
        print("Touch")
    elif dist < radius:
        print("Intersect")
    else:
        print("Outside")

# Example usage
x1, y1 = 100, 100
x2, y2 = 400, 400
x, y = 255, 255
radius = 100

checkCollision(x1, y1, x2, y2, x, y, radius)