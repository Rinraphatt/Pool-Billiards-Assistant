from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

rgbColors = [
    [17.0, 163.0, 212.0],
    [92.0, 53.0, 37.0],
    [43.0, 76.0, 201.0],
    [129.0, 72.0, 107.0],
    [50.0, 136.0, 238.0],
    [106.0, 140.0, 47.0],
    [55.0, 77.0, 87.0],
    [200.0, 200.0, 200.0],
    [55.0, 55.0, 55.0]
]

for i in range(len(rgbColors)):
    r1 = 59 / 255
    g1 = 69 / 255
    b1 = 134 / 255

    r2 = rgbColors[i][2] / 255
    g2 = rgbColors[i][1] / 255
    b2 = rgbColors[i][0] / 255

    # Red Color
    color1_rgb = sRGBColor(r1, g1, b1)

    # Blue Color
    color2_rgb = sRGBColor(r2, g2, b2)

    # Convert from RGB to Lab Color Space
    color1_lab = convert_color(color1_rgb, LabColor)

    # Convert from RGB to Lab Color Space
    color2_lab = convert_color(color2_rgb, LabColor)

    # Find the color difference
    delta_e = delta_e_cie2000(color1_lab, color2_lab)

    print(f"{i} : The difference between the 2 color = ", delta_e)