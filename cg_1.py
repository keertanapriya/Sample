# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:21:44 2024

@author: keertanapriya
"""

# import turtle
# import random
# n=int(input("number of iterations of the C curve (give less than 10 ):-"))
# #try to give the iterations as even numbers to see symmetry
# # powers of 2 are more prefereable
# step=float(input("step length for forward propagation :- "))
# #adjust this step value to increase the size of the curve
# #note as you increase the number of intersections , decrease the forward step value
# """
# L system used :- 
# Variables:	F
# Constants:	+ -
# Start:	F
# Rules:	F → +F--F+
# where "F" means "draw forward", "+" means "turn clockwise 45°", and "-" means "turn anticlockwise 45°". 
    
# """
# def generate_string(n):
#     if n == 0:
#         return "F"
#     else:
#         prev_string = generate_string(n-1)
#         new_string = ""
#         for char in prev_string:
#             if char == "F":
#                 new_string += "+F--F+"
#             else:
#                 new_string += char
#         return new_string
# draw_string=generate_string(n)
# turtle.penup()
# turtle.goto(0, 200)#starting pos of turtle to draw the curve
# turtle.pendown()
# turtle.speed(10000)#speed of drawing can be still increased
# for i in draw_string:
#     if i=='F':
#         color = (random.random(), random.random(), random.random())  # Random RGB color
#         turtle.color(color)
#         turtle.forward(step)
#     elif i=='+':
#         turtle.right(45)
#     elif i=='-':
#         turtle.left(45)
# turtle.done()







# import turtle
# import random
# n=int(input("number of iterations of the dragon curve (give less than 10 ):-"))
# #try to give the iterations as even numbers to see symmetry
# # powers of 2 are more prefereable
# step=float(input("step length for forward propagation :- "))
# #adjust this step value to increase the size of the curve
# #note as you increase the number of intersections , decrease the forward step value
# """
# L system used :- 
# variables : F G
# constants : + -
# start  : F
# rules  : (F → F-G), (G → F+G)
# angle  : 90°
# Here, F and G both mean "draw forward", + means "turn right by angle", and - means "turn left by angle". 
# """
# def generate_string(n):
#     if n == 0:
#         return "F"
#     else:
#         prev_string = generate_string(n-1)
#         new_string = ""
#         for char in prev_string:
#             if char == "F":
#                 new_string += "F-G"
#             elif char == "G":
#                 new_string += "F+G"
#             else:
#                 new_string += char
#         return new_string
# draw_string=generate_string(n)
# print(len(draw_string))
# turtle.penup()
# turtle.goto(0, 0)#starting pos of turtle to draw the curve
# turtle.pendown()
# turtle.speed(10000000000000)#speed of drawing can be still increased
# for i in draw_string:
#     if i=='F' or i=='G':
#         color = (random.random(), random.random(), random.random())  # Random RGB color
#         turtle.color(color)
#         turtle.forward(step)
#     elif i=='+':
#         turtle.left(90)# - sign to denote anticlockwise direction
#     elif i=='-':
#         turtle.right(90)
# turtle.done()
    
# import turtle
# import math

# def mandelbrot(z , c , n=20):
#     if abs(z) > 10 ** 12:
#         return float("nan")
#     elif n > 0:
#         return mandelbrot(z ** 2 + c, c, n - 1)
#     else:
#         return z ** 2 + c

# # screen size (in pixels)
# screenx, screeny = 800, 600
# # complex plane limits
# complexPlaneX, complexPlaneY = (-2.0, 2.0), (-2.0, 2.0)
# # discretization step
# step = 2

# # turtle config
# turtle.tracer(0, 0)
# screen = turtle.Screen()
# screen.title("Mandelbrot Fractal (discretization step = %d)" % (int(step)))

# mTurtle = turtle.Turtle()
# mTurtle.penup()

# # px * pixelToX = x in complex plane coordinates
# pixelToX, pixelToY = (complexPlaneX[1] - complexPlaneX[0])/screenx, (complexPlaneY[1] - complexPlaneY[0])/screeny
# # plot
# for px in range(-int(screenx/2), int(screenx/2), int(step)):
#     for py in range(-int(screeny/2), int(screeny/2), int(step)):
#         x, y = px * pixelToX, py * pixelToY
#         m =  mandelbrot(0, x + 1j * y)
#         if not math.isnan(m.real):
#             r = int(abs(math.sin(m.imag)) * 255)
#             g = int(abs(math.sin(m.imag + (2 * math.pi / 3))) * 255)
#             b = int(abs(math.sin(m.imag + (4 * math.pi / 3))) * 255)
#             color = (r, g, b)
#             hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
#             mTurtle.color(hex_color)
#             mTurtle.dot(2.4, hex_color)
#             mTurtle.goto(px, py)
#     turtle.update()

# turtle.mainloop()

# import turtle
# import random
# n=int(input("number of iterations of the sier-triangle (give less than 10 ):-"))
# #try to give the iterations as even numbers to see symmetry
# # powers of 2 are more prefereable
# step=float(input("step length for forward propagation :- "))
# #adjust this step value to increase the size of the curve
# #note as you increase the number of intersections , decrease the forward step value

# """
# L system used :- 
# variables : A B
# constants : + -
# start  : A
# rules  : (A → B-A-B), (B → A+B+A)
# angle  : 60°
# Here, A and B both mean "draw forward", + means "turn left by angle", and - means "turn right by angle" 
# """
# def generate_string(n):
#     if n == 0:
#         return "A"
#     else:
#         prev_string = generate_string(n-1)
#         new_string = ""
#         for char in prev_string:
#             if char == "A":
#                 new_string += "B-A-B"
#             elif char == "B":
#                 new_string += "A+B+A"
#             else:
#                 new_string += char
#         return new_string
# draw_string=generate_string(n)
# print(len(draw_string))
# turtle.penup()
# turtle.goto(-100, -100)#starting pos of turtle to draw the curve
# turtle.pendown()
# turtle.speed(1000000)#speed of drawing can be still increased
# for i in draw_string:
#     if i=='A' or i=='B':
#         color = (random.random(), random.random(), random.random())  # Random RGB color
#         turtle.color(color)
#         turtle.forward(step)
#     elif i=='+':
#         turtle.left(60)
#     elif i=='-':
#         turtle.right(60)
# turtle.done()

# import turtle
# import random
# n=int(input("number of iterations of the terrain (give less than 10 ):-"))
# #try to give the iterations as even numbers to see symmetry
# # powers of 2 are more prefereable
# step=float(input("step length for forward propagation :- "))
# #adjust this step value to increase the size of the curve
# #note as you increase the number of intersections , decrease the forward step value

# """
# L system used :- 
# variables : F
# constants : + *
# start  : F
# rules  : (F -> F+F*F+F)
# angle  : 60°,120°
# Here, F means "draw forward", + means "turn left by angle" , and * means turn right by 120 deg
# """
# def generate_string(n):
#     if n == 0:
#         return "F"
#     else:
#         prev_string = generate_string(n-1)
#         new_string = ""
#         for char in prev_string:
#             if char == "F":
#                 new_string += "F+F*F+F"
#             else:
#                 new_string += char
#         return new_string
# draw_string=generate_string(n)
# print(len(draw_string))
# turtle.penup()
# turtle.goto(-150, -150)#starting pos of turtle to draw the curve
# turtle.pendown()
# turtle.speed(1000000)#speed of drawing can be still increased
# for i in draw_string:
#     if i=='F':
#         color = (random.random(), random.random(), random.random())  # Random RGB color
#         turtle.color(color)
#         turtle.forward(step)
#     elif i=='+':
#         turtle.left(60)
#     elif i=='*':
#         turtle.right(120)
# turtle.done()


    

    

