# Liav Trabelsy - 315870345
# Yishay Schlesinger -208438119

import numpy as np
import sys
import matplotlib.pyplot as plt

import pygame
import math
import random

# Const for better understanding of the code
START_POS_X = 200
START_POS_Y = 100
ROWS = 9
FIRST_ROWS = 5
LAST_ROWS = 4
COLUMNS = 9
OFFSET_X = 40
OFFSET_ROW = 20
OFFSET_Y = 35
MID_ROW = 5
# COLORS = [pygame.Color('blue'), pygame.Color('black'), pygame.Color('green'), pygame.Color(153, 0, 76),
#           pygame.Color('orange'), pygame.Color('red'), pygame.Color('cyan'), pygame.Color('brown'),
#           pygame.Color('yellow'), pygame.Color('purple')]
COLORS = [pygame.Color("#F08080"), pygame.Color("#CD5C5C"), pygame.Color("#FF0000"), pygame.Color("#B22222"),
          pygame.Color("#800000")]


# For each index we get returns his neighbors in the nest
def find_neighbors(row, col):
    if row < MID_ROW:
        return [(row, col - 1), (row, col + 1), (row - 1, col - 1), (row - 1, col), (row + 1, col), (row + 1, col + 1)]
    if row > MID_ROW:
        return [(row, col - 1), (row, col + 1), (row - 1, col + 1), (row - 1, col), (row + 1, col), (row + 1, col - 1)]
    return [(row, col - 1), (row, col + 1), (row - 1, col - 1), (row - 1, col), (row + 1, col), (row + 1, col - 1)]


# Draw each hexagon
def draw_regular_polygon(screen, color, vertex_count, radius, position, width=0):
    n, r = vertex_count, radius
    x, y = position
    pygame.draw.polygon(screen, color, [
        (x + r * math.cos(math.pi / 2 + 2 * math.pi * i / n), y + r * math.sin(math.pi / 2 + 2 * math.pi * i / n))
        for i in range(n)
    ], width)


# Draw all nest
def draw_nest(row, col, screen, color):
    # Draw the last 4 rows
    if row > 4:
        # Adding an offset for the nest shape
        col += 1
        offset = row - FIRST_ROWS
        draw_regular_polygon(screen, color, 6, 20,
                             [START_POS_X + (col * OFFSET_X) - (row * OFFSET_ROW) + offset * OFFSET_X,
                              START_POS_Y + (row * OFFSET_Y)])
    # Draw the regular 5 first rows
    else:
        draw_regular_polygon(screen, color, 6, 20,
                             [START_POS_X + (col * OFFSET_X) - (row * OFFSET_ROW), START_POS_Y + (row * OFFSET_Y)])


# Find most correlative random vector
def find_most_corr_vect(v1, list_of_vects):
    # Find the closest vector using oklid distance
    min_dist = np.linalg.norm(v1 - list_of_vects[0][0])
    row_index = 0
    most_corr_row = 0
    most_corr_col = 0
    for vectors in list_of_vects:
        col_index = 0
        for vect in vectors:
            if vect is not None:
                dist = np.linalg.norm(v1 - vect)
                if dist < min_dist:
                    most_corr_row = row_index
                    most_corr_col = col_index
                    min_dist = dist
            col_index += 1
        row_index += 1
    # Return the indexes of the correlative one and the distance itself - to find the best sol
    return most_corr_row, most_corr_col, min_dist


# Update the first and second neighbors of the corr vect in nest
def update_neighbors(neighbors, election, vect_list, rank, update_vects):
    sec_neighbors = []
    for neighbor in neighbors:
        for row, col in neighbor:
            # Check if the row and the col are valid and that the specific vector didn't update yet
            if 0 < row < 9 and 0 < col < 9 and (row, col) not in update_vects:
                rand_vect = vect_list[row][col]
                if rand_vect is not None:
                    for i in range(len(election)):
                        dist = rand_vect[i] - election[i]
                        # For first neighbors update by 0.4 and for the second ones update by 0.1
                        rand_vect[i] -= (0.1*rank) * dist
                    update_vects.append((row, col))
                    # Create a list for all second neighbors
                    if rank == 2:
                            sec_neighbors.append(find_neighbors(row, col))
    if rank == 2:
        update_neighbors(sec_neighbors, election, vect_list, 1, update_vects)


# Update the corr vector
def update_vect(row, col, election, vect_list, update_vects):
    rand_vect = vect_list[row][col]
    for i in range(len(election)):
        dist = rand_vect[i] - election[i]
        rand_vect[i] -= 0.7 * dist
    update_vects.append((row, col))
    update_neighbors([find_neighbors(row, col)], election, vect_list, 4, update_vects)


# Print the graph
def show_graph(info):
    colors = []
    min_ind = 0
    min_val = float('inf')
    indexes = []
    for i in range(len(info)):
        if info[i] < min_val:
            min_ind = i
            min_val = info[i]
        colors.append('red')
        indexes.append(i+1)
    colors[min_ind] = 'green'
    plt.bar(indexes, info, color=colors, width=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Average Distance')
    plt.title('Finding the best solution')
    plt.show()


# Main method
file_path = input("Write the path for the input file: ")
# Parse the input file
with open(file_path) as input_file:
    elect_list = {}
    cities = []
    line = input_file.readline()
    parties = line.split(",")[1:]
    line = input_file.readline()
    index = 0
    while line:
        line_arr = line.split(",")
        cities.append(line_arr[0])
        int_arr = [int(i) for i in line_arr[1:]]
        elect_list[index] = int_arr
        line = input_file.readline()
        index += 1

to_shuffle = list(zip(elect_list.values(), cities))
random.shuffle(to_shuffle)
elect_list, cities = zip(*to_shuffle)

# Vars for best sol
best_mat = None
min_dist_mat = float('inf')
dist_list = []
# Run 10 times the program
for test in range(10):
    # Create the random vectors in the nest shape
    j = 4
    mat = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    for i in range(FIRST_ROWS):
        for k in range(COLUMNS - j):
            mat[i][k] = np.random.randint(0, 20000, (len(parties)))
        j -= 1
    j = 1
    for i in range(FIRST_ROWS, ROWS):
        for k in range(COLUMNS - j):
            mat[i][k] = np.random.randint(0, 20000, (len(parties)))
        j += 1

    # Try 30 time to find corr and update for best classification
    for iteration in range(30):
        for elect in elect_list:
            update_list = []
            corr_row, corr_col, minimum_dist = find_most_corr_vect(elect, mat)
            update_vect(corr_row, corr_col, elect, mat, update_list)

    # After updating the random vectors, classify each input vector to it's corr vect position
    ref_mat = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    j = 4
    for i in range(FIRST_ROWS):
        for k in range(COLUMNS - j):
            ref_mat[i][k] = []
        j -= 1
    j = 1
    for i in range(FIRST_ROWS, ROWS):
        for k in range(COLUMNS - j):
            ref_mat[i][k] = []
        j += 1
    avg_min = 0
    for ind in range(len(elect_list)):
        corr_row, corr_col, minimum_dist = find_most_corr_vect(elect_list[ind], mat)
        avg_min += minimum_dist
        if ref_mat[corr_row][corr_col] is not None:
            ref_mat[corr_row][corr_col].append(ind)
    # Finding the best sol yet
    if not best_mat:
        best_mat = ref_mat
        min_dist_mat = avg_min / len(elect_list)
    else:
        if min_dist_mat > avg_min / len(elect_list):
            best_mat = ref_mat
            min_dist_mat = avg_min / len(elect_list)
    # Save all avg distance for the graph
    dist_list.append(avg_min / len(elect_list))

# Print the list of cities that classified to the same place
for x in range(len(best_mat)):
    for y in range(len(best_mat[0])):
        if best_mat[x][y] is not None and len(best_mat[x][y]) > 0:
            print(f'row: {x + 1}, col: {y + 1} :')
            for ind in best_mat[x][y]:
                print(cities[ind], end=", ")
            print()
            print("-----------------------------------------------------------------------------------")

# Show the nest
pygame.init()
surface = pygame.display.set_mode((600, 470))
surface.fill('white')
color1 = pygame.Color(230, 230, 230)
# game loop
while True:
    # for loop through the event queue
    for event in pygame.event.get():
        # Check for QUIT event
        if event.type == pygame.QUIT:
            pygame.quit()
            # Show the graph
            show_graph(dist_list)
            sys.exit()
    # Draw each hexagon with the right color according to the avg of the cities Economic Cluster
    for x in range(len(best_mat)):
        for y in range(len(best_mat[0])):
            if best_mat[x][y] is not None and len(best_mat[x][y]) > 0:
                sum = 0
                for ind in best_mat[x][y]:
                    sum += elect_list[ind][0]
                avg = sum / len(best_mat[x][y])
                draw_nest(x, y, surface, COLORS[round(round(avg - 1) / 2)])
            if best_mat[x][y] is not None and len(best_mat[x][y]) == 0:
                draw_nest(x, y, surface, color1)
    pygame.display.update()

