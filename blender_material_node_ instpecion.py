
import bpy

def inspect_principled_bsdf_inputs():
    """
    Creates a new material and prints all input sockets
    of the Principled BSDF shader node.
    """

    material_name = "TestMaterial"
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes

    principled_node = None

    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_node = node
            break

    if principled_node:
        print("Principled BSDF Node Inputs:")
        for socket in principled_node.inputs:
            print(f"- {socket.name}")
    else:
        print("Principled BSDF node was not found.")


# Run Blender inspection (only works inside Blender Python)
try:
    inspect_principled_bsdf_inputs()
except:
    print("Blender bpy module not available outside Blender environment.")




import pygame
import math
import sys

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Iridescent Metallic Cubes")

clock = pygame.time.Clock()

BACKGROUND_COLOR = (10, 10, 30)
GRID_COLOR = (40, 40, 60)

CUBE_SIZE = 80
CUBE_SPACING = 150

cube_positions = [
    (SCREEN_WIDTH // 2 - CUBE_SPACING, SCREEN_HEIGHT // 2 - CUBE_SPACING),
    (SCREEN_WIDTH // 2 + CUBE_SPACING, SCREEN_HEIGHT // 2 - CUBE_SPACING),
    (SCREEN_WIDTH // 2 - CUBE_SPACING, SCREEN_HEIGHT // 2 + CUBE_SPACING),
    (SCREEN_WIDTH // 2 + CUBE_SPACING, SCREEN_HEIGHT // 2 + CUBE_SPACING)
]

def generate_iridescent_color(angle, time, phase_offset=0):
    """
    Generates smooth RGB color transitions based on sine waves.
    Simulates metallic iridescent behavior.
    """
    red = (math.sin(angle + time * 0.5 + phase_offset) * 0.5 + 0.5) * 255
    green = (math.sin(angle + time * 0.7 + phase_offset + 2) * 0.5 + 0.5) * 255
    blue = (math.sin(angle + time * 0.9 + phase_offset + 4) * 0.5 + 0.5) * 255
    return (int(red), int(green), int(blue))

def draw_3d_cube(surface, center_position, size, rotation, time, color_phase):
    """
    Draws a rotating pseudo-3D cube with iridescent metallic lines and faces.
    """

    vertices = []

    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                rotated_x = x * math.cos(rotation[0]) - y * math.sin(rotation[0])
                rotated_y = x * math.sin(rotation[0]) + y * math.cos(rotation[0])
                rotated_z = z

                perspective = 500 / (500 - rotated_z * size)

                projected_x = center_position[0] + rotated_x * size * perspective
                projected_y = center_position[1] + rotated_y * size * perspective

                vertices.append((projected_x, projected_y, rotated_z))

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        start_point = vertices[edge[0]]
        end_point = vertices[edge[1]]

        angle = math.atan2(end_point[1] - start_point[1],
                           end_point[0] - start_point[0])

        color = generate_iridescent_color(angle, time, color_phase)

        pygame.draw.line(surface, color,
                         (start_point[0], start_point[1]),
                         (end_point[0], end_point[1]), 3)

    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5]
    ]

    for face in faces:
        face_points = [(vertices[i][0], vertices[i][1]) for i in face]

        center_x = sum(p[0] for p in face_points) / len(face_points)
        center_y = sum(p[1] for p in face_points) / len(face_points)

        angle = math.atan2(center_y - center_position[1],
                           center_x - center_position[0])

        color = generate_iridescent_color(angle, time, color_phase)

        face_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

        pygame.draw.polygon(face_surface, (*color, 50), face_points)

        highlight_color = (255, 255, 255, 80)
        pygame.draw.polygon(face_surface, highlight_color, face_points, 2)

        surface.blit(face_surface, (0, 0))

def draw_background_grid(surface):
    for x in range(0, SCREEN_WIDTH, 20):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1)
    for y in range(0, SCREEN_HEIGHT, 20):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1)

rotation_speeds = [
    (0.01, 0.012, 0.008),
    (0.012, 0.008, 0.01),
    (0.008, 0.01, 0.012),
    (0.009, 0.011, 0.013)
]

rotations = [(0, 0, 0) for _ in range(4)]

time_value = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    screen.fill(BACKGROUND_COLOR)

    draw_background_grid(screen)

    time_value += 0.05

    for i in range(4):
        rx, ry, rz = rotations[i]
        sx, sy, sz = rotation_speeds[i]

        rotations[i] = (rx + sx, ry + sy, rz + sz)

        draw_3d_cube(
            screen,
            cube_positions[i],
            CUBE_SIZE,
            rotations[i],
            time_value,
            i * math.pi / 2
        )

    title_font = pygame.font.SysFont("Arial", 36)
    title_text = title_font.render("Iridescent Metallic Cubes", True, (200, 200, 220))
    screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 20))

    info_font = pygame.font.SysFont("Arial", 18)
    instructions = [
        "Four rotating cubes with metallic iridescent colors",
        "Dynamic light simulation using sine-based RGB gradients",
        "Press ESC to exit"
    ]

    for i, line in enumerate(instructions):
        rendered_line = info_font.render(line, True, (150, 150, 170))
        screen.blit(rendered_line, (SCREEN_WIDTH // 2 - rendered_line.get_width() // 2, 80 + i * 25))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
