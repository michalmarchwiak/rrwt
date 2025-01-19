import matplotlib
from classes import OffensivePlayer, DefensivePlayer, Ball
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

field_x_min, field_x_max = float(0), float(68)
field_y_min, field_y_max = float(0), float(105)

def draw_pitch():
    plt.plot([field_x_min, field_x_max], [0, 0], 'k-')
    plt.plot([field_x_min, field_x_max], [field_y_max, field_y_max], 'k-')
    plt.plot([field_x_min, field_x_min], [0, field_y_max], 'k-')
    plt.plot([field_x_max, field_x_max], [0, field_y_max], 'k-')
    plt.plot([field_x_min, field_x_max], [field_y_max / 2, field_y_max / 2], 'k--')
    plt.plot([13.84, 13.84], [0, 16.5], 'k-')
    plt.plot([field_x_max - 13.84, field_x_max - 13.84], [0, 16.5], 'k-')
    plt.plot([13.84, field_x_max - 13.84], [16.5, 16.5], 'k-')
    plt.plot([13.84, 13.84], [field_y_max, field_y_max - 16.5], 'k-')
    plt.plot([field_x_max - 13.84, field_x_max - 13.84], [field_y_max, field_y_max - 16.5], 'k-')
    plt.plot([13.84, field_x_max - 13.84], [field_y_max - 16.5, field_y_max - 16.5], 'k-')
    plt.scatter([34], [11], color='k', s=30)
    plt.scatter([34], [field_y_max - 11], color='k', s=30)
    circle = plt.Circle((34, field_y_max / 2), 9.15, color='k', fill=False)
    plt.gca().add_artist(circle)
    plt.scatter([34], [field_y_max / 2], color='k', s=30)

d1 = DefensivePlayer(18, 15, 18, 18)
d2 = DefensivePlayer(23, 15, 28, 18)
d3 = DefensivePlayer(41, 15, 38, 18)
d4 = DefensivePlayer(48, 15, 48, 18)
d5 = DefensivePlayer(25, 30, 25, 30)
d6 = DefensivePlayer(35, 30, 35, 30)
d7 = DefensivePlayer(45, 30, 45, 30)

o1 = OffensivePlayer("o1", 20, 40, 14, 40, False)
o2 = OffensivePlayer("o2", 34, 45, 30, 45, True)
o3 = OffensivePlayer("o3", 48, 40, 52, 40, False)
o4 = OffensivePlayer("o4", 25, 55, 20, 55, False)
o5 = OffensivePlayer("o5", 34, 60, 32, 60, False)
o6 = OffensivePlayer("o6", 43, 55, 46, 55, False)

k_goal = 1.0
k_opp = 5.0
k_team = 1.0

ball = Ball(x=34, y=45, owner=o2)

defenders = [d1, d2, d3, d4, d5, d6, d7]
offensives = [o1, o2, o3, o4, o5, o6]

delta_t = 0.3
steps = 2000

def plot_state():
    plt.gca().cla()
    draw_pitch()
    for i, defender in enumerate(defenders):
        plt.scatter(defender.x, defender.y, color="blue", label="Defensywni" if i == 0 else "", s=100)
        plt.text(defender.x, defender.y + 2, f"D{i+1}", color="blue", ha="center")
    for i, offensive in enumerate(offensives):
        plt.scatter(offensive.x, offensive.y, color="orange", label="Ofensywni" if i == 0 else "", s=100)
        plt.text(offensive.x, offensive.y + 2, f"O{i+1}", color="orange", ha="center")
    plt.scatter(ball.x, ball.y, color="black", label="Piłka", s=50)
    plt.xlim(field_x_min - 5, field_x_max + 5)
    plt.ylim(-5, field_y_max + 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.pause(0.01)

def find_closest_to_ball(defenders, ball):
    distances = [(defender, np.linalg.norm((defender.x - ball.x, defender.y - ball.y))) for defender in defenders]
    closest_defender, min_distance = min(distances, key=lambda x: x[1])
    return closest_defender

def simulate_step():
    if ball.is_moving:
        ball.update_position(delta_t)
    else:
        ball.move()
    for offensive in offensives:
        if offensive.has_ball:
            offensive.move(delta_t, defenders)
            closest_defender_distance = offensive.closest_defender_distance(defenders)
            if closest_defender_distance < 1.5:
                offensive.pass_ball(ball, offensives, defenders)
        else:
            offensive.move(delta_t, defenders)
    for defender in defenders:
        if ball.is_moving and defender.intercept_pass(ball):
            ball.is_moving = False
            ball.owner = defender
            defender.has_ball = True
            for offensive in offensives:
                offensive.has_ball = False
            return False
    if ball.owner and isinstance(ball.owner, OffensivePlayer):
        closest_defender = find_closest_to_ball(defenders, ball)
        success = closest_defender.tackle(ball.owner)
        if success:
            ball.owner = closest_defender
            for offensive in offensives:
                offensive.has_ball = False
            closest_defender.has_ball = True
            return False
    for defender in defenders:
        if ball.owner is None:
            defender.move_towards(ball.x, ball.y, delta_t)
        else:
            defender.move(offensives, defenders, k_goal, k_opp, k_team, delta_t)
    if ball.y < 16 and ball.owner and 13 < ball.x < 52:
        return False
    return True

def animation():
    plt.figure(figsize=(10, 15))
    plt.ion()
    for step in range(steps):
        if not simulate_step():
            break
        plot_state()
    plt.ioff()