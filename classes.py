import numpy as np
import random

class OffensivePlayer:
    def __init__(self, name, x:int, y, ideal_x, ideal_y, has_ball=False)->None:
        self.name = name
        self.initial_x = x
        self.initial_y = y
        self.x = self.initial_x
        self.y = self.initial_y
        self.ideal_x = ideal_x
        self.ideal_y = ideal_y
        self.has_ball = has_ball
        self.speed = 1.0

    def reset_position(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.has_ball = False

    def __str__(self):
        return self.name

    def move(self, delta_t, defenders):
        if self.has_ball:
            closest_defender_y = min(defender.y for defender in defenders)
            if self.y < closest_defender_y:
                target_y = min(closest_defender_y - 2, 105 / 2)
                direction = np.arctan2(target_y - self.y, 0)
                self.y += self.speed * np.sin(direction) * delta_t
            else:
                self.y -= self.speed * delta_t
        else:
            direction = np.arctan2(self.ideal_y - self.y, self.ideal_x - self.x)
            self.x += self.speed * np.cos(direction) * delta_t * 0.5
            self.y += (-delta_t * 0.75) + (self.speed * np.sin(direction) * delta_t * 0.5)

    def closest_defender_distance(self, defenders):
        distances = [np.linalg.norm((self.x - d.x, self.y - d.y)) for d in defenders]
        return min(distances)

    def find_best_teammate(self, teammates, defenders):
        best_teammate = None
        max_distance = -np.inf
        for teammate in teammates:
            if teammate != self:
                distance_to_closest_defender = teammate.closest_defender_distance(defenders)
                if distance_to_closest_defender > max_distance:
                    max_distance = distance_to_closest_defender
                    best_teammate = teammate
        return best_teammate

    def pass_ball(self, ball, teammates, defenders):
        best_teammate = self.find_best_teammate(teammates, defenders)
        if best_teammate:
            self.has_ball = False
            ball.owner = None
            ball.target = best_teammate
            ball.is_moving = True

class DefensivePlayer:
    def __init__(self, x, y, ideal_x, ideal_y):
        self.initial_x = x
        self.initial_y = y
        self.x = self.initial_x
        self.y = self.initial_y
        self.ideal_x = ideal_x
        self.ideal_y = ideal_y
        self.speed = 1.1
        self.has_ball = False

    def reset_position(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.has_ball = False

    def closest_offensive_distance(self, offs):
        distances = [np.linalg.norm((self.x - o.x, self.y - o.y)) for o in offs]
        return min(distances)

    def move_towards(self, target_x, target_y, delta_t):
        direction = np.arctan2(target_y - self.y, target_x - self.x)
        self.x += self.speed * np.cos(direction) * delta_t
        self.y += self.speed * np.sin(direction) * delta_t

    def calculate_total_force(self, offs, teammates, k_goal, k_opp, k_team, epsilon=5e-1):
        r_x, r_y = self.x, self.y
        r_goal_x, r_goal_y = self.ideal_x, self.ideal_y
        f_goal_x = -k_goal * (r_x - r_goal_x)
        f_goal_y = -k_goal * (r_y - r_goal_y)
        f_opp_x, f_opp_y = 0.0, 0.0
        for opponent in offs:
            r_opp_x, r_opp_y = opponent.x, opponent.y
            r_diff_opp_x = r_opp_x - r_x
            r_diff_opp_y = r_opp_y - r_y
            distance_opp = max((r_diff_opp_x ** 2 + r_diff_opp_y ** 2) ** 0.5, epsilon)
            f_opp_x += (k_opp * r_diff_opp_x) / (distance_opp ** 2 + epsilon)
            f_opp_y += (k_opp * r_diff_opp_y) / (distance_opp ** 2 + epsilon)
        f_team_x, f_team_y = 0.0, 0.0
        for teammate in teammates:
            if teammate == self:
                continue
            r_j_x, r_j_y = teammate.x, teammate.y
            r_diff_team_x = r_j_x - r_x
            r_diff_team_y = r_j_y - r_y
            distance_team = max((r_diff_team_x ** 2 + r_diff_team_y ** 2) ** 0.5, epsilon)
            f_team_x += (-k_team * r_diff_team_x) / (distance_team ** 2 + epsilon)
            f_team_y += (-k_team * r_diff_team_y) / (distance_team ** 2 + epsilon)
        f_total_x = f_goal_x + f_opp_x + f_team_x
        f_total_y = f_goal_y + f_opp_y + f_team_y
        return f_total_x, f_total_y

    def move(self, offs, teammates, k_goal, k_opp, k_team, delta_t):
        att_x, att_y = self.calculate_total_force(offs, teammates, k_goal, k_opp, k_team)
        self.x += att_x * self.speed * delta_t
        self.y += att_y * self.speed * delta_t

    def intercept_pass(self, ball):
        if ball.is_moving and not ball.owner:
            distance_to_ball = np.linalg.norm((self.x - ball.x, self.y - ball.y))
            if random.random() < 0.5 and distance_to_ball < 0.75:
                return True
        return False

    def tackle(self, offensive_player):
        distance_to_player = np.linalg.norm((self.x - offensive_player.x, self.y - offensive_player.y))
        if offensive_player.has_ball:
            if distance_to_player <= 1.5:
                tackle_chance = max(0.5 * (1.5 - distance_to_player) / 1.5, 0)
                if random.random() < tackle_chance:
                    offensive_player.has_ball = False
                    return True
        return False

class Ball:
    def __init__(self, x, y, owner=None):
        self.x = x
        self.y = y
        self.initial_x = self.x
        self.initial_y = self.y
        self.owner = owner
        self.is_moving = False
        self.speed = 4.0
        self.target = None

    def reset(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.owner = None
        self.is_moving = False

    def closest_defender_distance(self, defenders):
        distances = [np.linalg.norm((self.x - d.x, self.y - d.y)) for d in defenders]
        return min(distances)

    def update_position(self, delta_t):
        if self.is_moving and self.target:
            direction = np.arctan2(self.target.y - self.y, self.target.x - self.x)
            self.x += self.speed * np.cos(direction) * delta_t
            self.y += self.speed * np.sin(direction) * delta_t
            if np.linalg.norm((self.x - self.target.x, self.y - self.target.y)) < 1:
                self.is_moving = False
                self.owner = self.target
                self.owner.has_ball = True
                self.target = None

    def move(self):
        if self.owner:
            self.x = self.owner.x
            self.y = self.owner.y