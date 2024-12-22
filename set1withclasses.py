import numpy as np
import random

class OffensivePlayer:
    def __init__(self, x, y, has_ball=False):
        """
        Inicjalizacja zawodnika ofensywnego.
        """
        self.x = x
        self.y = y
        self.has_ball = has_ball
        self.speed = 1.0  # Prędkość zawodnika

    def move(self, delta_t):
        """
        Ruch ofensywnego zawodnika:
        - Bez piłki: minimalnie do przodu.
        - Z piłką: intensywnie do przodu.
        """
        if self.has_ball:
            self.y -= self.speed * delta_t  # Ruch w stronę przodu boiska
        else:
            self.y -= 0.5 * delta_t  # Minimalny ruch naprzód

    def closest_defender_distance(self, defenders):
        """
        Znajdź najbliższego obrońcę i zwróć odległość.
        """
        distances = [np.linalg.norm((self.x - d.x, self.y - d.y)) for d in defenders]
        return min(distances)

    def find_best_teammate(self, teammates, defenders):
        """
        Znajdź kolegę z drużyny, który jest najdalej od najbliższego obrońcy.
        """
        best_teammate = None
        max_distance = -np.inf
        for teammate in teammates:
            if teammate != self:  # Nie podawaj do siebie
                distance_to_closest_defender = teammate.closest_defender_distance(defenders)
                if distance_to_closest_defender > max_distance:
                    max_distance = distance_to_closest_defender
                    best_teammate = teammate
        return best_teammate

    def pass_ball(self, ball, teammates, defenders):
        """
        Podaj piłkę do najlepszego kolegi z drużyny.
        """
        if self.has_ball and self.closest_defender_distance(defenders)<1:
            best_teammate = self.find_best_teammate(teammates, defenders)
            if best_teammate:
                self.has_ball = False
                ball.initiate_pass(self, best_teammate)


class DefensivePlayer:
    def __init__(self, x, y, ideal_x, ideal_y):
        """
        Inicjalizacja zawodnika defensywnego.
        """
        self.x = x
        self.y = y
        self.ideal_x = ideal_x  # Idealna pozycja w formacji
        self.ideal_y = ideal_y
        self.speed = 1.0  # Prędkość zawodnika

    def move_towards(self, target_x, target_y, delta_t):
        """
        Poruszaj się w kierunku celu.
        """
        direction = np.arctan2(target_y - self.y, target_x - self.x)
        self.x += self.speed * np.cos(direction) * delta_t
        self.y += self.speed * np.sin(direction) * delta_t

    def move(self, ball, delta_t):
        """
        Poruszaj się pod wpływem:
        - Przyciągania do idealnej pozycji.
        - Przyciągania do piłki (silniejsze, jeśli bliżej piłki).
        """
        attraction_to_ball_x = (ball.x - self.x) / (np.linalg.norm((ball.x - self.x, ball.y - self.y)) + 1e-5)
        attraction_to_ball_y = (ball.y - self.y) / (np.linalg.norm((ball.x - self.x, ball.y - self.y)) + 1e-5)

        attraction_to_ideal_x = self.ideal_x - self.x
        attraction_to_ideal_y = self.ideal_y - self.y

        self.x += (0.7 * attraction_to_ball_x + 0.3 * attraction_to_ideal_x) * self.speed * delta_t
        self.y += (0.7 * attraction_to_ball_y + 0.3 * attraction_to_ideal_y) * self.speed * delta_t

    def intercept_pass(self, ball):
        """
        Sprawdź, czy możesz przeciąć podanie.
        """
        if ball.is_moving and not ball.owner:
            distance_to_ball = np.linalg.norm((self.x - ball.x, self.y - ball.y))
            return distance_to_ball < 1.0
        return False

    def tackle(self, offensive_player):
        """
        Sprawdź, czy możesz odebrać piłkę przeciwnikowi.
        """
        distance_to_player = np.linalg.norm((self.x - offensive_player.x, self.y - offensive_player.y))
        if distance_to_player < 1.0 and offensive_player.has_ball:
            if random.random() < 0.3:
                offensive_player.has_ball = False
            else:
                offensive_player.has_ball = True
            return True
        return False


class Ball:
    def __init__(self, x, y, owner=None):
        """
        Inicjalizacja piłki.
        """
        self.x = x
        self.y = y
        self.owner = owner  # Zawodnik posiadający piłkę (lub None)
        self.is_moving = False
        self.speed = 2.0
        self.target = None

    def initiate_pass(self, passer, receiver):
        """
        Rozpocznij podanie piłki.
        """
        self.owner = None
        self.is_moving = True
        self.target = receiver

    def update_position(self, delta_t):
        """
        Zaktualizuj pozycję piłki podczas podania.
        """
        if self.is_moving and self.target:
            direction = np.arctan2(self.target.y - self.y, self.target.x - self.x)
            self.x += self.speed * np.cos(direction) * delta_t
            self.y += self.speed * np.sin(direction) * delta_t

            # Sprawdź, czy piłka dotarła do celu
            if np.linalg.norm((self.x - self.target.x, self.y - self.target.y)) < 0.1:
                self.is_moving = False
                self.owner = self.target
                self.target = None