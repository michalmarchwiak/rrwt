#chyba lepszy pomysl
import numpy as np


class OffensivePlayer:
    def __init__(self, x, y, has_ball=False):
        """
        Inicjalizacja zawodnika ofensywnego.
        """
        self.x = x
        self.y = y
        self.has_ball = has_ball  # Czy zawodnik posiada piłkę
        self.speed = 0.95  # Prędkość zawodnika

    def move(self, target_x, target_y, delta_t, field_x_min, field_x_max, field_y_min, field_y_max):
        """
        Poruszaj się w kierunku celu z ograniczeniem do boiska.
        """
        direction = np.arctan2(target_y - self.y, target_x - self.x)
        self.x += self.speed * np.cos(direction) * delta_t
        self.y += self.speed * np.sin(direction) * delta_t

        # Ograniczenie ruchu do wymiarów boiska
        self.x = np.clip(self.x, field_x_min, field_x_max)
        self.y = np.clip(self.y, field_y_min, field_y_max)

    def closest_opponent_distance(self, opponents):
        """
        Znajdź najbliższego przeciwnika i zwróć odległość.
        """
        distances = [np.linalg.norm((self.x - op.x, self.y - op.y)) for op in opponents]
        return min(distances)

    def pass_ball(self, ball, teammates):
        """
        Przeprowadź podanie do najbliższego kolegi z drużyny.
        """
        if self.has_ball:
            closest_teammate = min(teammates, key=lambda tm: np.linalg.norm((self.x - tm.x, self.y - tm.y)))
            self.has_ball = False
            ball.initiate_pass(self, closest_teammate)


class DefensivePlayer:
    def __init__(self, x, y):
        """
        Inicjalizacja zawodnika defensywnego.
        """
        self.x = x
        self.y = y
        self.speed = 1.05  # Prędkość zawodnika

    def move_towards(self, target_x, target_y, delta_t, field_x_min, field_x_max, field_y_min, field_y_max):
        """
        Poruszaj się w kierunku wskazanej pozycji z ograniczeniem do boiska.
        """
        direction = np.arctan2(target_y - self.y, target_x - self.x)
        self.x += self.speed * np.cos(direction) * delta_t
        self.y += self.speed * np.sin(direction) * delta_t

        # Ograniczenie ruchu do wymiarów boiska
        self.x = np.clip(self.x, field_x_min, field_x_max)
        self.y = np.clip(self.y, field_y_min, field_y_max)

    def can_intercept_ball(self, ball):
        """
        Sprawdź, czy zawodnik defensywny może przejąć piłkę.
        """
        if not ball.owner and ball.is_moving:
            distance_to_ball = np.linalg.norm((self.x - ball.x, self.y - ball.y))
            return distance_to_ball < 1.0
        return False

    def can_tackle(self, opponent):
        """
        Sprawdź, czy zawodnik defensywny może odebrać piłkę przeciwnikowi.
        """
        distance_to_opponent = np.linalg.norm((self.x - opponent.x, self.y - opponent.y))
        return distance_to_opponent < 1.0


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
        Zainicjuj podanie.
        """
        self.owner = None
        self.is_moving = True
        self.target = receiver

    def update_position(self, delta_t, field_x_min, field_x_max, field_y_min, field_y_max):
        """
        Zaktualizuj pozycję piłki w czasie podania z ograniczeniem do boiska.
        """
        if self.is_moving and self.target:
            direction = np.arctan2(self.target.y - self.y, self.target.x - self.x)
            self.x += self.speed * np.cos(direction) * delta_t
            self.y += self.speed * np.sin(direction) * delta_t

            # Ograniczenie ruchu do wymiarów boiska
            self.x = np.clip(self.x, field_x_min, field_x_max)
            self.y = np.clip(self.y, field_y_min, field_y_max)

            # Sprawdź, czy piłka dotarła do celu
            if np.linalg.norm((self.x - self.target.x, self.y - self.target.y)) < 0.1:
                self.is_moving = False
                self.owner = self.target
                self.target = None