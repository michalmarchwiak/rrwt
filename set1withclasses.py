import numpy as np
import random




class OffensivePlayer:
    def __init__(self, name, x, y, has_ball=False):
        """
        Inicjalizacja zawodnika ofensywnego.
        """
        self.name = name
        self.x = x
        self.y = y
        self.has_ball = has_ball
        self.speed = 1.0  # Prędkość zawodnika



    def __str__(self):
        return self.name


    def move(self, delta_t, defenders):
        """
        Poruszanie zawodnika ofensywnego.
        Jeśli zawodnik jest na pozycji spalonej, cofa się.
        """
        # Znalezienie najbliższego obrońcy bliżej bramki niż zawodnik
        closest_defender_y = min(defender.y for defender in defenders)

        # Sprawdzenie, czy zawodnik jest na pozycji spalonej
        if self.y < closest_defender_y and self.has_ball == False:
            # Cofanie się w kierunku piłki (lub środka boiska)
            target_y = min(closest_defender_y - 2, 105 / 2)  # Ustal cel w bezpiecznej odległości od linii spalonego
            direction = np.arctan2(target_y - self.y, 0)  # Ruch w pionie
            self.y += self.speed * np.sin(direction) * delta_t
        else:
            # Ruch w stronę bramki przeciwnika
            self.y -= self.speed * delta_t

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
        best_teammate = self.find_best_teammate(teammates, defenders)
        self.has_ball = False
        ball.owner = None  # Piłka przestaje mieć właściciela podczas ruchu
        ball.target = best_teammate  # Ustawienie celu podania
        ball.is_moving = True  # Piłka zaczyna się poruszać
        print(f"Zawodnik {self} podaje piłkę do {best_teammate}!")


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
        # Obliczenie odległości do zawodnika ofensywnego
        distance_to_player = np.linalg.norm((self.x - offensive_player.x, self.y - offensive_player.y))

        # Jeśli odległość jest mniejsza niż 1 metr i przeciwnik ma piłkę
        if distance_to_player < 2.0 and offensive_player.has_ball:
            # 3/10 szans na odebranie piłki
            if random.random() < 0.1:
                offensive_player.has_ball = False
                print("Zawodnik defensywny przejął piłkę!")
                return True  # Piłka została przejęta
            else:
                return False  # Piłka nie została przejęta





class Ball:
    def __init__(self, x, y, owner=None):
        """
        Inicjalizacja piłki.
        """
        self.x = x
        self.y = y
        self.owner = owner  # Zawodnik posiadający piłkę (lub None)
        self.is_moving = False
        self.speed = 4.0
        self.target = None

    def closest_defender_distance(self, defenders):
        """
        Znajdź najbliższego obrońcę i zwróć odległość.
        """
        distances = [np.linalg.norm((self.x - d.x, self.y - d.y)) for d in defenders]
        return min(distances)

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

    def move(self):
        if self.owner:
            self.x = self.owner.x
            self.y = self.owner.y