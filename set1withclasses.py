import numpy as np
import random



class OffensivePlayer:
    def __init__(self, name, x:int, y, ideal_x, ideal_y, has_ball=False)->None:
        """
        Inicjalizacja zawodnika ofensywnego.
        """
        self.name = name
        self.initial_x = x  # Zapisanie początkowej pozycji
        self.initial_y = y
        self.x = self.initial_x
        self.y = self.initial_y
        self.ideal_x = ideal_x  # Idealna pozycja w formacji (X)
        self.ideal_y = ideal_y  # Idealna pozycja w formacji (Y)
        self.has_ball = has_ball
        self.speed = 1.0  # Prędkość zawodnika


    def reset_position(self):
        """
        Resetuje pozycję zawodnika defensywnego do pozycji początkowej.
        """
        self.x = self.initial_x
        self.y = self.initial_y
        self.has_ball = False

    def __str__(self):
        return self.name

    def move(self, delta_t, defenders):
        """
        Poruszanie zawodnika ofensywnego.
        - Jeśli zawodnik jest na pozycji spalonej, cofa się.
        - Jeśli zawodnik nie ma piłki, wraca do swojej idealnej pozycji.
        """
        if self.has_ball:
            # Znalezienie najbliższego obrońcy bliżej bramki niż zawodnik
            closest_defender_y = min(defender.y for defender in defenders)

            # Sprawdzenie, czy zawodnik jest na pozycji spalonej
            if self.y < closest_defender_y:
                # Cofanie się w kierunku piłki (lub środka boiska)
                target_y = min(closest_defender_y - 2, 105 / 2)  # Ustal cel w bezpiecznej odległości od linii spalonego
                direction = np.arctan2(target_y - self.y, 0)  # Ruch w pionie
                self.y += self.speed * np.sin(direction) * delta_t
            else:
                # Ruch w stronę bramki przeciwnika
                self.y -= self.speed * delta_t
        else:
            # Jeśli zawodnik nie ma piłki, wraca do swojej idealnej pozycji
            direction = np.arctan2(self.ideal_y - self.y, self.ideal_x - self.x)
            self.x += self.speed * np.cos(direction) * delta_t * 0.5 # Wolniejszy ruch do idealnej pozycji
            self.y += (-delta_t * 0.75) + (self.speed * np.sin(direction) * delta_t * 0.5)


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
        best_teammate = self.find_best_teammate(teammates, defenders)
        if best_teammate:
            self.has_ball = False
            ball.owner = None  # Piłka przestaje mieć właściciela podczas ruchu
            ball.target = best_teammate  # Ustawienie celu podania
            ball.is_moving = True  # Piłka zaczyna się poruszać

class DefensivePlayer:
    def __init__(self, x, y, ideal_x, ideal_y):
        """
        Inicjalizacja zawodnika defensywnego.
        """
        self.initial_x = x  # Zapisanie początkowej pozycji
        self.initial_y = y
        self.x = self.initial_x
        self.y = self.initial_y
        self.ideal_x = ideal_x  # Idealna pozycja w formacji
        self.ideal_y = ideal_y
        self.speed = 1.1  # Prędkość zawodnika
        self.has_ball = False

    def reset_position(self):
        """
        Resetuje pozycję zawodnika defensywnego do pozycji początkowej.
        """
        self.x = self.initial_x
        self.y = self.initial_y
        self.has_ball = False

    def closest_offensive_distance(self, offs):
        """
        Znajdź najbliższego obrońcę i zwróć odległość.
        """
        distances = [np.linalg.norm((self.x - o.x, self.y - o.y)) for o in offs]
        return min(distances)

    def move_towards(self, target_x, target_y, delta_t):
        """
        Poruszaj się w kierunku celu.
        """
        direction = np.arctan2(target_y - self.y, target_x - self.x)
        self.x += self.speed * np.cos(direction) * delta_t
        self.y += self.speed * np.sin(direction) * delta_t

    def calculate_total_force(self, offs, teammates, k_goal, k_opp, k_team, epsilon=5e-1):
        """
        Oblicza całkowitą siłę działającą na zawodnika, uwzględniając wpływ wszystkich przeciwników.

        Parametry:
            offs (list of OffensivePlayer): Lista zawodników ofensywnych (przeciwników).
            teammates (list of DefensivePlayer): Lista kolegów z drużyny.
            k_goal (float): Współczynnik siły dążenia do celu.
            k_opp (float): Współczynnik siły przyciągania do przeciwników.
            k_team (float): Współczynnik siły odpychania między zawodnikami.
            epsilon (float): Mała wartość zapobiegająca dzieleniu przez zero (domyślnie 1e-3).

        Zwraca:
            tuple: Wektor całkowitej siły działającej na zawodnika (F_x, F_y).
        """
        # Aktualna pozycja zawodnika
        r_x, r_y = self.x, self.y

        # Idealna pozycja zawodnika
        r_goal_x, r_goal_y = self.ideal_x, self.ideal_y

        # Siła dążenia do idealnej pozycji (celu)
        f_goal_x = -k_goal * (r_x - r_goal_x)
        f_goal_y = -k_goal * (r_y - r_goal_y)

        # Siła przyciągania do wszystkich napastników
        f_opp_x, f_opp_y = 0.0, 0.0
        for opponent in offs:
            r_opp_x, r_opp_y = opponent.x, opponent.y
            r_diff_opp_x = r_opp_x - r_x
            r_diff_opp_y = r_opp_y - r_y
            distance_opp = max((r_diff_opp_x ** 2 + r_diff_opp_y ** 2) ** 0.5, epsilon)

            # Siła malejąca z kwadratem odległości (lub inną funkcją, np. liniową)
            f_opp_x += (k_opp * r_diff_opp_x) / (distance_opp ** 2 + epsilon)
            f_opp_y += (k_opp * r_diff_opp_y) / (distance_opp ** 2 + epsilon)

        # Siła odpychania od kolegów z drużyny
        f_team_x, f_team_y = 0.0, 0.0
        for teammate in teammates:
            if teammate == self:  # Pomijamy samego siebie
                continue
            r_j_x, r_j_y = teammate.x, teammate.y
            r_diff_team_x = r_j_x - r_x
            r_diff_team_y = r_j_y - r_y
            distance_team = max((r_diff_team_x ** 2 + r_diff_team_y ** 2) ** 0.5, epsilon)

            # Siła odpychania malejąca z kwadratem odległości
            f_team_x += (-k_team * r_diff_team_x) / (distance_team ** 2 + epsilon)
            f_team_y += (-k_team * r_diff_team_y) / (distance_team ** 2 + epsilon)

        # Łączna siła
        f_total_x = f_goal_x + f_opp_x + f_team_x
        f_total_y = f_goal_y + f_opp_y + f_team_y

        return f_total_x, f_total_y


    def move(self, offs, teammates, k_goal, k_opp, k_team, delta_t):
        """
        Poruszaj się pod wpływem:
        - Przyciągania do idealnej pozycji.
        - Przyciągania do piłki (silniejsze, jeśli bliżej piłki).
        """
        att_x, att_y = self.calculate_total_force(offs, teammates, k_goal, k_opp, k_team)

        self.x += att_x * self.speed * delta_t
        self.y += att_y * self.speed * delta_t

    def intercept_pass(self, ball):
        """
        Sprawdź, czy możesz przeciąć podanie.
        """
        if ball.is_moving and not ball.owner:
            distance_to_ball = np.linalg.norm((self.x - ball.x, self.y - ball.y))
            if random.random() < 0.5 and distance_to_ball < 0.75:
                return True
        return False

    def tackle(self, offensive_player):
        """
        Sprawdź, czy możesz odebrać piłkę przeciwnikowi.
        Szansa na odbiór piłki jest odwrotnie proporcjonalna do odległości od obrońcy.
        Przy odległości 0 wynosi 0.5, a przy odległości większej niż 1.5 wynosi 0.
        """
        # Obliczenie odległości do zawodnika ofensywnego
        distance_to_player = np.linalg.norm((self.x - offensive_player.x, self.y - offensive_player.y))

        # Jeśli przeciwnik ma piłkę
        if offensive_player.has_ball:
            # Wyznaczenie szansy na odbiór piłki
            if distance_to_player <= 1.5:
                tackle_chance = max(0.5 * (1.5 - distance_to_player) / 1.5, 0)
                # Rzut kostką na odbiór piłki
                if random.random() < tackle_chance:
                    offensive_player.has_ball = False
                    return True  # Piłka została przejęta
        return False  # Piłka nie została przejęta



class Ball:
    def __init__(self, x, y, owner=None):
        """
        Inicjalizacja piłki.
        """
        self.x = x
        self.y = y
        self.initial_x = self.x
        self.initial_y = self.y
        self.owner = owner  # Zawodnik posiadający piłkę (lub None)
        self.is_moving = False
        self.speed = 4.0
        self.target = None


    def reset(self):
        """
        Resetuje pozycję zawodnika defensywnego do pozycji początkowej.
        """
        self.x = self.initial_x
        self.y = self.initial_y
        self.owner = None
        self.is_moving = False

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
            if np.linalg.norm((self.x - self.target.x, self.y - self.target.y)) < 1:
                self.is_moving = False
                self.owner = self.target
                self.owner.has_ball = True
                self.target = None

    def move(self):
        if self.owner:
            self.x = self.owner.x
            self.y = self.owner.y