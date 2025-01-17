import matplotlib

from set1withclasses import OffensivePlayer, DefensivePlayer, Ball
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Wymiary boiska
field_x_min, field_x_max = float(0), float(68)
field_y_min, field_y_max = float(0), float(105)


# Funkcja rysująca boisko piłki nożnej
def draw_pitch():
    # Granice boiska
    plt.plot([field_x_min, field_x_max], [0, 0], 'k-')  # Dolna linia
    plt.plot([field_x_min, field_x_max], [field_y_max, field_y_max], 'k-')  # Górna linia
    plt.plot([field_x_min, field_x_min], [0, field_y_max], 'k-')  # Lewa linia
    plt.plot([field_x_max, field_x_max], [0, field_y_max], 'k-')  # Prawa linia
    plt.plot([field_x_min, field_x_max], [field_y_max / 2, field_y_max / 2], 'k--')  # Linia środkowa

    # Pole karne
    plt.plot([13.84, 13.84], [0, 16.5], 'k-')  # Lewa linia pola karnego
    plt.plot([field_x_max - 13.84, field_x_max - 13.84], [0, 16.5], 'k-')  # Prawa linia pola karnego
    plt.plot([13.84, field_x_max - 13.84], [16.5, 16.5], 'k-')  # Górna linia pola karnego
    plt.plot([13.84, 13.84], [field_y_max, field_y_max - 16.5], 'k-')  # Lewa linia pola karnego (góra)
    plt.plot([field_x_max - 13.84, field_x_max - 13.84], [field_y_max, field_y_max - 16.5], 'k-')  # Prawa linia pola karnego (góra)
    plt.plot([13.84, field_x_max - 13.84], [field_y_max - 16.5, field_y_max - 16.5], 'k-')  # Dolna linia pola karnego (góra)

    # Punkt karny
    plt.scatter([34], [11], color='k', s=30)  # Punkt karny (dół)
    plt.scatter([34], [field_y_max - 11], color='k', s=30)  # Punkt karny (góra)

    # Koło środkowe
    circle = plt.Circle((34, field_y_max / 2), 9.15, color='k', fill=False)
    plt.gca().add_artist(circle)  # Koło środkowe
    plt.scatter([34], [field_y_max / 2], color='k', s=30)  # Punkt środkowy

d1 = DefensivePlayer(18, 15, 18, 18)
d2 = DefensivePlayer(23, 15, 28, 18)
d3 = DefensivePlayer(41, 15, 38, 18)
d4 = DefensivePlayer(48, 15, 48, 18)
d5 = DefensivePlayer(25, 30, 25, 30)
d6 = DefensivePlayer(35, 30, 35, 30)
d7 = DefensivePlayer(45, 30, 45, 30)

o1 = OffensivePlayer("o1", 20, 40, 14, 40, False)
o2 = OffensivePlayer("o2", 34, 45, 30, 45, True)
o3 = OffensivePlayer("o3", 48, 40, 52, 40,False)
o4 = OffensivePlayer("o4", 25, 55, 20, 55, False)
o5 = OffensivePlayer("o5", 34, 60, 32, 60, False)
o6 = OffensivePlayer("o6", 43, 55, 46, 55, False)

k_goal = 1.0
k_opp = 8.0
k_team = 1.0


ball = Ball(x=34, y=45, owner=o2)

# Zawodnicy defensywni i ofensywni
defenders = [d1, d2, d3, d4, d5, d6, d7]
offensives = [o1, o2, o3, o4, o5, o6]

# Parametry symulacji
delta_t = 0.3  # Krok czasowy
steps = 2000  # Liczba kroków w symulacji

# Funkcja rysująca boisko i zawodników
def plot_state():
    plt.gca().cla()  # Czyści tylko aktualną oś zamiast całej figury
    draw_pitch()

    # Rysowanie zawodników defensywnych
    for i, defender in enumerate(defenders):
        plt.scatter(defender.x, defender.y, color="blue", label="Defensywni" if i == 0 else "", s=100)
        plt.text(defender.x, defender.y + 2, f"D{i+1}", color="blue", ha="center")

    # Rysowanie zawodników ofensywnych
    for i, offensive in enumerate(offensives):
        plt.scatter(offensive.x, offensive.y, color="orange", label="Ofensywni" if i == 0 else "", s=100)
        plt.text(offensive.x, offensive.y + 2, f"O{i+1}", color="orange", ha="center")

    # Rysowanie piłki
    plt.scatter(ball.x, ball.y, color="black", label="Piłka", s=50)

    plt.xlim(field_x_min - 5, field_x_max + 5)
    plt.ylim(-5, field_y_max + 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.pause(0.01)


def find_closest_to_ball(defenders, ball):
    """
    Znajdź zawodnika defensywnego, który jest najbliżej piłki.
    Zwraca obiekt zawodnika oraz odległość.
    """
    # Oblicz odległości do piłki dla wszystkich obrońców
    distances = [(defender, np.linalg.norm((defender.x - ball.x, defender.y - ball.y))) for defender in defenders]

    # Znajdź obrońcę z minimalną odległością
    closest_defender, min_distance = min(distances, key=lambda x: x[1])
    return closest_defender


# Funkcja symulująca jeden krok z poprawionym ruchem zawodnika posiadającego piłkę
def simulate_step():
    """
    Symulacja jednego kroku gry.
    """
    # Aktualizacja piłki w ruchu
    if ball.is_moving:
        ball.update_position(delta_t)
    else:
        ball.move()

    # Ruch zawodników ofensywnych
    for offensive in offensives:
        if offensive.has_ball:
            offensive.move(delta_t, defenders)  # Ruch z piłką
            closest_defender_distance = offensive.closest_defender_distance(defenders)
            if closest_defender_distance < 1.5:  # Obrońca blisko, podanie
                offensive.pass_ball(ball, offensives, defenders)
                print(f"Zawodnik {offensive} podał piłkę!")
        else:
            offensive.move(delta_t, defenders)

    # Próba przejęcia piłki przez obrońców
    for defender in defenders:
        # Próba przechwycenia podania
        if ball.is_moving and defender.intercept_pass(ball):
            ball.is_moving = False
            ball.owner = defender
            defender.has_ball = True
            for offensive in offensives:
                offensive.has_ball = False  # Reset flag ofensywnych
            print(f"Zawodnik defensywny {defender} przeciął podanie!")
            return False  # Zakończenie symulacji, piłka przejęta

    # Próba odebrania piłki przez najbliższego obrońcę
    if ball.owner and isinstance(ball.owner, OffensivePlayer):
        closest_defender = find_closest_to_ball(defenders, ball)
        success = closest_defender.tackle(ball.owner)
        if success:
            ball.owner = closest_defender  # Piłka przejęta przez obrońcę
            for offensive in offensives:
                offensive.has_ball = False  # Reset flag ofensywnych
            closest_defender.has_ball = True
            print(f"Zawodnik defensywny {closest_defender} przejął piłkę!")
            return False  # Zakończenie symulacji, piłka przejęta

    # Ruch obrońców w kierunku piłki lub idealnych pozycji
    for defender in defenders:
        if ball.owner is None:  # Piłka bez właściciela
            defender.move_towards(ball.x, ball.y, delta_t)
        else:  # Obrońcy pilnują pozycji
            defender.move(offensives, defenders, k_goal, k_opp, k_team, delta_t)

    if ball.y < 16 and ball.owner:
        print("Ofensywa wygrała")
        return False
    return True  # Kontynuacja symulacji
# Symulacja
plt.figure(figsize=(10, 15))
plt.ion()
for step in range(steps):
    if not simulate_step():
        break  # Koniec symulacji, piłka przejęta przez obrońcę
    plot_state()

plt.ioff()
plt.show()