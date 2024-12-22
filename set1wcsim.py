from set1withclasses import OffensivePlayer, DefensivePlayer, Ball
import numpy as np
import matplotlib.pyplot as plt

# Wymiary boiska
field_x_min, field_x_max = 0, 68
field_y_min, field_y_max = 0, 105

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

# Klasy (DefensivePlayer, OffensivePlayer, Ball) są już zaimplementowane w wcześniejszych etapach.
# Tworzenie obiektów zawodników ofensywnych i defensywnych oraz piłki
d1 = DefensivePlayer(18, 15, 18, 15)
d2 = DefensivePlayer(28, 15, 28, 15)
d3 = DefensivePlayer(38, 15, 38, 15)
d4 = DefensivePlayer(48, 15, 48, 15)
d5 = DefensivePlayer(25, 30, 25, 30)
d6 = DefensivePlayer(35, 30, 35, 30)
d7 = DefensivePlayer(45, 30, 45, 30)

o1 = OffensivePlayer(20, 40, False)
o2 = OffensivePlayer(34, 45, True)
o3 = OffensivePlayer(48, 40, False)
o4 = OffensivePlayer(25, 55, False)
o5 = OffensivePlayer(34, 60, False)
o6 = OffensivePlayer(43, 55, False)

ball = Ball(x=34, y=45, owner=o2)

# Zawodnicy defensywni i ofensywni
defenders = [d1, d2, d3, d4, d5, d6, d7]
offensives = [o1, o2, o3, o4, o5, o6]

# Parametry symulacji
delta_t = 0.1  # Krok czasowy
steps = 200  # Liczba kroków w symulacji

# Funkcja rysująca boisko i zawodników
def plot_state():
    plt.clf()
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

# Funkcja symulująca jeden krok z poprawionym ruchem zawodnika posiadającego piłkę
def simulate_step():
    # Aktualizacja piłki i zawodnika, który ją posiada
    if ball.is_moving:
        ball.update_position(delta_t)
    elif ball.owner:
        ball.x, ball.y = ball.owner.x, ball.owner.y  # Piłka porusza się razem z właścicielem

    # Ruch zawodników ofensywnych
    for offensive in offensives:
        if offensive.has_ball:
            offensive.move(delta_t)  # Ruch z piłką
            ball.x, ball.y = offensive.x, offensive.y  # Aktualizacja pozycji piłki
            closest_defender_distance = offensive.closest_defender_distance(defenders)
            if closest_defender_distance < 1.0:  # Obrońca blisko, podanie
                offensive.pass_ball(ball, offensives, defenders)
        else:
            offensive.move(delta_t)  # Minimalny ruch do przodu

    # Ruch zawodników defensywnych
    for defender in defenders:
        if defender.intercept_pass(ball):
            ball.is_moving = False
            ball.owner = defender
            print(f"Zawodnik defensywny D przejął piłkę!")
            return False  # Zakończenie symulacji, piłka przejęta
        elif ball.owner is None:  # Brak właściciela piłki
            defender.move_towards(ball.x, ball.y, delta_t)
        else:  # Pilnują idealnych pozycji
            defender.move(ball, delta_t)

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