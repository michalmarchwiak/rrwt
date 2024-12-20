import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry boiska
field_x_min, field_x_max = 0, 68
field_y_min, field_y_max = 0, 105

# Parametry początkowe
time_steps = 200
delta_t = 0.1
ball_speed_factor = 2  # Piłka porusza się 2x szybciej niż zawodnicy

# Pozycje bramki (środek bramki)
goal_x, goal_y = 34, 0

# Pozycje początkowe zawodników i piłki
num_defenders = 4
num_attackers = 4
defenders_positions = np.array([[30, 16], [34, 16], [38, 16], [32, 16]])
attackers_positions = np.array([[28, 30], [36, 32], [40, 28], [34, 34]])
ball_position = attackers_positions[0].copy()  # Piłka startuje u pierwszego napastnika
ball_target = None  # Cel piłki przy podaniu

# Prędkości i kierunki ruchu
defenders_speeds = [3, 3, 3, 3]
attackers_speeds = [4, 4.5, 4, 4.2]
defenders_directions = [np.random.uniform(-np.pi, np.pi) for _ in range(num_defenders)]
attackers_directions = [np.arctan2(goal_y - y, goal_x - x) for x, y in attackers_positions]

# Tablice dla śledzenia pozycji w czasie
defenders_trails = [np.array([pos]) for pos in defenders_positions]
attackers_trails = [np.array([pos]) for pos in attackers_positions]
ball_trail = [ball_position.copy()]  # Inicjalizacja trajektorii piłki

# Tablica prawdopodobieństw odbioru piłki
P_i_values = []

# Flaga zakończenia animacji
animation_ended = False
intercepted_defender = None

# Symulacja czasowa
ball_holder = 0  # Indeks zawodnika ofensywnego, który trzyma piłkę
for t in range(time_steps):
    if animation_ended:
        break

    # Sprawdzenie odległości obrońców od piłki
    distances_to_ball = [
        np.linalg.norm(defenders_positions[i] - ball_position) for i in range(num_defenders)
    ]
    close_defender = np.argmin(distances_to_ball)
    if distances_to_ball[close_defender] < 0.5:  # Obrońca przejmuje piłkę
        animation_ended = True
        intercepted_defender = close_defender
        break

    # Sprawdzenie odległości obrońców od zawodnika z piłką
    distances_to_holder = [
        np.linalg.norm(defenders_positions[i] - attackers_positions[ball_holder])
        for i in range(num_defenders)
    ]
    close_defender = np.argmin(distances_to_holder)
    if distances_to_holder[close_defender] < 0.5:  # Obrońca blisko, podanie
        available_targets = [i for i in range(num_attackers) if i != ball_holder]
        ball_target = np.random.choice(available_targets)
        ball_direction = np.arctan2(
            attackers_positions[ball_target][1] - ball_position[1],
            attackers_positions[ball_target][0] - ball_position[0],
        )
        ball_speed = attackers_speeds[0] * ball_speed_factor
        ball_in_motion = True
    else:
        ball_in_motion = False

    # Aktualizacja pozycji piłki
    if ball_in_motion:
        ball_position += ball_speed * delta_t * np.array(
            [np.cos(ball_direction), np.sin(ball_direction)]
        )
        ball_trail.append(ball_position.copy())
        # Sprawdzenie, czy piłka dotarła do celu
        if (
            np.linalg.norm(ball_position - attackers_positions[ball_target]) < 0.5
        ):
            ball_holder = ball_target  # Nowy właściciel piłki
            ball_in_motion = False
    else:
        ball_trail.append(ball_position.copy())  # Piłka statyczna, dodanie pozycji

    # Aktualizacja pozycji napastników (tylko gdy piłka nie jest w ruchu)
    if not ball_in_motion:
        for i in range(num_attackers):
            attackers_positions[i, 0] += attackers_speeds[i] * np.cos(
                attackers_directions[i]
            ) * delta_t
            attackers_positions[i, 1] += attackers_speeds[i] * np.sin(
                attackers_directions[i]
            ) * delta_t
            # Ograniczenie ruchu do boiska
            attackers_positions[i, 0] = np.clip(attackers_positions[i, 0], field_x_min, field_x_max)
            attackers_positions[i, 1] = np.clip(attackers_positions[i, 1], field_y_min, field_y_max)
            attackers_trails[i] = np.vstack(
                (attackers_trails[i], attackers_positions[i])
            )

    # Aktualizacja pozycji obrońców
    for i in range(num_defenders):
        defenders_positions[i, 0] += defenders_speeds[i] * np.cos(defenders_directions[i]) * delta_t
        defenders_positions[i, 1] += defenders_speeds[i] * np.sin(defenders_directions[i]) * delta_t
        # Ograniczenie ruchu do boiska
        defenders_positions[i, 0] = np.clip(defenders_positions[i, 0], field_x_min, field_x_max)
        defenders_positions[i, 1] = np.clip(defenders_positions[i, 1], field_y_min, field_y_max)
        defenders_trails[i] = np.vstack((defenders_trails[i], defenders_positions[i]))

    # Obliczanie prawdopodobieństwa odbioru piłki jako średnia algebraiczna
    probabilities = [max(0, 1 - d / 50) for d in distances_to_ball]
    average_probability = sum(probabilities) / len(probabilities)  # Średnia algebraiczna
    P_i_values.append(average_probability)

# Przygotowanie danych do animacji
positions = {
    "defenders": defenders_trails,
    "attackers": attackers_trails,
    "ball": np.array(ball_trail),
}

# Funkcja do aktualizacji klatki animacji
def update(frame):
    if animation_ended:
        ani.event_source.stop()  # Zatrzymanie animacji

    ax1.clear()
    ax2.clear()

    # Wykres 1: Pozycje zawodników i piłki na boisku
    ax1.set_xlim(field_x_min, field_x_max)
    ax1.set_ylim(field_y_min, field_y_max)
    ax1.set_title("Symulacja ruchu zawodników i piłki (68x105)")
    ax1.set_xlabel("Pozycja X (m)")
    ax1.set_ylabel("Pozycja Y (m)")

    # Rysowanie trajektorii i pozycji zawodników
    for i in range(num_defenders):
        color = 'g' if i != intercepted_defender else 'y'  # Zmiana koloru obrońcy, który przejął piłkę
        ax1.plot(positions["defenders"][i][:frame + 1, 0], positions["defenders"][i][:frame + 1, 1], color + '-')
        ax1.plot(positions["defenders"][i][frame, 0], positions["defenders"][i][frame, 1], color + 'o', markersize=5)
    for i in range(num_attackers):
        ax1.plot(positions["attackers"][i][:frame + 1, 0], positions["attackers"][i][:frame + 1, 1], 'r-')
        ax1.plot(positions["attackers"][i][frame, 0], positions["attackers"][i][frame, 1], 'ro', markersize=5)

    # Rysowanie trajektorii i pozycji piłki
    ax1.plot(positions["ball"][:frame + 1, 0], positions["ball"][:frame + 1, 1], 'b-')
    ax1.plot(positions["ball"][frame, 0], positions["ball"][frame, 1], 'bo', markersize=5)

    # Wykres 2: Prawdopodobieństwo odbioru piłki
    ax2.set_xlim(0, time_steps * delta_t)
    ax2.set_ylim(0, 1)
    ax2.set_title("Prawdopodobieństwo odbioru piłki (średnia algebraiczna)")
    ax2.set_xlabel("Czas (s)")
    ax2.set_ylabel("Prawdopodobieństwo P_i")
    ax2.plot(np.arange(frame + 1) * delta_t, P_i_values[:frame + 1], 'b-')

# Utworzenie figur i osi
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Utworzenie animacji
ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100)

# Zapisanie animacji jako GIF
output_path = 'f1.gif'
ani.save(output_path, writer='imagemagick')
print(f"GIF zapisany jako {output_path}")