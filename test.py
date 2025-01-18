from set1withclasses import OffensivePlayer, DefensivePlayer, Ball
import numpy as np

# Wymiary boiska
field_x_min, field_x_max = float(0), float(68)
field_y_min, field_y_max = float(0), float(105)


# Funkcja rysująca boisko piłki nożnej

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

delta_t = 0.5
steps = 2000




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

    if ball.y < 16 and ball.owner:
        return False
    return True


def reset_simulation():
    global defenders, offensives, ball

    for defender in defenders:
        defender.reset_position()
        defender.has_ball = False

    for offensive in offensives:
        offensive.reset_position()
        offensive.has_ball = False

    ball.reset()
    offensives[1].has_ball = True
    ball.owner = offensives[1]
    ball.x, ball.y = offensives[1].x, offensives[1].y
    ball.is_moving = False


def simulate_multiple_times(num_simulations=100, k_goal=1.0, k_opp=1.0, k_team=1.0):
    # Ustaw parametry dla obrońców
    global defenders, offensives, ball
    for defender in defenders:
        defender.k_goal = k_goal
        defender.k_opp = k_opp
        defender.k_team = k_team

    defender_wins = 0

    for simulation in range(num_simulations):
        reset_simulation()

        for step in range(steps):
            if not simulate_step():
                if ball.owner and isinstance(ball.owner, DefensivePlayer):
                    defender_wins += 1
                break

    return defender_wins / num_simulations  # Zwraca proporcję zwycięstw obrońców


num_simulations = 100
defender_win_rate = simulate_multiple_times(num_simulations)
print(f"Procent zwycięstw obrońców: {defender_win_rate:.2f}")


def gradient_ascent(num_simulations=100, alpha=0.005, max_iters=100, epsilon=1e-6, decay_rate=0.9):
    """
    Gradient ascent z zapamiętywaniem najlepszych parametrów.
    """
    # Inicjalizacja parametrów w przedziale (0.1, 10.0)
    k_goal, k_opp, k_team = 5, 5, 5

    # Inicjalizacja zmiennych do śledzenia najlepszego wyniku
    best_k_goal, best_k_opp, best_k_team = k_goal, k_opp, k_team
    best_value = 0.0  # Najlepsza wartość funkcji celu

    for iteration in range(max_iters):
        # Oblicz wartość funkcji celu
        current_value = simulate_multiple_times(num_simulations, k_goal, k_opp, k_team)

        # Sprawdź, czy bieżący wynik jest najlepszy
        if current_value > best_value:
            best_value = current_value
            best_k_goal, best_k_opp, best_k_team = k_goal, k_opp, k_team

        # Oblicz gradient numerycznie
        grad_k_goal = (simulate_multiple_times(num_simulations, k_goal + epsilon, k_opp, k_team) - current_value) / epsilon
        grad_k_opp = (simulate_multiple_times(num_simulations, k_goal, k_opp + epsilon, k_team) - current_value) / epsilon
        grad_k_team = (simulate_multiple_times(num_simulations, k_goal, k_opp, k_team + epsilon) - current_value) / epsilon

        # Normalizacja gradientów
        grad_norm = np.sqrt(grad_k_goal**2 + grad_k_opp**2 + grad_k_team**2 + 1e-8)
        grad_k_goal /= grad_norm
        grad_k_opp /= grad_norm
        grad_k_team /= grad_norm

        # Aktualizacja parametrów
        k_goal = max(0.1, min(k_goal + alpha * grad_k_goal, 10.0))
        k_opp = max(0.1, min(k_opp + alpha * grad_k_opp, 10.0))
        k_team = max(0.1, min(k_team + alpha * grad_k_team, 10.0))

        # Zmniejsz krok uczenia
        alpha *= decay_rate

        # Log postępu
        print(f"Iteracja {iteration + 1}: k_goal={k_goal:.4f}, k_opp={k_opp:.4f}, k_team={k_team:.4f}, wartość={current_value:.4f}")

        # Sprawdź kryterium zbieżności
        if grad_norm < epsilon:
            print("Gradient ascent zakończony - osiągnięto zbieżność.")
            break

    print(f"\nNajlepsza wartość funkcji celu: {best_value:.4f}")
    print(f"Odpowiadające jej parametry: k_goal={best_k_goal:.4f}, k_opp={best_k_opp:.4f}, k_team={best_k_team:.4f}")

    return best_k_goal, best_k_opp, best_k_team


optimal_k_goal, optimal_k_opp, optimal_k_team = gradient_ascent(num_simulations=100, alpha=1, max_iters=20, decay_rate=0.95)
print(f"Optymalne wartości: k_goal={optimal_k_goal:.4f}, k_opp={optimal_k_opp:.4f}, k_team={optimal_k_team:.4f}")