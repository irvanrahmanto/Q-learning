import math
import numpy as np
import pandas as pd

# definisi baca file txt
tabel = pd.read_csv("dataTugas3ML2019.txt", sep="\t", header=None).values

# fungsu numpy dimasukan kepada array
qtabel = np.zeros((tabel.shape[0], tabel.shape[1], 4))

for i in range(0, 15):
    qtabel[0, i, 0] = -math.inf
    qtabel[14, i, 1] = -math.inf
    qtabel[i, 0, 2] = -math.inf
    qtabel[i, 14, 3] = -math.inf

# fungsi menjalankan agent dari start ke goal


def walk(action, state):
    nextState = [state[0], state[1]]
    if action == 0:
        nextState[0] -= 1

    if action == 1:
        nextState[0] += 1

    if action == 2:
        nextState[1] -= 1

    if action == 3:
        nextState[1] += 1

    reward = tabel[nextState[0]][nextState[1]]

    if nextState[0] == 0 and nextState[1] == 14:
        done = True
    else:
        done = False

    return nextState, reward, done

# fungsi aksi yang berjalan


def actionstate():
    result = np.argmax(qtabel[state[0]][state[1]])
    return result


episode = 100
a = 1.0
y = 1.0

Maksimum_reward = -math.inf
goal_state = []

# tabel yang digunakan berukuran 15x15
for i in range(0, episode):
    state = [14, 0]

    goal = [state]
    rewardEpisode = tabel[state[0]][state[1]]

    while True:
        action = np.argmax(qtabel[state[0]][state[1]])

        nextState, reward, done = walk(action, state)

        nextMax = np.argmax(qtabel[nextState[0]][nextState[1]])

        hasil1 = y * qtabel[nextState[0]][nextState[1]
                                          ][nextMax] - qtabel[state[0]][state[1]][action]
        hasil2 = a * (reward + hasil1)
        hasil3 = qtabel[state[0]][state[1]][action] + hasil2

        qtabel[state[0]][state[1]][action] = hasil3

        goal.append(nextState)
        rewardEpisode += reward

        state = nextState

        if done:
            break

    if rewardEpisode > Maksimum_reward:
        Maksimum_reward = rewardEpisode
        goal_state = goal

# print("reward " + str(rewardMaksimum))
print("goal " + str(goal_state))
print("reward " + str(Maksimum_reward))
