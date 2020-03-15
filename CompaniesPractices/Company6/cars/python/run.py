import numpy as np

from track import Track
from environment import Car, Environment
from sys import argv

def predict(state):
    return np.random.choice([0, 1, 2, 3])

def main():
    track = Track.read_from(argv[1])
    env = Environment(track)

    broken = False
    done = False

    while not broken and not done:
        action = predict(env.state())
        broken, done, progress = env.step(action)

    print(progress)

if __name__ == '__main__':
    main()
