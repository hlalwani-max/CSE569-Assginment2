import json
import sys

import numpy as np


class Hmm(object):
    def __init__(self, model_name):
        if model_name is None:
            print("Error!! Model json not found!!")
            sys.exit()

        self.model = json.loads(open(model_name).read())["hmm"]
        self.A = self.model["A"]
        self.states = self.A.keys()
        self.N = len(self.states)  # number of states
        self.B = self.model["B"]
        self.symbols = ['S', 'A', 'B', 'C', 'D']
        self.M = len(self.symbols)  # number of symbols
        self.pi = self.model["pi"]
        return

    def forward(self, obs):
        self.fwd = [{}]
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        for t in range(1, len(obs)):
            self.fwd.append({})
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob

    def viterbi(self, obs):
        vit = [{}]
        path = {}
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]

        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}
            for y in self.states:
                (prob, state) = max((vit[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        n = 0
        if len(obs) != 1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def observation_generator(self):
        observations = []
        stateSEQ = []
        pi = [self.pi[item] for item in self.pi]
        states = list(self.states)
        firstState = np.random.choice(states, 1, p=pi)
        # B = self.B[(str(firstState))]
        print(str(firstState))
        # print(self.B)
        A = []

        for state in self.A:
            # print(state)
            for symbol in state:
                # print(self.B[symbol])
                arr = (list(self.A[symbol].values()))
            A.append(arr)
        print(A)

        B = []
        for state in self.B:
            # print(state)
            for symbol in state:
                # print(self.B[symbol])
                arr = (list(self.B[symbol].values()))
            B.append(arr)

        # print(B)
        first_obs = np.random.choice(self.symbols, 1, p=B[int(firstState[0]) - 1])
        print(first_obs)
        #
        stateSEQ.append(int(firstState[0]))
        observations.append(first_obs[0])

        while observations[-1] != 'S':
            curr_state = stateSEQ[-1]
            next_state = np.random.choice(states, 1, p=A[curr_state - 1])
            stateSEQ.append(next_state[0])
            next_observation = np.random.choice(self.symbols, 1, p=B[curr_state - 1])
            observations.append(next_observation[0])
        #
        # return observations, state_seq
