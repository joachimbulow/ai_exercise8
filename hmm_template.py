import numpy as np

"""
Hidden Markov Model using Viterbi algorithm to find most
likely sequence of hidden states.

The problem is to find out the most likely sequence of states
of the weather (hot, cold) from a describtion of the number
of ice cream eaten by a boy in the summer.
"""


def main():
    np.set_printoptions(suppress=True)

    states = np.array(["initial", "hot", "cold", "final"])

    # To simulate starting from index 1, we add a dummy value at index 0
    observationss = [
        [None, 3, 1, 3]
        #[None, 3, 3, 1, 1, 2, 2, 3, 1, 3],
        #[None, 3, 3, 1, 1, 2, 3, 3, 1, 2],
    ]

    # Markov transition matrix
    # transitions[start, end]
    transitions = np.array([[.0, .8, .2, .0],  # Initial state
                            [.0, .6, .3, .1],  # Hot state
                            [.0, .4, .5, .1],  # Cold state
                            [.0, .0, .0, .0],  # Final state
                            ])

    # P(v|q), sandsynlighed for antal is givet temperatur
    # emission[state, observation]
    emissions = np.array([[.0, .0, .0, .0],  # Initial state
                          [.0, .2, .4, .4],  # Hot state
                          [.0, .5, .4, .1],  # Cold state
                          [.0, .0, .0, .0],  # Final state
                          ])

    for observations in observationss:
        print("Observations: {}".format(' '.join(map(str, observations[1:]))))

        probability = compute_forward(states, observations, transitions, emissions)
        print("Probability: {}".format(probability))

        path = compute_viterbi(states, observations, transitions, emissions)
        print("Path: {}".format(' '.join(path)))

        print('')


def inclusive_range(a, b):
    return range(a, b + 1)


def compute_forward(states, observations, transitions, emissions):
    forward = np.zeros((states.size, len(observations)))
    for s in range(1, states.size):
        # transitions, sandsynligheden for at gå fra init til de forskellige states
        # emission, sandsynligheden for at vi får den første observation givet det nuværende state
        forward[s][1] = transitions[0][s] * emissions[s][observations[1]]

    for t in range(2, len(observations)):
        for s in range(1, states.size):
            # Pointen her er, at man for hvert state udregner hvert forrige states sandsynlighed, gange sandsynlighedn
            # for at gå det forrige state til det vi er i, ganget med sandsynligheden for at der x antal is i det givne state
            # Til sidst summerer vi alle de forrige der var.
            # Derfor laver vi en sum variabel og looper gennem og lægger til denne
            sum = 0
            for ss in range(1, states.size):
                sum += forward[ss][t-1] * transitions[ss][s] * emissions[s][observations[t]]
            forward[s][t] = sum

    finalsum = 0
    for s in range(1, states.size):
        finalsum += forward[s][len(observations) - 1] * transitions[s][states.size - 1]
    forward[states.size - 1][len(observations) - 1] = finalsum
    return forward[states.size - 1][len(observations) - 1]


def compute_viterbi(states, observations, transitions, emissions):
    # Viterbi udregner modsat forward den højeste af sin forgænger ganget med sands. for transition
    # ganget med sands for den mængde is givet temperaturen.
    viterbi = np.zeros((states.size, len(observations)))
    # Backpointer holder for hvert state en reference til det index, hvor største forgænger lå.
    # (Inklusiv transition, så states samme dag har ikke nødvendigvis samme pointer)
    backpointer = np.zeros((states.size, len(observations)))
    # Init første dag basically:)
    for s in range(1, len(states)):
        viterbi[s][1] = transitions[0][s] * emissions[s][observations[1]]
        backpointer[s][1] = 0
    # Så fylder vi resten
    for t in range(2, len(observations)):
        for s in range(1, states.size - 1):
            probabilities_list = []
            backpointer_list = states.size * [0]
            # For hver dag t, looper vi hvert state s igennem for hver forgænger ss.
            # Ganger forgængerens sands med transition til s, og ganger med sands for den mængde is givet statet.
            for ss in range(1, states.size):
                probabilities_list.append(viterbi[ss][t-1] * transitions[ss][s] * emissions[s][observations[1]])
                # Backpointeren beregnes ved at gange forgængerens sandsynlighed med transition til s
                backpointer_list[ss] = viterbi[ss][t-1] * transitions[ss][s]
            # vi gemmer den maximale værdi beregnet i for loopet ovenover i vores viterbi matrix
            viterbi[s][t] = max(probabilities_list)
            # vi gemmer i
            backpointer[s][t] = argmax(backpointer_list)
    final_list = []
    final_backpointer_list = states.size * [0]
    for s in range(1, states.size):
        final_list.append(viterbi[s, len(observations) - 1] * transitions[s][states.size-1])
        final_backpointer_list[s] = viterbi[s, len(observations) - 1] * transitions[s][states.size-1]
    viterbi[states.size - 1][len(observations) - 1] = max(final_list)
    backpointer[states.size-1][len(observations) - 1] = argmax(final_backpointer_list)

    # returner bedste path fra slut til start: [hot, hot, hot] ex.
    return []


def argmax(sequence):
    # Note: You could use np.argmax(sequence), but only if sequence is a list.
    # If it is a generator, first convert it: np.argmax(list(sequence))
    return max(enumerate(sequence), key=lambda x: x[1])[0]


if __name__ == '__main__':
    main()
