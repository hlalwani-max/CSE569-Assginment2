import math
import os

from hmm import Hmm

models_dir = os.path.join('.', 'models')  #

seq0 = ('A', 'D', 'C', 'B', 'D', 'C', 'C', 'S')
seq1 = ('B', 'D', 'S')
seq2 = ('B', 'C', 'C', 'B', 'D', 'D', 'C', 'A', 'C', 'S')
seq3 = ('A', 'C', 'D', 'S')
seq4 = ('A', 'D', 'A', 'C', 'S')
seq5 = ('D', 'B', 'B', 'S')
seq6 = ('A', 'B', 'S')
seq7 = ('D', 'D', 'B', 'D', 'D', 'B', 'A', 'C', 'C', 'D', 'A', 'B', 'B', 'C', 'D', 'B', 'B', 'B', 'S')
seq8 = ('D', 'B', 'D', 'S')
seq9 = ('A', 'A', 'A', 'A', 'D', 'C', 'B', 'S')

observations = [seq0, seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9]
# observations = [('A','C','D','D','C','C','S'), ('A','D','S')]

if __name__ == '__main__':
    model_file1 = "model1.json"
    hmm1 = Hmm(os.path.join(model_file1))
    model_file2 = "model2.json"
    hmm2 = Hmm(os.path.join(model_file2))

    hmm1.observation_generator()
'''
    print("FORWARD ALGORITHM:")
    for obs in observations:
        p1 = hmm1.forward(obs)
        p2 = hmm2.forward(obs)
        #

        print("Observations = ", obs,
              " Fwd Prob (Machine 1) = ", p1,
              ", Fwd Prob (Machine 2) = ", p2,
              ", Fwd Prob log (Machine 1) = ", (math.log(p1) if p1 != 0 else "NA"),
              ", Fwd Prob log (Machine 2) = ", (math.log(p2) if p2 != 0 else "NA"),
              ", Best sequence coming from - ", "Machine 1" if p1 > p2 else "Machine 2")
        # print("Viterbi (Machine 1): ","Observations = ", obs, "Prob = ", prob,"Log Probability = ", (math.log(p1) if prob!=0 else "NA"), " Hidden State Sequence = ", hidden_states)

    print("\nViterbi (Machine 2): ")
    for obs in observations:
        prob, hidden_states = hmm2.viterbi(obs)
        print("Observations = ", obs,
              "Prob = ", prob,
              ", Log Probability = ", (math.log(prob) if prob != 0 else "NA"),
              ", Hidden State Sequence = ", hidden_states if prob != 0 else "NA")
'''