import numpy as np

n=1000
datassets = np.zeros((4,1000))
datassets[0] = np.random.normal(1, 2.5, size=(1, n))
datassets[1] = np.random.normal(2, 2.5, size=(1, n))
datassets[2] = np.random.normal(3, 2.5, size=(1, n))
datassets[3] = np.random.normal(4, 2.5, size=(1, n))

def epsilon_greedy(arms, epsilon, plays, rewards):
    n_arms = len(arms)

    if np.random.rand() > epsilon:
        arm = np.argmax(rewards / (plays))
    else:
        arm = np.random.randint(n_arms)
    return arm

arms=[0,1,2,3]
e = 0.1
plays=[0.00001,0.00001,0.00001,0.00001]
rewards=[0,0,0,0]

for i in range(n):
    choix_arm = epsilon_greedy(arms,e,plays,rewards)
    plays[choix_arm]+=1
    rewards[choix_arm]+=datassets[i][choix_arm]

print(rewards)
