import numpy as np
class ExampleEnv:
    TRANS = {
        (1, 1): 1, (1, 2): 4, (1, 3): 1, (1, 4): 2,
        (2, 1): 2, (2, 2): 5, (2, 3): 1, (2, 4): 3,
        (3, 1): 3, (3, 2): 6, (3, 3): 2, (3, 4): 3,
        (4, 1): 1, (4, 2): 7, (4, 3): 4, (4, 4): 5,
        (5, 1): 2, (5, 2): 8, (5, 3): 4, (5, 4): 6,
        (6, 1): 3, (6, 2): 9, (6, 3): 5, (6, 4): 6,
        (7, 1): 4, (7, 2): 7, (7, 3): 7, (7, 4): 8,
        (8, 1): 5, (8, 2): 8, (8, 3): 7, (8, 4): 9,
        (9, 1): 6, (9, 2): 9, (9, 3): 8, (9, 4): 9
    }
    def __init__(self) -> None:
        self.now_state = 8
    def reset(self):
        # exploration start
        self.now_state = np.random.choice([1, 2, 4, 5, 6, 7, 8, 9])
        return self.now_state
    
    def step(self, a):
        magic = np.random.uniform()
        if magic < 0.1:
            next_state = self.now_state
        else:
            next_state = ExampleEnv.TRANS[(self.now_state, a)]
        if next_state == 3:
            r = 99
            d = True
        else:
            r = -1
            d = False
        self.now_state = next_state
        return self.now_state, r, d
    
def uniform_policy(s):
    return np.random.randint(1, 5)
def best_policy(s):
    p = {
        1: 4, 2: 4, 3: 4,
        4: 1, 5: 1, 6: 1,
        7: 1, 8: 1, 9: 1
    }
    return p[s]
def policy_eval(policy):
    env = ExampleEnv()
    # returns = {k: [] for k in range(1, 10)}
    returns = {(x,y): [] for x in range(1, 10) for y in range(1,5)}
    # for episode in range(10000):
    for episode in range(40000):        
        state_seq = []
        reward_seq = []
        
        s = env.reset()
        d = False
        while not d:
            a = policy(s)
            s_, r, d = env.step(a)
            # state_seq.append(s)
            state_seq.append((s,a))
            reward_seq.append(r)
            s = s_
        rev_s = list(reversed(state_seq))
        rev_r = list(reversed(reward_seq))
        cumulative_reward = 0
        for i in range(len(rev_r)):
            cumulative_reward += rev_r[i]
            returns[rev_s[i]].append(cumulative_reward)
 
 
    cnt = 0
    for k, v in returns.items():
        #to make 4 output per line
        if cnt > 3:
            print()
            cnt = 0
        cnt += 1
        if len(v) == 0:
            # print(f'v({k})={0}', end=',')
            print(f'q({k})={0}', end=',')
        else:
            # print(f'v({k})={np.mean(v):.2f}', end=',')
            print(f'q({k})={np.mean(v):.2f}', end=',')
    # print()
def main():
    policy_eval(uniform_policy)
    # policy_eval(best_policy)
if __name__ == '__main__':
    main()