import math

def q1():
    s = [0, 1, 2, 3, 4]
    r = [1, 0, 1, 0, 1]
    v = [0, .1, .2, .3, .4]
    discount = .9
    decay = .5
    T = 5
    # T = horizon
    # T = len(s)
    ret = 0
    t = 0
    for n in range(1, T-t):
        # calculate G_t:t+n
        gttn = 0
        #intermediate rewards
        for idx in range(t, n):
            gttn += pow(discount, idx) * r[idx]
        # value func
        gttn += pow(discount, n) * v[n]

        ret += (1-decay) * pow(decay, n-1) * gttn

    gtth = 0
    for idx in range(t, T):
        gtth += pow(discount, idx) * r[idx] 
    # gtth += pow(discount, T) * v[T-1] # something to note: for the "post termination reward", you don't use the value of the next state (there is no next state!). 
    ret += pow(decay, T-t-1) * gtth
    print(ret)
    # print(n)

# q1()
# print(.5*(1 +(.5 * 1.081) + (.25 * 1.9558) + (.125 * 2.00683) + (.0625*2.702296)))

def q2():
    td_error = [1.1, .4, .8, -.5, .7]
    discount = .9
    decay = .5
    ans = 0
    for i in range(len(td_error)):
        ans += pow(discount*decay, i) * td_error[i]
    print(ans)

# q2()

def q3():
    v = [0, 10, 15, 20, 25, 30]
    w = [1, 1, 1, 1, 1, 1]
    z = [0, 0, 0, 0, 0, 0]
    r = [1, 0, 0, 0, 0, 0]
    discount = .9
    alpha = .1
    decay = .3
    for s in range(len(v)-1, 0, -1): #stop before goal (terminal state)
        feature = [0 for i in range(len(v))]
        feature[s] = 1
        z = [i*discount*decay + feature[idx] for idx, i in enumerate(z)]
        td_error = r[s-1] + discount*v[s-1]*w[s-1] - v[s]*w[s]
        v = [i + alpha*td_error*z[idx] for idx, i in enumerate(v)]

    print([v[i]*w[i] for i in range(len(v))])
    print(w)
    print(z)
# q3()

def q4():
    



