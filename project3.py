import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import eye, hstack, ones, vstack, zeros
from cvxopt import matrix, solvers


class match():

    def __init__(self, player1, player2):
        self.Game_Begin = True
        self.soccerball = np.random.randint(2)
        self.player1 = player1
        self.player2 = player2
        self.col = 4
        self.row = 2
        self.end_loc_1 = [0, 3]
        self.end_loc_2 = [4, 7]
        self.soccerball = np.random.randint(2)
        self.Soccer_ball_loc = player1.loc


    def setMatch(self):

        s_loc = [1, 2, 5, 6]
        rand_i = np.random.choice(len(s_loc), 2, replace=True)

        player1.loc = s_loc[rand_i[0]]
        player2.loc = s_loc[rand_i[1]]

        if np.random.randint(2) == 0:
            self.soccerball = player1.ball
            self.Soccer_ball_loc = player1.loc
        else:
            self.soccerball = player2.ball
            self.Soccer_ball_loc = player2.loc

    def shiftAgent(self, player, action):

        return self.agentLocLogic(action, player)


    def agentLocLogic(self, action, player):
        if action == 0 and player.loc > 3:
            return player.loc - 4

        elif action == 1 and player.loc not in self.end_loc_2:
            return player.loc + 1

        elif action == 2 and player.loc < 4:
            return player.loc + 4

        elif action == 3 and player.loc not in self.end_loc_1:
            return player.loc - 1

        else:
            return player.loc


    def actions(self, player1, player2, action1, action2):

        temp_loc1 = self.shiftAgent(player1, action1)
        temp_loc2 = self.shiftAgent(player2, action2)

        if temp_loc1 != player2.loc:

          player1.loc = temp_loc1
        else:

          self.soccerball = player2.ball

        if temp_loc2 != player1.loc:
          player2.loc = temp_loc2
        else:
          self.soccerball = player1.ball

        self.Soccer_ball_loc = player1.loc if self.soccerball else player2.loc

    def next_step(self, act1, act2):

        self.actions(self.player1, self.player2, act1, act2) \
            if np.random.randint(2) == 0 \
            else self.actions(self.player2, self.player1, act2, act1)


        if self.Soccer_ball_loc in self.end_loc_1:

            return self.state(),100,-100,1

        elif self.Soccer_ball_loc in self.end_loc_2:

            return self.state(), -100, 100, 1

        else:

            return self.state(), 0, 0, 0

    @staticmethod
    def state():
        return [player1.loc, player2.loc, game.soccerball]


class player():

    def __init__(self, name="AnyPlayer", ball=None):
        self.name = name
        self.score = 0
        self.loc = 0
        self.ball = ball


    def ball_pos(self):
        return self.ball


def plotgraph(err, iter, name="Q Learning", linewidth=1):
    plt.plot(iter, err, linewidth=linewidth)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title(name)
    plt.xlabel('Iteration')
    plt.ylabel('Q Val Difference')
    plt.show()
    plt.gcf().clear()
def friend_q(game, playerA, playerB):
    s_t = time.time()
    iterations = 10 ** 6
    epsilon = 0.2
    min_epsilon = 0.01
    decay_eps = (epsilon - min_epsilon) / iterations
    apl = 0.9
    min_alp = 0.001

    gamma = 0.9

    w = 8
    h = 8
    d = 2
    act = 5
    _Q_table = np.random.random([h, w, d, act, act])


    game.setMatch()
    done = 1

    err = []
    iter = []

    for i in range(iterations):

        while done:
            game.setMatch()
            done = 0

        loc_1 = playerA.loc
        loc_2 = playerB.loc
        ball = game.soccerball

        p_q_v = _Q_table[2, 1, 1, 2, 4]

        act_1_pl = np.random.randint(5)
        act_2_pl = np.random.randint(5)

        n_s, rewd_1, rewd_2, done = game.next_step(act1=act_1_pl, act2=act_2_pl)
        pl_1_n_act, pl_2_n_act, ball_n_game = n_s

        _Q_table[loc_1, loc_2, ball, act_1_pl, act_2_pl] = (1 - apl) * _Q_table[loc_1, loc_2, ball, act_1_pl, act_2_pl] + \
                                                    apl * ((1 - gamma) * rewd_1 + gamma * np.max(_Q_table[pl_1_n_act, pl_2_n_act, ball_n_game]))

        if [loc_1, loc_2, ball, act_1_pl, act_2_pl] == [2, 1, 1, 2, 4]:
            err.append(abs(_Q_table[2, 1, 1, 2, 4] - p_q_v))
            iter.append(i)
            print("Iteration: ", i)


        epsilon -= decay_eps
        apl *= np.e ** (-np.log(200.0) / 10 ** 6)

    print("Time:", time.time() - s_t)


    plotgraph(err, iter, name="Friend-Q ")

def qlearning(game, player1, player2):
    iterations = 10 ** 6
    epsilon = 0.9
    min_eps = 0.01
    s_t = time.time()
    decay_eps = (epsilon - min_eps) / iterations
    alp_s = 0.5
    alp = 1.0
    min_alp = 0.001
    decay_alp = (alp - min_alp) / iterations
    gamma = 0.9

    w = 8
    h = 8
    d = 2
    act = 5
    q_1_val = np.zeros([h, w, d, act])
    q_2_val = np.zeros([h, w, d, act])

    game.setMatch()
    done = 1

    err = []
    iter = []

    for i in range(iterations):

        if done == 1:
            game.setMatch()

        loc_1 = player1.loc
        loc_2 = player2.loc
        ball = game.soccerball

        p_q_v = q_1_val[2, 1, 1, 2]


        if epsilon > np.random.random():
            act_1_val = np.random.choice(act)
            act_2_val = np.random.choice(act)
        else:
            act_1_val = np.argmax(q_1_val[loc_1, loc_2, ball])

            act_2_val = np.random.choice(act)

        n_s, rewd_1, rewd_2, done = game.next_step(act1=act_1_val, act2=act_2_val)
        na, nb, nball = n_s

        q_1_val[loc_1, loc_2, ball, act_1_val] = (1 - alp) * q_1_val[loc_1, loc_2, ball, act_1_val] + \
                                                 alp * ((1 - gamma) * rewd_1 + gamma * np.max(q_1_val[na, nb, nball]))



        if [loc_1, loc_2, ball, act_1_val, act_2_val] == [2, 1, 1, 2, 4]:
            err.append(abs(q_1_val[2, 1, 1, 2] - p_q_v))
            print("Iteration: ", i, alp)
            iter.append(i)


        alp -= decay_alp


    print("Time:", time.time() - s_t)


    plotgraph(err, iter, name="Q-Learner", linewidth=0.5)





def foe_q(game, player1, player2):
    iterations = 10 ** 6
    alp = 0.9
    min_alp = 0.001
    decay_alp = (alp - min_alp) / 10 ** 6
    gamma = 0.9

    s_t = time.time()
    w = 8
    h = 8
    d = 2
    act = 5
    _Q_table = np.zeros([h, w, d, act, act])
    game.setMatch()
    done = 1

    errt = []
    iter = []

    for i in range(0,10**6):


        while done:
            game.setMatch()
            done = 0

        loc_1 = player1.loc
        loc_2 = player2.loc
        ball = game.soccerball

        p_q_v = _Q_table[2, 1, 1, 2, 4]

        q_cur_val = _Q_table[player1.loc, player2.loc, game.soccerball]
        act_1_Pl = np.random.choice(act)
        act_2_pl = np.random.choice(act)



        n_s, rewd_1, rewd_2, done = game.next_step(act1=act_1_Pl, act2=act_2_pl)
        pl_1_n_act, pl_2_n_act, ball_n_game = n_s
        _ = _Q_table[pl_1_n_act, pl_2_n_act, ball_n_game]

        s_prime_goal = solve_maximin(q_cur_val)

        _Q_table[loc_1, loc_2, ball, act_1_Pl, act_2_pl] = (1 - alp) * _Q_table[loc_1, loc_2, ball, act_1_Pl, act_2_pl] + \
                                                           alp * ((1 - gamma) * rewd_1 + gamma * s_prime_goal)



        if [loc_1, loc_2, ball, act_1_Pl, act_2_pl] == [2, 1, 1, 2, 4]:
            errt.append(abs(_Q_table[2, 1, 1, 2, 4] - p_q_v))
            iter.append(i)
            print("Iteration: ", i)


        alp *= np.e ** (-np.log(500.0) / 10 ** 6)

    print(_Q_table[2, 1, 1])
    print("Time:", time.time() - s_t)


    plotgraph(errt, iter, name="Foe-Q")
def solve_maximin(q):
    glpksolver = 'glpk'
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_ON'}
    solvers.options['msg_lev'] = 'GLP_MSG_ON'
    solvers.options['LPX_K_MSGLEV'] = 0

    mat = matrix(q).trans()
    Nsize = mat.size[1]

    Fmat = hstack((ones((mat.size[0], 1)), mat))

    buff = hstack((zeros((Nsize, 1)), -eye(Nsize)))

    Fmat = vstack((Fmat, buff))

    Fmat = matrix(vstack((Fmat, hstack((0, ones(Nsize))), hstack((0, -ones(Nsize))))))


    hstack_t = matrix(hstack((zeros(Fmat.size[0] - 2), [1, -1])))


    h_stack_m = matrix(hstack(([-1], zeros(Nsize))))


    res = solvers.lp(h_stack_m, Fmat, hstack_t, solver=glpksolver)

    return res['dual objective']


def foe_q(game, player1, player2):
    iterations = 10 ** 6
    alp = 0.9
    min_alp = 0.001
    decay_alp = (alp - min_alp) / 10 ** 6
    gamma = 0.9

    s_t = time.time()
    w = 8
    h = 8
    d = 2
    act = 5
    _Q_table = np.zeros([h, w, d, act, act])
    game.setMatch()
    done = 1

    errt = []
    iter = []

    for i in range(0,10**6):


        while done:
            game.setMatch()
            done = 0

        loc_1 = player1.loc
        loc_2 = player2.loc
        ball = game.soccerball

        p_q_v = _Q_table[2, 1, 1, 2, 4]

        q_cur_val = _Q_table[player1.loc, player2.loc, game.soccerball]
        act_1_Pl = np.random.choice(act)
        act_2_pl = np.random.choice(act)



        n_s, rewd_1, rewd_2, done = game.next_step(act1=act_1_Pl, act2=act_2_pl)
        pl_1_n_act, pl_2_n_act, ball_n_game = n_s
        _ = _Q_table[pl_1_n_act, pl_2_n_act, ball_n_game]

        s_prime_goal = solve_maximin(q_cur_val)

        _Q_table[loc_1, loc_2, ball, act_1_Pl, act_2_pl] = (1 - alp) * _Q_table[loc_1, loc_2, ball, act_1_Pl, act_2_pl] + \
                                                           alp * ((1 - gamma) * rewd_1 + gamma * s_prime_goal)



        if [loc_1, loc_2, ball, act_1_Pl, act_2_pl] == [2, 1, 1, 2, 4]:
            errt.append(abs(_Q_table[2, 1, 1, 2, 4] - p_q_v))
            iter.append(i)
            print("Iteration: ", i)


        alp *= np.e ** (-np.log(500.0) / 10 ** 6)

    print(_Q_table[2, 1, 1])
    print("Time:", time.time() - s_t)


    plotgraph(errt, iter, name="Foe-Q")
def ce_q(game, player1, player2):
    iterations = 10 ** 6
    alp = 0.9
    min_alp = 0.001
    decay_alp = (alp - min_alp) / 10 ** 6
    gamma = 0.9
    s_t = time.time()
    w = 8
    h = 8
    d = 2
    acts = 5


    q_1 = np.zeros([h, w, d, acts, acts])
    q_2 = np.zeros([h, w, d, acts, acts])

    game.setMatch()
    done=1

    err = []
    iter = []

    for i in range(iterations):

        while (done):
            game.setMatch()
            done = 0

        a = player1.loc
        b = player2.loc
        ball = game.soccerball

        prev_q_val = q_1[2, 1, 1, 2, 4]

        act_1_val = np.random.choice(acts)
        act_2_val = np.random.choice(acts)

        q_val_1 = q_1[player1.loc, player2.loc, game.soccerball]
        q_val_2 = q_2[player1.loc, player2.loc, game.soccerball]


        nex_s, rewd_1, rewd_2, done = game.next_step(act1=act_1_val, act2=act_2_val)
        na, nb, nball = nex_s


        r_exp_A, r_exp_B = CEQ_imp(q_val_1, q_val_2)

        q_1[a, b, ball, act_1_val, act_2_val] = (1 - alp) * q_1[a, b, ball, act_1_val, act_2_val] + \
                                                alp * ((1 - gamma) * rewd_1 + gamma * r_exp_A)

        q_2[a, b, ball, act_1_val, act_2_val] = (1 - alp) * q_2[a, b, ball, act_1_val, act_2_val] + \
                                                alp * ((1 - gamma) * rewd_2 + gamma * r_exp_B)

        if [a, b, ball, act_1_val, act_2_val] == [2, 1, 1, 2, 4]:
            err.append(abs(q_1[2, 1, 1, 2, 4] - prev_q_val))
            iter.append(i)
            print("Iteration: ", i)

        alp *= np.e ** (-np.log(500.0) / 10 ** 6)

    plotgraph(err, iter, name="CE-Q")

    print(q_1[2, 1, 1])
    print("Total Time:", time.time() - s_t)


def CEQ_imp(valq1, valq2):
    glpksolver = 'glpk'
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_ON'}
    solvers.options['msg_lev'] = 'GLP_MSG_ON'
    solvers.options['LPX_K_MSGLEV'] = 0


    mat = matrix(valq1).trans()
    Nsize = mat.size[1]

    Fmat = np.zeros((2 * Nsize * (Nsize - 1), (Nsize * Nsize)))
    valq1 = np.array(valq1)
    valq2 = np.array(valq2)
    r = 0


    for i in range(Nsize):
        for j in range(Nsize):
            if i != j:
                Fmat[r, i * Nsize:(i + 1) * Nsize] = valq1[i] - valq1[j]
                Fmat[r + Nsize * (Nsize - 1), i:(Nsize * Nsize):Nsize] = valq2[:, i] - valq2[:, j]
                r += 1

    Fmat = matrix(Fmat)


    Fmat = hstack((ones((Fmat.size[0], 1)), Fmat))

    buff = hstack((zeros((Nsize * Nsize, 1)), -eye(Nsize * Nsize)))

    Fmat = vstack((Fmat, buff))

    Fmat = matrix(vstack((Fmat, hstack((0, ones(Nsize * Nsize))), hstack((0, -ones(Nsize * Nsize))))))


    hstack_t = matrix(hstack((zeros(Fmat.size[0] - 2), [1, -1])))



    hstack_flat = matrix(hstack(([-1.], -(valq1 + valq2).flatten())))

    res = solvers.lp(hstack_flat, Fmat, hstack_t, solver=glpksolver)


    if res['x'] is None:
        return 0, 0
    f_res = res['x'][1:]
    q1_flat = valq1.flatten()
    q2_flat = valq2.transpose().flatten()

    f_q_ret1 = np.matmul(q1_flat, f_res)[0]
    f_q_ret2 = np.matmul(q2_flat, f_res)[0]


    return f_q_ret1, f_q_ret2

if __name__ == "__main__":
    player1 = player(name="1", ball=0)
    player2 = player(name="2", ball=1)
    game = match(player1, player2)
    ce_q(game, player1, player2)
    foe_q(game, player1, player2)
    qlearning(game, player1, player2)
    friend_q(game, player1, player2)


