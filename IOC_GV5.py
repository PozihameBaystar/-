import numpy as np
import math
import numba
from dataclasses import dataclass
import jax.numpy as jnp
import casadi as ca

def func_disc(x,u):
    x_next = x + u
    return x_next

A = np.array([[1,0],
              [0,1]])

B = np.array([[1,0],
              [0,1]])

@dataclass
class Args:
    Q = np.array([[1,0],
                  [0,1]])
    R = np.array([[1,0],
                  [0,1]]) * 1
    S = np.array([[0,0],
                  [0,0]])
    N = 2
    x_ob = np.array([5,5])
    q = -x_ob @ Q
    r = np.array([0,0])
    s = -x_ob @ S
    
args = Args()

#@numba.jit
def LQR_control(x):
    
    def Backward_Pass():

        Sk = args.S
        sk = args.s

        Ks = []
        ds = []

        for i in reversed(range(args.N)):
            K_ = - np.linalg.inv(args.R + B.T@Sk@B) @ B.T @ Sk @ A
            d_ = - np.linalg.inv(args.R + B.T@Sk@B) @ (args.r.T + B.T@sk.T)
            Ks.append(K_)
            ds.append(d_)
            Sk_ = args.Q + K_.T @ (args.R + B.T@Sk@B) @ K_ \
                + A.T @ Sk @ A + A.T @ Sk @ B @ K_ + K_ @ B.T @ Sk @ A
            sk_ = args.q + (d_.T@args.R+args.r) @ K_ + (d_.T@B.T@Sk) @ (A+B@K_)
            Sk = Sk_
            sk = sk_

        return np.flip(Ks,axis=0), np.flip(ds,axis=0)
    

    def Forward_Pass(x,Ks,ds):
        
        xs = []
        us = []

        xs.append(x)
        
        for i in range(args.N):
            K = Ks[i]
            d = ds[i]
            u = K @ xs[i] + d
            x_next = func_disc(xs[i],u)
            xs.append(x_next)
            us.append(u)

        return xs, us
    
    Ks, ds = Backward_Pass()
    xs, us = Forward_Pass(x,Ks,ds)

    return xs, us


def cost_hessian(xs,us):
    #A_cos = np.array([[xs[0][0]-args.x_ob[0], 0, 0, 0],
    #                  [0, xs[0][1]-args.x_ob[1], 0, 0],
    #                  [xs[1][0]-args.x_ob[0], 0, 0, 0],
    #                  [0, xs[1][1]-args.x_ob[1], 0, 0],
    #                  [0, 0, us[0][0], 0],
    #                  [0, 0, 0, us[0][1]],
    #                  [0, 0, us[1][0], 0],
    #                  [0, 0, 0, us[1][1]]])

    A_cos = np.array([[xs[1][0]-args.x_ob[0], 0, 0, 0],
                      [0, xs[1][1]-args.x_ob[1], 0, 0],
                      [0, 0, us[0][0], 0],
                      [0, 0, 0, us[0][1]],
                      [0, 0, us[1][0], 0],
                      [0, 0, 0, us[1][1]]])
    return A_cos


# データ格納バッファ
class ReplayBuffer():

    def __init__(self, total_timesteps, episode_length, obs_dim, action_dim):

        # バッファの定義
        num_episode = (total_timesteps // episode_length) + 1
        self.obs_buffer = np.empty((num_episode, episode_length + 1, obs_dim), dtype=np.float32)
        self.action_buffer = np.empty((num_episode, episode_length, action_dim), dtype=np.float32)
        #self.reward_buffer = np.empty((num_episode, episode_length, 1), dtype=np.float32)

        self.episode_length = episode_length
        self.iter = 0


    def add(self, obss, actions):

        # バッファにシーケンスを格納
        self.obs_buffer[self.iter] = obss
        self.action_buffer[self.iter] = actions
        #self.reward_buffer[self.iter] = rewards[:, None]
        self.iter += 1


    def sample(self, batch_size, horizon):

        # バッファから取得する要素の index をサンプリング
        idx = np.random.randint(self.iter, size=batch_size)
        idy = np.random.randint(self.episode_length - horizon, size=batch_size)

        obs_dim = self.obs_buffer.shape[2]
        action_dim = self.action_buffer.shape[2]

        # バッファからデータを取得
        obss = np.empty((horizon, batch_size, obs_dim), dtype=np.float32)
        #obss = np.empty((horizon, batch_size, obs_dim), dtype=np.float32)
        actions = np.empty((horizon, batch_size, action_dim), dtype=np.float32)
        #rewards = np.empty((horizon, batch_size, 1), dtype=np.float32)
        for t in range(horizon):
            obss[t] = self.obs_buffer[idx, idy+t]
            actions[t] = self.action_buffer[idx, idy+t]
            #rewards[t] = self.reward_buffer[idx, idy+t]
        #obss[horizon] = self.obs_buffer[idx, idy+horizon]
        return jnp.array(obss), jnp.array(actions) #, jnp.array(rewards)





## ここから実験コード
if __name__ == "__main__":

    x0 = np.array([0,0])
    xs, us = LQR_control(x0)

    A_cos = cost_hessian(xs,us)

    A_ca = ca.DM(A_cos)

    w = ca.SX.sym("w",4)

    ioc = {
            "x":w,
            "f":w.T @ A_ca.T @ A_ca @ w,
            "g":ca.vertcat(
                w[0]**2 + w[1]**2 + w[2]**2 + w[3]**2,
                w[0],
                w[1],
                w[2],
                w[3]
            )
            }
        
    S = ca.nlpsol("S","ipopt",ioc)
    r = S(x0=[0,0,0,0],
        lbg=[1,0,0,0,0],
        ubg=[1,np.inf,np.inf,np.inf,np.inf])
    w_opt = np.array(r["x"])
    print(A_cos @ w_opt)
    print(w_opt)