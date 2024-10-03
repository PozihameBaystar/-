import jax
import jax.numpy as jnp
import numpy as np
import math
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## ロボットのモデルに関する関数

@dataclass
class Model:
    R = 0.05
    T = 0.2
    r = R/2
    rT = r/T


model = Model()

#離散化状態方程式
@jax.jit
def model_func_risan(x, u, dt):
    Bk = jnp.array([[jnp.cos(x[2]), 0],
                    [jnp.sin(x[2]), 0],
                    [0, 1]], dtype=jnp.float32) * dt
    A = jnp.concatenate([jnp.zeros((3,3),dtype=jnp.float32), Bk],1)
    A = jnp.concatenate([A,jnp.zeros((2,5),dtype=jnp.float32)],0)
    B = jnp.concatenate([Bk,jnp.eye(2,dtype=jnp.float32)],0)
    x_next = x + A @ x + B @ u
    return x_next

model_dfdx = jax.jit(jax.jacfwd(model_func_risan,0))
model_dfdu = jax.jit(jax.jacfwd(model_func_risan,1))

## コントローラーに関する関数

@dataclass
class Cont_Args:

    # コントローラーのパラメータ
    Ts = 0.1
    tf = 1.0
    N = 10
    dt = Ts
    iter = 10
    torelance = 1.0

    # 評価関数中の重み
    # 状態変数の項
    Q = jnp.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=jnp.float32) * 100

    # 制御入力の項
    R = jnp.array([[1,0],
                   [0,1]], dtype=jnp.float32) * 10

    # 制御入力の加速度項
    R_del = jnp.array([[1,0],
                       [0,1]], dtype=jnp.float32) * 10000

    # 最終地点の項
    S = jnp.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=jnp.float32) * 100

    # 目標地点
    x_ob = jnp.array([5, 5, 0], dtype=jnp.float32)

    # 目標入力
    u_ob = jnp.array([0, 0], dtype=jnp.float32)

    # 次元データ
    len_x = 5
    len_u = 2

    # 状態と入力
    x = None
    u = None
    us = None
    
    # カーネル関数の変数
    sigma = 1.0
    kyori = [0.45, 1.2, 3.6]
    weights = jnp.zeros((3,3),dtype=jnp.float32)

    # 想定するノイズの平均値
    mean = jnp.zeros((2,),dtype=jnp.float32)
    # 想定するノイズの共分散行列
    cov = 0.1*jnp.eye(2,dtype=jnp.float32)
    # Risk-sensitiveの定数θ
    theta = 1.0

    # 障害物の場所
    ev_pos = jnp.array([[2.0,0.1,0]],dtype=jnp.float32)
    # 障害物から取るべき距離
    d = 0.3

    # 緩和対数バリア関数の緩和値
    del_bar = 0.05
    # 計算用の行列
    E = jnp.array([[1,0,0],
                   [0,1,0],
                   [0,0,0]],dtype=jnp.float32)

    # バリア関数の重み
    r = 0

    # 速度制限
    umax = jnp.array([1,1],dtype=jnp.float32)
    umin = jnp.array([-1,-1],dtype=jnp.float32)

    # 計算用行列
    bar_C = np.concatenate([jnp.eye(len_u,dtype=jnp.float32),-jnp.eye(len_u,dtype=jnp.float32)],0)
    bar_d = np.concatenate([umax,-umin],0)

    # 速度制限の重み
    b = 50


args = Cont_Args()


# バリア関数（-logか二次関数かを勝手に切り替える
@jax.jit
def barrier_z(z):
    pred = z > args.del_bar
    true_fun = lambda z: - jnp.log(z)
    false_fun = lambda z: 0.5 * (((z - 2*args.del_bar) / args.del_bar)**2 - 1) - jnp.log(args.del_bar)
    return jax.lax.cond(pred, true_fun, false_fun, z)


# バリア関数（全体）
@jax.jit
def barrier(u):

    zs = args.bar_d - args.bar_C @ u

    def vmap_fun(b, z, margin=0.5):
        return b * jnp.where(z>=margin,barrier_z(margin),barrier_z(z))

    Bars = jax.vmap(vmap_fun, (None,0))(args.b, zs)
    Bar = jnp.sum(Bars)
    return Bar


# 対数バリア関数型回避関数
@jax.jit
def evasion(x):

    def vmap_fun(x, xe, r, d, margin=0.5):
        distance = jnp.linalg.norm(x-xe,ord=2)
        z = distance**2 - d**2
        kijun = d + margin
        return r * jnp.where(distance>=kijun, barrier_z(kijun**2-d**2), barrier_z(z))

    evas = jax.vmap(vmap_fun, (None, 0, None, None))(x, args.ev_pos, args.r, args.d)
    eva = jnp.sum(evas)
    return eva


# カーネル関数
def kernel(x,kyori,pos):
    return jnp.exp( -1/(2*args.sigma**2) * (jnp.linalg.norm(x[:2]-pos[:2])-kyori)**2 )
    
def weight_kernel(x,weights,pos):
    return 1 + weights[0]*kernel(x,args.kyori[0],pos) + weights[1]*kernel(x,args.kyori[1],pos) + weights[2]*kernel(x,args.kyori[2],pos)

# ステージコスト
def stage_cost(x,u):
    cost = 0.5 * ( weight_kernel(x,args.weights[0],args.x_ob) * (x[:3]-args.x_ob).T @ args.Q @ (x[:3]-args.x_ob) \
                + weight_kernel(x,args.weights[1],args.x_ob) * (x[3:5]+u-args.u_ob).T @ args.R @ (x[3:5]+u-args.u_ob) \
                + weight_kernel(x,args.weights[2],args.x_ob) * u @ args.R_del @ u ) \
                + barrier(x[3:5])
    return cost

# 終端コスト
def term_cost(x):
    cost = 0.5 * (x[:3]-args.x_ob).T @ args.S @ (x[:3]-args.x_ob)
    return cost

# ステージコストの微分
grad_x_stage = jax.jit(jax.grad(stage_cost,0))
grad_u_stage = jax.jit(jax.grad(stage_cost,1))
hes_x_stage = jax.jit(jax.hessian(stage_cost,0))
hes_u_stage = jax.jit(jax.hessian(stage_cost,1))
hes_ux_stage = jax.jit(jax.jacfwd(jax.grad(stage_cost,1),0))

# 終端コストの微分
grad_x_term = jax.jit(jax.grad(term_cost))
hes_x_term = jax.jit(jax.hessian(term_cost))


# コントローラー関数
@jax.jit
def iLQR_control(x,us):

    def rollout(x_init,us):

        def rollout_body(carry,u):
            x = carry
            x = model_func_risan(x,u,args.dt)
            return x, x
        
        _, xs = jax.lax.scan(rollout_body, x_init, us)
        xs = jnp.concatenate([x_init[None], xs])

        return xs
    
    
    def calcJ(xs,us):

        J = 0 #評価関数の値を初期化

        def calcJ_body(carry,val):
            _ = None
            J = carry
            x, u = val
            J += stage_cost(x,u)
            return J, _
        
        J, _ = jax.lax.scan(calcJ_body, J, (xs[:-1],us))
        J += term_cost(xs[-1])

        J = jnp.float32(J)

        return J
    

    def Backward(xs,us):
        Vxx = hes_x_term(xs[-1])
        Vx = grad_x_term(xs[-1])

        dV1 = 0
        dV2 = 0

        def Backward_body(carry, val):
            Vx, Vxx, dV1, dV2 = carry
            x, u = val

            Ak = model_dfdx(x,u,args.dt)
            Bk = model_dfdu(x,u,args.dt)
            #Ck = Bk

            ## iLEQGにするにあたって min max 問題のmaxの部分の計算を済ませておく
            #cov = args.cov
            #Vx = Vx + Vx @ Ck @ jnp.linalg.inv( (1/args.theta) * jnp.linalg.pinv(cov) - Ck.T @ Vxx @ Ck ) @ Ck.T @ Vxx
            #Vxx = Vxx + Vxx @ Ck @ jnp.linalg.inv( (1/args.theta) * jnp.linalg.pinv(cov) - Ck.T @ Vxx @ Ck ) @ Ck.T @ Vxx

            Qx = grad_x_stage(x,u) + Vx @ Ak
            Qxx = hes_x_stage(x,u) + Ak.T @ Vxx @ Ak

            Qu = grad_u_stage(x,u) + Vx @ Bk
            Quu = hes_u_stage(x,u) + Bk.T @ Vxx @ Bk

            Qux = hes_ux_stage(x,u) + Bk.T @ Vxx @ Ak

            #Quuが正定かどうかの判定
            try:
                kekka = jnp.linalg.cholesky(Quu)
            except:
                #もし違ったら
                #正定化の為にまず固有値の最小値を特定する
                alpa = -jnp.amin(jnp.linalg.eig(Quu))
                Quu = Quu + (alpa + 1e-5) * jnp.eye(args.len_u) #正定化

            K = - jnp.linalg.pinv(Quu) @ Qux # 閉ループゲインの計算
            d = - jnp.linalg.pinv(Quu) @ Qu.T # 開ループゲインの計算

            dV1 += Qu @ d
            dV2 += 0.5 * d.T @ Quu @ d

            Vx = Qx + d.T @ Quu @ K + Qu @ K + d.T @ Qux # Vxの更新
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K # Vxxの更新

            return (Vx, Vxx, dV1, dV2), (K, d)
        
        carry, output_val = jax.lax.scan(Backward_body, (Vx, Vxx, dV1, dV2), (jnp.flip(xs[:-1], 0), jnp.flip(us, 0)))
        dV1 = carry[2]
        dV2 = carry[3]
        Ks, ds = output_val

        Ks = jnp.flip(Ks, 0)
        ds = jnp.flip(ds, 0)
        
        return Ks, ds, dV1, dV2
    
    
    def Forward(xs,us,Ks,ds,dV1,dV2,J):

        z = 1e5
        alpha = 1.0 #直線探索の係数を初期化
        ls_iteration = 0 #直線探索を何回やったかの変数

        # 直線探索でのロールアウト関数
        def ls_rollout(xs,us,Ks,ds,alpha):

            x_init = xs[0]

            def ls_rollout_body(carry,val):
                x_ = carry
                x, u, K, d = val
                u_ = u + K @ (x_-x) + alpha * d
                x_ = model_func_risan(x_,u_,args.dt)
                return x_, (x_,u_)
        
            _, output_val = jax.lax.scan(ls_rollout_body, x_init, (xs[:-1],us,Ks,ds))
            xs_, us_ = output_val
            xs_ = jnp.concatenate([x_init[None], xs_])
            
            return xs_, us_

        # 直線探索が完了したかをチェックする関数
        def ls_check(val):
            z, _, ls_iteration, _, _, _ = val
            return jnp.logical_and((jnp.logical_or(1e-4 > z, z > 10)), ls_iteration < 10)
        
        # Forward Pass 一反復分の関数
        def Forwrd_body(val):
            z, alpha, ls_iteration, xs_, us_, J_ = val

            xs_, us_ = ls_rollout(xs,us,Ks,ds,alpha) #新しい状態予測と入力
            J_ = calcJ(xs_,us_) #新しい状態と予測での評価関数値
            z = (J-J_)/-(alpha*dV1+alpha**2*dV2)
            alpha = 0.5*alpha #係数の更新
            ls_iteration += 1 #反復回数の更新

            return (z, alpha, ls_iteration, xs_, us_, J_)
        
        output_val = jax.lax.while_loop(ls_check, Forwrd_body, (z,alpha,ls_iteration,xs,us,J))
        _, _, _, xs, us, J_ = output_val

        return xs, us, J_
    

    # 収束判定関数
    def conv_check(val):
        _, _, J_new, J_old, iteration = val
        return jnp.logical_and(abs(J_old-J_new) > args.torelance , iteration < args.iter)
    

    def iLQR_body(val):
        xs, us, J_old, _, iteration = val

        Ks, ds, dV1, dV2 = Backward(xs,us)
        xs, us, J_new = Forward(xs,us,Ks,ds,dV1,dV2,J_old)

        iteration += 1 #反復回数の更新

        return (xs, us, J_new, J_old, iteration)

    
    # iLQRループ全体

    xs = rollout(x,us)
    J = calcJ(xs,us)
    iteration = 0 #LQRステップの繰り返し数をリセット

    val = jax.lax.while_loop(conv_check, iLQR_body, (xs,us,J,J+1e5,iteration)) #whileループが最初で止まらない様に1e5の差をつける
    xs, us, _, _, _ = val

    return us


########### ここからIOC

## パラメーター
@dataclass
class ioc_args:
    # 評価関数関連
    S = jnp.array([[1,0,0],
                   [0,1,0],
                   [0,0,0]],dtype=jnp.float32) * 100
    Q = jnp.array([[1,0,0],
                   [0,1,0],
                   [0,0,0]],dtype=jnp.float32) * 100
    R = 1 * jnp.eye(2,dtype=jnp.float32)  # 速度の重み
    R_del = 100 * jnp.eye(2,dtype=jnp.float32)  # 速度変化の重み
    D = 1.0  # 目的方向への余弦の重み
    E = 1.0  # 回避関数の重み
    b = 50   # バリア関数の重み
    x_ob = jnp.array([5,5,0],dtype=jnp.float32)
    x_ev = jnp.array([3,2,0],dtype=jnp.float32)
    u_ob = jnp.array([0,0],dtype=jnp.float32)
    alpha = 1e4 # 逆最適制御の尖り具合
    sigma = 1.0 # カーネル関数の分散
    kyori = [0.45, 1.2, 3.6]
    weights = jnp.zeros((3,3),dtype=jnp.float32)
    len_u = 2
    obs_dim = 5
    action_dim = 2
    # コントローラーのその他のパラメータ
    Ts = 0.1
    tf = 1.0
    N = 10
    dt = Ts

ioc_args = ioc_args()



# 尤度関数
@jax.jit
def IOC_yuudo(xs,us,S,Q,R,R_del,weightss):

    
    # カーネル関数
    def kernel(x,kyori,pos):
        return jnp.exp( -1/(2*ioc_args.sigma**2) * (jnp.linalg.norm(x[:2]-pos[:2])-kyori)**2 )
    
    def weight_kernel(x,weights,pos):
        return 1 + weights[0]*kernel(x,ioc_args.kyori[0],pos) + weights[1]*kernel(x,ioc_args.kyori[1],pos) + weights[2]*kernel(x,ioc_args.kyori[2],pos)

    # ステージコスト
    def stage_cost(x,u):
        cost = 0.5 * ( weight_kernel(x,weightss[0],ioc_args.x_ob) * (x[:3]-ioc_args.x_ob).T @ Q @ (x[:3]-ioc_args.x_ob) \
                    + weight_kernel(x,weightss[1],ioc_args.x_ob) * (x[3:5]+u-ioc_args.u_ob).T @ R @ (x[3:5]+u-ioc_args.u_ob) \
                    + weight_kernel(x,weightss[2],ioc_args.x_ob) * u @ R_del @ u ) \
                    + barrier(x[3:5])
        return cost

    # 終端コスト
    def term_cost(x):
        cost = 0.5 * (x[:3]-ioc_args.x_ob).T @ S @ (x[:3]-ioc_args.x_ob)
        return cost

    # ステージコストの微分（状態変数と入力による微分
    grad_x_stage = jax.jit(jax.grad(stage_cost,0))
    grad_u_stage = jax.jit(jax.grad(stage_cost,1))
    hes_x_stage = jax.jit(jax.hessian(stage_cost,0))
    hes_u_stage = jax.jit(jax.hessian(stage_cost,1))
    hes_ux_stage = jax.jit(jax.jacfwd(jax.grad(stage_cost,1),0))

    # 終端コストの微分
    grad_x_term = jax.jit(jax.grad(term_cost))
    hes_x_term = jax.jit(jax.hessian(term_cost))

    def Backward(xs,us):

        Vxx = hes_x_term(xs[-1])
        Vx = grad_x_term(xs[-1])

        L = 0 # 尤度関数の値

        def Backward_body(carry, val):
            Vx, Vxx, L = carry
            x, u = val

            Ak = model_dfdx(x,u,ioc_args.dt)
            Bk = model_dfdu(x,u,ioc_args.dt)

            Qx = grad_x_stage(x,u) + Vx @ Ak
            Qxx = hes_x_stage(x,u) + Ak.T @ Vxx @ Ak

            Qu = grad_u_stage(x,u) + Vx @ Bk
            Quu = hes_u_stage(x,u) + Bk.T @ Vxx @ Bk

            Qux = hes_ux_stage(x,u) + Bk.T @ Vxx @ Ak

            #Quuが正定かどうかの判定
            try:
                kekka = jnp.linalg.cholesky(Quu)
            except:
                #もし違ったら
                #正定化の為にまず固有値の最小値を特定する
                alpa = -jnp.amin(jnp.linalg.eig(Quu))
                Quu = Quu + (alpa + 1) * jnp.eye(ioc_args.len_u) #正定化

            # 尤度関数の更新
            L = - (1/2*ioc_args.alpha) * Qu @ jnp.linalg.pinv(Quu) @ Qu.T \
                + (1/2) * jnp.log(jnp.linalg.det(Quu)) - (ioc_args.action_dim/2) * jnp.log(2*jnp.pi*ioc_args.alpha)

            K = - jnp.linalg.pinv(Quu) @ Qux # 閉ループゲインの計算
            d = - jnp.linalg.pinv(Quu) @ Qu.T # 開ループゲインの計算

            Vx = Qx + d.T @ Quu @ K + Qu @ K + d.T @ Qux # Vxの更新
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K # Vxxの更新

            return (Vx, Vxx, L), (K, d)
        
        
        carry, _ = jax.lax.scan(Backward_body, (Vx, Vxx, L), (jnp.flip(xs[:-1], 0), jnp.flip(us, 0)))
        L = carry[2]
        
        return L
    
    L = Backward(xs,us)
    
    return L


# 尤度関数の勾配計算
grad_L_S = jax.jit(jax.grad(IOC_yuudo,2))
grad_L_Q = jax.jit(jax.grad(IOC_yuudo,3))
grad_L_R = jax.jit(jax.grad(IOC_yuudo,4))
grad_L_Rd = jax.jit(jax.grad(IOC_yuudo,5))
grad_L_W = jax.jit(jax.grad(IOC_yuudo,6))



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
        idx = np.random.randint(self.iter, size=batch_size) # どのエピソードからデータを取るか
        idy = np.random.randint(self.episode_length - horizon, size=batch_size) # そのエピソードの内どこから取るか

        obs_dim = self.obs_buffer.shape[2]
        action_dim = self.action_buffer.shape[2]

        # バッファからデータを取得
        obss = np.empty((horizon+1, batch_size, obs_dim), dtype=np.float32)
        actions = np.empty((horizon, batch_size, action_dim), dtype=np.float32)
        #rewards = np.empty((horizon, batch_size, 1), dtype=np.float32)
        for t in range(horizon):
            obss[t] = self.obs_buffer[idx, idy+t]
            actions[t] = self.action_buffer[idx, idy+t]
            #rewards[t] = self.reward_buffer[idx, idy+t]
        obss[horizon] = self.obs_buffer[idx, idy+horizon]
        return jnp.array(obss), jnp.array(actions) #, jnp.array(rewards)






## ここから実験コード
if __name__ == "__main__":


    # 初期条件
    args.u = jnp.zeros((args.len_u), dtype=jnp.float32)
    args.us = jnp.zeros((args.N, args.len_u), dtype=jnp.float32)
    args.x = jnp.zeros((args.len_x), dtype=jnp.float32)

    Time = 0
    x_log = []
    u_log = []


    # データ格納庫の用意
    rb = ReplayBuffer(1, int(20/args.Ts), ioc_args.obs_dim, ioc_args.action_dim)

    # データ作成のループ
    while Time <= 20:
        print("-------------Position-------------")
        print(args.x)
        print("-------------Input-------------")
        print(args.u)

        x_log.append(args.x)
        u_log.append(args.u)

        x = model_func_risan(args.x,args.u,args.dt)

        us = iLQR_control(args.x,args.us)
        
        Time += args.Ts
        args.x = x
        args.u = us[0]
        args.us = us

    x_log.append(args.x)

    rb.add(x_log, u_log)



    # 学習ループ

    lr = 1e-4 # 学習率

    for loop in range(10000):

        bat_xs, bat_us = rb.sample(10,args.N)

        for bat in range(10):
            #print(IOC_yuudo(bat_xs[:,bat],bat_us[:,bat],ioc_args.S,ioc_args.Q,ioc_args.R,ioc_args.R_del,ioc_args.weights))
            grad_L_S_ = grad_L_S(bat_xs[:,bat],bat_us[:,bat],ioc_args.S,ioc_args.Q,ioc_args.R,ioc_args.R_del,ioc_args.weights)
            grad_L_Q_ = grad_L_Q(bat_xs[:,bat],bat_us[:,bat],ioc_args.S,ioc_args.Q,ioc_args.R,ioc_args.R_del,ioc_args.weights)
            grad_L_R_ = grad_L_R(bat_xs[:,bat],bat_us[:,bat],ioc_args.S,ioc_args.Q,ioc_args.R,ioc_args.R_del,ioc_args.weights)
            grad_L_Rd_ = grad_L_Rd(bat_xs[:,bat],bat_us[:,bat],ioc_args.S,ioc_args.Q,ioc_args.R,ioc_args.R_del,ioc_args.weights)

            grad_L_S_ = jnp.diag(jnp.diag(grad_L_S_))
            grad_L_Q_ = jnp.diag(jnp.diag(grad_L_Q_))
            grad_L_R_ = jnp.diag(jnp.diag(grad_L_R_))
            grad_L_Rd_ = jnp.diag(jnp.diag(grad_L_Rd_))
            #print(grad_L_R_)

            grad_L_S_ = jnp.sum(jnp.diag(grad_L_S_))
            grad_L_Q_ = jnp.sum(jnp.diag(grad_L_Q_))
            grad_L_R_ = jnp.sum(jnp.diag(grad_L_R_))
            grad_L_Rd_ = jnp.sum(jnp.diag(grad_L_Rd_))
            print(grad_L_Rd_)


            #grad_L_S_ = grad_L_S_.at[2,2].set(0)
            #grad_L_Q_ = grad_L_Q_.at[2,2].set(0)

            #ioc_args.S = jnp.maximum(0,ioc_args.S + lr * grad_L_S_)
            #ioc_args.Q = jnp.maximum(0,ioc_args.Q + lr * grad_L_Q_)
            #ioc_args.R = jnp.maximum(0,ioc_args.R + lr * grad_L_R_)
            #ioc_args.R_del = jnp.maximum(0,ioc_args.R_del + lr * grad_L_Rd_)

            ioc_args.S = jnp.maximum(0,ioc_args.S + lr * grad_L_S_ * jnp.array([[1,0,0],
                                                                                [0,1,0],
                                                                                [0,0,0]],dtype=jnp.float32))
            ioc_args.Q = jnp.maximum(0,ioc_args.Q + lr * grad_L_Q_ * jnp.array([[1,0,0],
                                                                                [0,1,0],
                                                                                [0,0,0]],dtype=jnp.float32))
            ioc_args.R = jnp.maximum(0,ioc_args.R + lr * grad_L_R_ * jnp.eye(2,dtype=jnp.float32))
            ioc_args.R_del = jnp.maximum(0,ioc_args.R_del + lr * grad_L_Rd_ * jnp.eye(2,dtype=jnp.float32))

        #print(ioc_args.S)
        #print(ioc_args.Q)
        #print(ioc_args.R)
        #print(ioc_args.R_del)

    print(ioc_args.S)
    print(ioc_args.Q)
    print(ioc_args.R)
    print(ioc_args.R_del)


x_log = np.array(x_log)
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
plt.xlim(-1,6)
plt.ylim(-1,5)
plt.axis("equal")
#s1 = patches.Circle(xy=args.ev_pos[0], radius=args.d, ec="k")
ax.plot(x_log[:,0],x_log[:,1])
#ax.add_patch(s1)
plt.show()