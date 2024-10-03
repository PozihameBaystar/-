import jax
import jax.numpy as jnp
import numpy as np
import casadi as ca
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
    tf = 3.0
    N = 30
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
    x_ob = jnp.array([2, 1, 0], dtype=jnp.float32)

    # 目標入力
    u_ob = jnp.array([0, 0], dtype=jnp.float32)

    # 次元データ
    len_x = 5
    len_u = 2

    # 状態と入力
    x = None
    u = None
    us = None

    # 想定するノイズの平均値
    mean = jnp.zeros((2,),dtype=jnp.float32)
    # 想定するノイズの共分散行列
    cov = 0.1*jnp.eye(2,dtype=jnp.float32)
    # Risk-sensitiveの定数θ
    theta = 1.0

    # 障害物の場所
    ev_pos = jnp.array([[2.0,0.15,0]],dtype=jnp.float32)
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
    umax = jnp.array([0.5,1.0],dtype=jnp.float32)
    umin = jnp.array([-0.5,-1.0],dtype=jnp.float32)

    # 計算用行列
    bar_C = np.concatenate([jnp.eye(len_u,dtype=jnp.float32),-jnp.eye(len_u,dtype=jnp.float32)],0)
    bar_d = np.concatenate([umax,-umin],0)

    # 速度制限の重み
    b = 0

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

    def vmap_fun(b, z, margin=0.3):
        return b * jnp.where(z>=margin,barrier_z(margin),barrier_z(z))

    Bars = jax.vmap(vmap_fun, (None,0))(args.b, zs)
    Bar = jnp.sum(Bars)
    return Bar


# 対数バリア関数型回避関数
@jax.jit
def evasion(x):

    def vmap_fun(x, xe, r, d, margin=0.3):
        distance = jnp.linalg.norm(x-xe,ord=2)
        z = distance**2 - d**2
        kijun = d + margin
        return r * jnp.where(distance>=kijun, barrier_z(kijun**2-d**2), barrier_z(z))

    evas = jax.vmap(vmap_fun, (None, 0, None, None))(x, args.ev_pos, args.r, args.d)
    eva = jnp.sum(evas)
    return eva


# ステージコスト
@jax.jit
def stage_cost(x,u):
    cost = 0.5 * ( (x[:3]-args.x_ob) @ args.Q @ (x[:3]-args.x_ob) \
                  + (x[3:5]+u-args.u_ob) @ args.R @ (x[3:5]+u-args.u_ob) \
                  + u @ args.R_del @ u ) \
                  + evasion(x[0:3]) \
                  + barrier(x[3:5])
    return cost

# 終端コスト
@jax.jit
def term_cost(x):
    cost = 0.5 * (x[:3]-args.x_ob) @ args.S @ (x[:3]-args.x_ob)
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

        Q = args.Q
        R = args.R

        dV1 = 0
        dV2 = 0

        def Backward_body(carry, val):
            Vx, Vxx, dV1, dV2 = carry
            x, u = val

            Ak = model_dfdx(x,u,args.dt)
            Bk = model_dfdu(x,u,args.dt)

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

    return us, xs


########### ここからIOC

## パラメーター
@dataclass
class ioc_args:
    # 評価関数関連
    S = jnp.array([1,1,0],dtype=jnp.float32) * 100
    Q = jnp.array([1,1,0],dtype=jnp.float32) * 100
    R = jnp.ones((2,),dtype=jnp.float32)  # 速度の重み
    R_del = jnp.ones((2,),dtype=jnp.float32)  # 速度変化の重み
    D = 1.0  # 目的方向への余弦の重み
    r = 1.0  # 回避関数の重み
    b = 1.0  # 速度制限関数の重み
    x_ob = jnp.array([2,1,0],dtype=jnp.float32)
    ev_pos = jnp.array([2,0.15,0],dtype=jnp.float32)
    u_ob = jnp.array([0,0],dtype=jnp.float32)
    alpha = 1000 # 逆最適制御の尖り具合
    sigma = 1.0 # カーネル関数の分散
    kyori = [0.45, 1.2, 3.6]
    weights = jnp.array([],dtype=jnp.float32)
    obs_dim = 5
    action_dim = 2
    # コントローラーのその他のパラメータ
    Ts = 0.1
    tf = 3.0
    N = int(tf/Ts)
    dt = Ts


ioc_args = ioc_args()


# ベースとなるコスト関数
# 二次形式関数
def base_cost_quad(weight,x):
    return x.T @ jnp.diag(weight) @ x

# ベース関数としての対数バリア関数型回避関数
def base_evasion(weight,x):

    def vmap_fun(x, xe, r, d, margin=0.3):
        distance = jnp.linalg.norm(x-xe,ord=2)
        z = distance**2 - d**2
        kijun = d + margin
        return r * jnp.where(distance>=kijun, barrier_z(kijun**2-d**2), barrier_z(z))

    evas = jax.vmap(vmap_fun, (None, 0, None, None))(x, ioc_args.ev_pos, weight, args.d)
    eva = jnp.sum(evas)
    return eva

# ベース関数としてのバリア関数
def base_barrier(weight,u):

    zs = args.bar_d - args.bar_C @ u

    def vmap_fun(b, z, margin=0.3):
        return b * jnp.where(z>=margin,barrier_z(margin),barrier_z(z))

    Bars = jax.vmap(vmap_fun, (None,0))(weight, zs)
    Bar = jnp.sum(Bars)
    return Bar


# コスト関数全体
@jax.jit
def cost_ioc(S,Q,R,Rd,r,b,xs,us):

    def cost_map(x,u):
        return base_cost_quad(Q,x[:3]) + base_cost_quad(R,x[3:5]) + base_cost_quad(Rd,u) + base_barrier(b,x[3:5]) + base_evasion(r,x[:3])
    
    #def const_map(lambda_,x,x_next,u):
    #    return lambda_ * (x_next - model_func_risan(x,u,ioc_args.dt))
    
    xs_ = jnp.reshape(xs,(-1,ioc_args.obs_dim))
    us_ = jnp.reshape(us,(-1,ioc_args.action_dim))
    cost = jnp.sum( jax.vmap(cost_map, (0,0))(xs_[:-1],us_) )
    cost += base_cost_quad(S,xs_[-1,:3])
    #cost += jnp.sum( jax.vmap(const_map, (0,0,0,0))(lambdas,xs_[:-1],xs_[1:],us_) )
    return cost


# コスト関数の重み微分
cost_grad = jax.jit(jax.grad(cost_ioc,[0,1,2,3,4,5]))
cost_hesx = jax.jit(jax.jacrev(cost_grad,6))
cost_hesu = jax.jit(jax.jacrev(cost_grad,7))
#cost_hesl = jax.jit(jax.jacrev(cost_grad,8))


# コスト関数のヘッシアン計算
@jax.jit
def cost_hes(S,Q,R,Rd,r,b,xs,us):

    def vmap_hes(hes):
        return jnp.vstack([hes[0],hes[1],hes[2],hes[3],hes[4],hes[5]])

    hes1 = cost_hesx(S,Q,R,Rd,r,b,xs,us)
    hes2 = cost_hesu(S,Q,R,Rd,r,b,xs,us)
    #hes3 = cost_hesl(S,Q,R,Rd,r,b,xs,us,lambdas)

    #hes = [hes1,hes2]

    #hess = jax.vmap(vmap_hes,(0))(hes)
    hes1 = vmap_hes(hes1)
    hes2 = vmap_hes(hes2)
    hes = jnp.hstack([hes1,hes2])

    return hes.T


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
        obss = np.empty((horizon+1, batch_size, obs_dim), dtype=np.float32)
        #obss = np.empty((horizon, batch_size, obs_dim), dtype=np.float32)
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
    xs_log = []
    us_log = []


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

        us,xs = iLQR_control(args.x,args.us)
        
        Time += args.Ts
        args.x = x
        args.u = us[0]
        args.us = us

        xs_log.append(xs)
        us_log.append(us)

    x_log.append(args.x)

    rb.add(x_log, u_log)



    # 学習ループ

    #lr = 0.0000001 # 学習率

    for loop in range(1):

        bat_xs, bat_us = rb.sample(1,ioc_args.N)

        for bat in range(1):
            xs = bat_xs[:,bat]
            us = bat_us[:,bat]
            xs = jnp.reshape(xs,(-1,))
            us = jnp.reshape(us,(-1,))

            #lambdas = jnp.ones((ioc_args.N),dtype=jnp.float32)

            A = cost_hes(ioc_args.S,
                         ioc_args.Q,
                         ioc_args.R,
                         ioc_args.R_del,
                         ioc_args.r,
                         ioc_args.b,
                         xs,
                         us)
            
            A = np.asarray(A)
            A_ca = ca.DM(A)

            w = ca.SX.sym("w",12)

            ioc = {
                "x":w,
                "f":w.T @ A_ca.T @ A_ca @ w,
                "g":ca.vertcat(
                    ca.sum1(w**2),
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                    w[4],
                    w[5],
                    w[6],
                    w[7],
                    w[8],
                    w[9],
                    w[10],
                    w[11]
                )
                }
            
            S = ca.qpsol("S","osqp",ioc)
            r = S(x0=[100,100,0,100,100,0,100,100,100,100,0,0],
                  lbg=[1,0,0,0,0,0,0,0,0,0,0,0,0],
                  ubg=[1,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
            w_opt = np.array(r["x"])
            w_opt.reshape((12,))
            print(A @ w_opt)

    x_log = np.array(x_log)
    fig = plt.figure(figsize=(7, 10))
    ax = plt.axes()
    plt.xlim(-1,6)
    plt.ylim(-2,2)
    plt.axis("equal")
    s1 = patches.Circle(xy=args.ev_pos[0], radius=args.d, ec="k")
    ax.plot(x_log[:,0],x_log[:,1])
    ax.add_patch(s1)

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)

    ax1.plot(x_log[:,3])
    ax2.plot(x_log[:,4])

    plt.show()