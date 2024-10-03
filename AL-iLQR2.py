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
    cos_ = jnp.cos(x[2])
    sin_ = jnp.sin(x[2])
    dx = jnp.array([model.r * cos_ * (u[0]+u[1]),
                    model.r * sin_ * (u[0]+u[1]),
                    model.rT * (u[0]-u[1])])
    x_next = x + dx * dt
    return x_next

model_dfdx = jax.jit(jax.jacfwd(model_func_risan,0))
model_dfdu = jax.jit(jax.jacfwd(model_func_risan,1))

## コントローラーに関する関数

@dataclass
class Cont_Args:

    # コントローラーのパラメータ
    Ts = 0.1
    tf = 1.0
    N = int(tf/Ts)
    dt = Ts
    iter = 10
    torelance = 1.0

    # 評価関数中の重み
    Q = 100 * jnp.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=jnp.float32)

    R = jnp.eye(2, dtype=jnp.float32)

    S = 100 * jnp.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=jnp.float32)

    # 目標地点
    x_ob = jnp.array([5, 0, 0], dtype=jnp.float32)

    # 目標入力
    u_ob = jnp.array([0, 0], dtype=jnp.float32)

    # 次元データ
    len_x = 3
    len_u = 2

    # 状態と入力
    x = None
    xs = None
    u = None
    us = None

    # 想定するノイズの平均値
    mean = jnp.zeros((2,),dtype=jnp.float32)
    # 想定するノイズの共分散行列
    cov = 0.1*jnp.eye(2,dtype=jnp.float32)
    # Risk-sensitiveの定数θ
    theta = 1000

    # 入力の制約条件
    umax = jnp.array([15, 15], dtype=jnp.float32)
    umin = jnp.array([-15, -15], dtype=jnp.float32)

    # ラグランジュ乗数λとμ
    mus_u = []
    lambdas_u = []

    # 障害物の場所
    ev_poss = jnp.array([[2.0,-0.1,0]],dtype=jnp.float32)
    # 障害物から取るべき距離
    d = 0.3

    # ラグランジュ乗数λとμ
    mus = []
    lambdas = []

    # 収束寛容度
    tore_const = 0.05
    

args = Cont_Args()


# 制約関数
@jax.jit
def const_x_func(x,ev_pos):
    #val = d - jnp.linalg.norm(jnp.array(x)-jnp.array(ev_pos))
    val = args.d - jnp.sqrt((x[0]-ev_pos[0])**2 + (x[1]-ev_pos[1])**2)
    return jnp.where(val>=0,val,0)


# 制約関数を並列化
const_x_map = jax.vmap(const_x_func,in_axes=(None,0))


# 拡張ラグランジュ法に基づく制約コスト関数
@jax.jit
def const_x_cost(x,k):

    # 状態変数に対する拘束条件
    ev_poss = args.ev_poss
    mu = args.mus[k]
    lambda_ = args.lambdas[k]
    cost_x = const_x_map(x,ev_poss)
    cost_x = lambda_ @ cost_x + 0.5 * cost_x @ jnp.diag(mu) @ cost_x
    
    return cost_x


# 入力上限関数
@jax.jit
def const_u_func(u,umax,umin):
    val = jnp.concatenate([u-umax,umin-u]) 
    return jnp.where(val>=0,val,0)


# 拡張ラグランジュ法に基づく制約コスト関数
@jax.jit
def const_u_cost(u,k):

    # 状態変数に対する拘束条件
    umax = args.umax
    umin = args.umin
    mu = args.mus_u[k]
    lambda_ = args.lambdas_u[k]
    cost_u = const_u_func(u,umax,umin)
    cost_u = lambda_ @ cost_u + 0.5 * cost_u @ jnp.diag(mu) @ cost_u
    
    return cost_u


# ステージコスト
@jax.jit
def stage_cost(x,u,k):
    cost = 0.5 * ( (x-args.x_ob) @ args.Q @ (x-args.x_ob) \
                  + (u-args.u_ob) @ args.R @ (u-args.u_ob)) \
                  + const_x_cost(x,k) \
                  + const_u_cost(u,k)
    return cost


# 終端コスト
@jax.jit
def term_cost(x):
    cost = 0.5 * (x-args.x_ob) @ args.S @ (x-args.x_ob)
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
            J,k = carry
            x, u = val
            J += stage_cost(x,u,k)
            return (J, k-1), _
        
        carry, _ = jax.lax.scan(calcJ_body, (J, args.N-1), (xs[:-1],us))
        J, _ = carry
        J += term_cost(xs[-1])

        J = jnp.float32(J)

        return J
    

    def Backward(xs,us):
        Vxx = hes_x_term(xs[-1])
        Vx = grad_x_term(xs[-1])

        dV1 = 0
        dV2 = 0

        def Backward_body(carry, val):
            # ステージを表す値としてkを追加
            Vx, Vxx, dV1, dV2, k = carry
            x, u = val

            Ak = model_dfdx(x,u,args.dt)
            Bk = model_dfdu(x,u,args.dt)
            Ck = Bk

            ## iLEQGにするにあたって min max 問題のmaxの部分の計算を済ませておく
            #cov = args.cov
            #Vx = Vx + Vx @ Ck @ jnp.linalg.inv( (1/args.theta) * jnp.linalg.pinv(cov) - Ck.T @ Vxx @ Ck ) @ Ck.T @ Vxx
            #Vxx = Vxx + Vxx @ Ck @ jnp.linalg.inv( (1/args.theta) * jnp.linalg.pinv(cov) - Ck.T @ Vxx @ Ck ) @ Ck.T @ Vxx

            Qx = grad_x_stage(x,u,k) + Vx @ Ak
            Qxx = hes_x_stage(x,u,k) + Ak.T @ Vxx @ Ak

            Qu = grad_u_stage(x,u,k) + Vx @ Bk
            Quu = hes_u_stage(x,u,k) + Bk.T @ Vxx @ Bk

            Qux = hes_ux_stage(x,u,k) + Bk.T @ Vxx @ Ak

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

            return (Vx, Vxx, dV1, dV2, k-1), (K, d)
        
        carry, output_val = jax.lax.scan(Backward_body, (Vx, Vxx, dV1, dV2, args.N-1), (jnp.flip(xs[:-1], 0), jnp.flip(us, 0)))
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

    return xs, us


# 拡張ラグランジュ法による制約付きiLQR
def AL_iLEQG(x,xs,us):

    # 乗数のアップデート
    def update_lambda_mu(xs):
        lam_new = []
        for i in range(args.N):
            cost_x = const_x_map(xs[i],args.ev_poss)
            lambda_ = args.lambdas[i]
            mu_ = args.mus[i]
            lam_new_ = jnp.maximum(0,lambda_+mu_*cost_x)
            lam_new.append(lam_new_)

            for j in range(4):
                if lam_new_[j] == 0 and cost_x[j] <= 0:
                    args.mus = args.mus.at[i,j].set(0)
            #    elif args.mus[i][j] == 0:
            #        args.mus = args.mus.at[i,j].set(1)

        return jnp.array(lam_new) #, jnp.array(mu_new)*2
    
    # 乗数のアップデート
    def update_lambda_mu_u(us):
        lam_new = []
        for i in range(args.N):
            cost_u = const_u_func(us[i],args.umax,args.umin)
            lambda_ = args.lambdas_u[i]
            mu_ = args.mus[i]
            lam_new_ = jnp.maximum(0,lambda_+mu_*cost_u)
            lam_new.append(lam_new_)

            for j in range(4):
                if lam_new_[j] == 0 and cost_u[j] <= 0:
                    args.mus_u = args.mus_u.at[i,j].set(0)
            #    elif args.mus[i][j] == 0:
            #        args.mus_u = args.mus_u.at[i,j].set(1)

        return jnp.array(lam_new) #, jnp.array(mu_new)*2
    
    # 収束条件のチェック
    def cons_check(xs):
        hantei = True
        for i in range(args.N):
            cost_x = const_x_map(xs[i],args.ev_poss)
            han = args.tore_const * jnp.ones(cost_x.shape)
            han2 = jnp.maximum(han,cost_x)
            #print(cost_x)
            if not (han==han2).all():
                hantei = False
                break
        
        return hantei
    
    # 収束条件のチェック
    def cons_check_u(us):
        hantei = True
        for i in range(args.N):
            cost_u = const_u_func(us[i],args.umax,args.umin)
            han = args.tore_const * jnp.ones(cost_u.shape)
            han2 = jnp.maximum(han,cost_u)
            #print(cost_u)
            if not (han==han2).all():
                hantei = False
                break
        
        return hantei

    con_dim = len(args.ev_poss)
    con_dim_u = 4
    args.lambdas = 0*jnp.ones((args.N, con_dim),dtype=jnp.float32)
    args.mus = 1*jnp.ones((args.N, con_dim),dtype=jnp.float32)
    args.lambdas_u = 0*jnp.ones((args.N, con_dim_u),dtype=jnp.float32)
    args.mus_u = 1*jnp.ones((args.N, con_dim_u),dtype=jnp.float32)
    #args.lambdas = update_lambda_mu(xs)
    #args.lambdas_u = update_lambda_mu_u(us)

    for i in range(30):
        xs, us = iLQR_control(x,us)
        if cons_check(xs) and cons_check_u(us):
            if i != 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("SHU")
            break
        args.lambdas = update_lambda_mu(xs)
        args.lambdas_u = update_lambda_mu_u(us)
        args.mus = jnp.minimum(1e10,args.mus*1)
        args.mus_u = jnp.minimum(1e10,args.mus_u*1)
    
    return us ,xs



# 初期条件
args.u = jnp.zeros((args.len_u), dtype=jnp.float32)
args.us = jnp.zeros((args.N, args.len_u), dtype=jnp.float32)
args.x = jnp.zeros((args.len_x), dtype=jnp.float32)
args.xs = jnp.zeros((args.N, args.len_x), dtype=jnp.float32)

# ノイズ生成の為のランダムシード
key = jax.random.PRNGKey(42)

# ラグランジュ乗数の設定
con_dim = len(args.ev_poss)
args.lambdas = 0.0*jnp.ones((args.N, con_dim),dtype=jnp.float32)
args.mus = 1.1*jnp.ones((args.N, con_dim),dtype=jnp.float32)

Time = 0
x_log = []

start = time.time()
while Time <= 40:
    print("-------------Position-------------")
    print(args.x)
    print("-------------Input-------------")
    print(args.u)

    x_log.append(args.x)

    x = model_func_risan(args.x,args.u,args.dt)

    us, xs = AL_iLEQG(args.x,args.xs,args.us)

    # 入力に入るノイズを計算する
    key, subkey = jax.random.split(key)
    noi = jax.random.multivariate_normal(subkey,args.mean,args.cov)
    
    Time += args.Ts
    args.x = jnp.array(x)
    args.xs = xs
    args.u = us[0] + noi
    args.us = us

end = time.time()
loop_time = end - start

print("計算時間：{}[s]".format(loop_time))

x_log = np.array(x_log)
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
plt.xlim(-1,6)
plt.ylim(-2,2)
plt.axis("equal")
s1 = patches.Circle(xy=args.ev_poss[0], radius=args.d, ec="k")
ax.plot(x_log[:,0],x_log[:,1])
ax.add_patch(s1)
plt.show()