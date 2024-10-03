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
    tf = 2.0
    N = 20
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
    u = None
    us = None

    # 想定するノイズの平均値
    mean = jnp.zeros((2,),dtype=jnp.float32)
    # 想定するノイズの共分散行列
    cov = 3.0*jnp.eye(2,dtype=jnp.float32)
    # Risk-sensitiveの定数θ
    theta = 1.0

    # 障害物の場所
    ev_pos = jnp.array([[2.5,0.1,0]],dtype=jnp.float32)
    # 障害物から取るべき距離
    d = 0.3

    # 緩和対数バリア関数の緩和値
    del_bar = 0.05
    # 計算用の行列
    E = jnp.array([[1,0,0],
                   [0,1,0],
                   [0,0,0]],dtype=jnp.float32)

    # バリア関数の重み
    r = 30

    # 速度制限
    umax = jnp.array([15,15],dtype=jnp.float32)
    umin = jnp.array([-15,-15],dtype=jnp.float32)

    # 計算用行列
    bar_C = np.concatenate([jnp.eye(len_u,dtype=jnp.float32),-jnp.eye(len_u,dtype=jnp.float32)],0)
    bar_d = np.concatenate([umax,-umin],0)

    # 速度制限の重み
    b = 10


args = Cont_Args()


# バリア関数（-logか二次関数かを勝手に切り替える
@jax.jit
def barrier_z(z):
    pred = z > args.del_bar
    true_fun = lambda z: - jnp.log(z)
    false_fun = lambda z: 0.5 * (((z - 2*args.del_bar) / args.del_bar)**2 - 1) - jnp.log(args.del_bar)
    return jax.lax.cond(pred, true_fun, false_fun, z)


#バリア関数の微分値（B(z)のz微分
@jax.jit
def barrier_dz(z):
    pred = z > args.del_bar
    true_fun = lambda z: -(1/z)
    false_fun = lambda z: (z - 2 * args.del_bar) / args.del_bar
    return jax.lax.cond(pred, true_fun, false_fun, z)


 #バリア関数のu微分（答えは行ベクトルになる）
@jax.jit
def barrier_du(u):
    zs = args.bar_d - args.bar_C @ u

    def vmap_fun(z, bar_C, b):
        dBdz = barrier_dz(z)
        return - b * dBdz * bar_C

    dBdus = jax.vmap(vmap_fun)(zs, args.bar_C, args.b)
    dBdu = jnp.sum(dBdus)
    return dBdu



#バリア関数のuヘッシアン（答えは行列になる）
@jax.jit
def barrier_hes_u(u):
    zs = args.bar_d - args.bar_C @ u

    def vmap_fun(z, bar_C, b):
        dBdzz = barrier_hes_z(z)
        return b * dBdzz * bar_C.T @ bar_C

    dBduus = jax.vmap(vmap_fun)(zs, args.bar_C, args.b)
    dBduu = jnp.sum(dBduus)
    return dBduu


#バリア関数のz二階微分（B(z)のz二階微分）
@jax.jit
def barrier_hes_z(z):
    pred = z > args.del_bar
    true_fun = lambda z: 1/(z**2)
    false_fun = lambda z: 1/args.del_bar
    return jax.lax.cond(pred, true_fun, false_fun, z)


#対数バリア関数型回避関数の微分
@jax.jit
def evasion_dx(x):

    def vmap_fun(xe, r, e):
        z = (x - xe) @ args.E @ (x - xe) - r**2
        dzdx = 2 * (x - xe) @ args.E
        dedz = barrier_dz(z)
        return e * dzdx * dedz

    dedxs = jax.vmap(vmap_fun,(0,None,None))(args.ev_pos, args.d, args.r)
    dedx = jnp.sum(dedxs)

    return dedx


#対数バリア関数型回避関数のヘッシアン
@jax.jit
def evasion_hes_x(x):
    def vmap_fun(xe, r, e):
        z = (x - xe) @ args.E @ (x - xe) - r**2
        dzdx = 2 * (x - xe) @ args.E
        dzdxx = 2 * args.E
        dedz = barrier_dz(z)
        dedzz = barrier_hes_z(z)
        return e * (dedz * dzdxx + dzdx.T @ (dedzz * dzdx))

    dedxxs = jax.vmap(vmap_fun, (None,None,None))(args.ev_pos, args.d, args.r)
    dedxx = jnp.sum(dedxxs)
    return dedxx


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
    cost = 0.5 * ( (x-args.x_ob) @ args.Q @ (x-args.x_ob) \
                  + (u-args.u_ob) @ args.R @ (u-args.u_ob)) \
                  + evasion(x) \
                  + barrier(u)
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
def iLEQG_control(x,us):

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
            Ck = Bk

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
    

    def iLEQG_body(val):
        xs, us, J_old, _, iteration = val

        Ks, ds, dV1, dV2 = Backward(xs,us)
        xs, us, J_new = Forward(xs,us,Ks,ds,dV1,dV2,J_old)

        iteration += 1 #反復回数の更新

        return (xs, us, J_new, J_old, iteration)

    
    # iLQRループ全体

    xs = rollout(x,us)
    J = calcJ(xs,us)
    iteration = 0 #LQRステップの繰り返し数をリセット

    val = jax.lax.while_loop(conv_check, iLEQG_body, (xs,us,J,J+1e5,iteration)) #whileループが最初で止まらない様に1e5の差をつける
    xs, us, _, _, _ = val

    return us



# 初期条件
args.u = jnp.zeros((args.len_u), dtype=jnp.float32)
args.us = jnp.zeros((args.N, args.len_u), dtype=jnp.float32)
args.x = jnp.zeros((args.len_x), dtype=jnp.float32)

# ノイズ生成の為のランダムシード
key = jax.random.PRNGKey(42)

Time = 0
x_log = []

start = time.time()
while Time <= 20:
    print("-------------Position-------------")
    print(args.x)
    print("-------------Input-------------")
    print(args.u)

    x_log.append(args.x)

    x = model_func_risan(args.x,args.u,args.dt)

    us = iLEQG_control(args.x,args.us)

    # 入力に入るノイズを計算する
    key, subkey = jax.random.split(key)
    noi = jax.random.multivariate_normal(subkey,args.mean,args.cov)
    
    Time += args.Ts
    args.x = x
    args.u = us[0] #+ noi
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
s1 = patches.Circle(xy=args.ev_pos[0], radius=args.d, ec="k")
ax.plot(x_log[:,0],x_log[:,1])
ax.add_patch(s1)
plt.show()