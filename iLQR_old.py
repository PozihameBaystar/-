import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# シミュレーションの基準となる時刻
Ts = 0.05
t_sim_end = 20
tsim = np.arange(0, t_sim_end + Ts, Ts)

# コントローラーのパラメーター
iLQR = {
    'Ts': Ts,
    'tf': 1.0,
    'N': 20,
    'iter': 100,
    'dt': 1.0 / 20,  # 1 / N
    'torelance': 1,
    'Q': 100 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'R': np.eye(2),
    'S': 100 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
}

# 車のパラメーター
car = {
    'R': 0.05,
    'T': 0.2,
    'r': 0.05 / 2,
    'rT': (0.05 / 2) / 0.2,
}

# 初期条件
car['x'] = np.array([0, 0, 0])
iLQR['u'] = np.array([0, 0])

iLQR['len_x'] = len(car['x'])
iLQR['len_u'] = len(iLQR['u'])

iLQR['U'] = np.zeros((iLQR['len_u'], iLQR['N']))

# 目標地点
iLQR['x_ob'] = np.array([3, 2, 0])

# 拘束条件
iLQR['umax'] = np.array([15, 15])
iLQR['umin'] = np.array([-15, -15])

# シミュレーション(odeint)の設定
opts = {'rtol': 1e-6, 'atol': 1e-8}


# iLQRコントローラー
def iLQR_control(iLQR, car):
    U = iLQR['U']  # 前ステップの入力を初期解として使う
    X = Predict(car['x'], U, iLQR, car)  # 状態変数の将来値を入力初期値から予測
    J = CalcJ(X, U, iLQR)  # 評価関数の初期値を計算

    loop = 0
    while True:
        K, d, dV1, dV2 = Backward(X, U, iLQR, car)  # Backward Passの計算

        X, U, J_new = Forward(X, U, K, d, dV1, dV2, J, iLQR, car)  # Forward Passの計算

        loop += 1

        if abs(J_new - J) <= iLQR['torelance']:  # 評価関数値が収束して来たら
            break

        if loop == iLQR['iter']:  # 繰り返し回数が限界に来たら
            break

        J = J_new

    return U


# 状態変数の初期予測関数
def Predict(x, U, iLQR, car):
    xk = x
    xk = func_risan(xk, U[:, 0], iLQR['dt'], car)
    X = np.zeros((iLQR['len_x'], iLQR['N'] + 1))
    X[:, 0] = xk
    for i in range(1, iLQR['N'] + 1):
        xk = func_risan(xk, U[:, i - 1], iLQR['dt'], car)
        X[:, i] = xk
    return X


# 評価関数の計算
def CalcJ(X, U, iLQR):
    # 終端コストの計算
    phi = np.dot((X[:, -1] - iLQR['x_ob']).T, np.dot(iLQR['S'], (X[:, -1] - iLQR['x_ob'])))

    # 途中のコストの計算
    L = 0
    for i in range(iLQR['N']):
        L += np.dot((X[:, i] - iLQR['x_ob']).T, np.dot(iLQR['Q'], (X[:, i] - iLQR['x_ob']))) + \
             np.dot(U[:, i].T, np.dot(iLQR['R'], U[:, i]))
    L *= iLQR['dt']  # 最後にまとめてdtをかける
    J = phi + L
    return J


# Backward Pass
def Backward(X, U, iLQR, car):
    sk = np.dot((X[:, -1] - iLQR['x_ob']).T, iLQR['S'])  # Vxの初期値
    Sk = iLQR['S']  # Vxxの初期値
    Qk = iLQR['Q']
    Rk = iLQR['R']

    K = np.zeros((iLQR['len_u'], iLQR['len_x'], iLQR['N']))
    d = np.zeros((iLQR['len_u'], iLQR['N']))
    dV1 = 0  # dが1次の項を入れる変数
    dV2 = 0  # dが2次の項を入れる変数

    for i in range(iLQR['N'] - 1, -1, -1):
        x = X[:, i]
        u = U[:, i]
        Ak = CalcA(x, u, iLQR['dt'], car)
        Bk = CalcB(x, iLQR['dt'], car)
        Qx = (np.dot((x - iLQR['x_ob']).T, Qk) * iLQR['dt'] + np.dot(sk, Ak)).T
        Qxx = (Qk * iLQR['dt'] + np.dot(Ak.T, np.dot(Sk, Ak))).T
        if iLQR['umin'][0] < u[0] < iLQR['umax'][0] and iLQR['umin'][1] < u[1] < iLQR['umax'][1]:
            Qu = (Rk[0, 0] * u[0] ** 2 + Rk[1, 1] * u[1] ** 2) * iLQR['dt'] + sk[0] * Bk[0] + 0.15 * np.array([
        (2 * u[0] - iLQR['umax'][0] - iLQR['umin'][0]) / ((u[0] - iLQR['umin'][0]) * (iLQR['umax'][0] - u[0])),
        (2 * u[1] - iLQR['umax'][1] - iLQR['umin'][1]) / ((u[1] - iLQR['umin'][1]) * (iLQR['umax'][1] - u[1]))
    ]) * iLQR['dt']
            Quu = (Rk * iLQR['dt'] + np.dot(Bk.T, np.dot(Sk, Bk)) +
          0.15 * np.array([[2 / ((iLQR['umax'][0] - u[0]) * (u[0] - iLQR['umin'][0])), 0],
                           [0, 2 / ((iLQR['umax'][1] - u[1]) * (u[1] - iLQR['umin'][1]))]])) * iLQR['dt']
        else:
            Qu = Qu = (Rk[0, 0] * u[0] ** 2 + Rk[1, 1] * u[1] ** 2) * iLQR['dt'] + sk[0] * Bk[0] + 10 * np.array([np.sign(u[0]), np.sign(u[1])]) * iLQR['dt']
            Quu = (Rk * iLQR['dt'] + np.dot(Bk.T, np.dot(Sk, Bk)) +
          10 * np.array([[1 / np.sign(u[0]), 0],
                         [0, 1 / np.sign(u[1])]])) * iLQR['dt']
        Qux = np.dot(Bk.T, np.dot(Sk, Ak))
        K_ = -np.linalg.inv(Quu).dot(Qux)  # 閉ループゲインの計算
        K[:, :, i] = K_
        d_ = -np.linalg.inv(Quu).dot(Qu.T)  # 開ループフィードバックの計算
        d[:, i] = d_
        sk = Qx + d_.T.dot(Quu).dot(K_) + Qu.dot(K_) + d_.T.dot(Qux)  # Vxの更新
        Sk = Qxx + K_.T.dot(Quu).dot(K_) + K_.T.dot(Qux) + Qux.T.dot(K_)  # Vxxの更新
        dV1 += Qu.dot(d_)
        dV2 += (1 / 2) * d_.T.dot(Quu).dot(d_)
    return K, d, dV1, dV2


# Forward
def Forward(X, U, K, d, dV1, dV2, J, iLQR, car):
    alpha = 1  # 直線探索の係数を初期化

    X_ = np.zeros((iLQR['len_x'], iLQR['N'] + 1))  # 新しいxの値を入れていく変数
    X_[:, 0] = X[:, 0]  # xの初期値は変化しない

    U_ = np.zeros((iLQR['len_u'], iLQR['N']))  # 新しいuの値を入れていく変数

    while True:
        for i in range(iLQR['N']):
            U_[:, i] = U[:, i] + K[:, :, i].dot(X_[:, i] - X[:, i]) + alpha * d[:, i]
            X_[:, i + 1] = func_risan(X_[:, i], U_[:, i], iLQR['dt'], car)

        J_new = CalcJ(X_, U_, iLQR)
        dV1_ = alpha * dV1
        dV2_ = (alpha ** 2) * dV2
        z = (J_new - J) / (dV1_ + dV2_)

        if 1e-4 <= z <= 10:  # 直線探索が条件を満たしていれば
            J = J_new
            U = U_
            X = X_
            break

        alpha = (1 / 2) * alpha

    return X, U, J


# Akの計算
def CalcA(x, u, dt, car):
    cos_ = np.cos(x[2])
    sin_ = np.sin(x[2])
    Ak = np.eye(3) + np.array([[0, 0, -car['r'] * sin_ * (u[0] + u[1])],
                                [0, 0, car['r'] * cos_ * (u[0] + u[1])],
                                [0, 0, 0]]) * dt
    return Ak


# Bkの計算
def CalcB(x, dt, car):
    cos_ = np.cos(x[2])
    sin_ = np.sin(x[2])
    Bk = np.array([[car['r'] * cos_, car['r'] * cos_],
                   [car['r'] * sin_, car['r'] * sin_],
                   [car['rT'], -car['rT']]]) * dt
    return Bk


# 差分駆動型二輪車のモデル(ode用)
def two_wheel_car(xi, t, u, car):
    dxi = np.zeros(3)  # dxiの型を定義
    r = car['r']
    rT = car['rT']
    cos_ = np.cos(xi[2])
    sin_ = np.sin(xi[2])
    dxi[0] = r * cos_ * (u[0] + u[1])
    dxi[1] = r * sin_ * (u[0] + u[1])
    dxi[2] = rT * (u[0] - u[1])
    return dxi


# 二輪車の離散状態方程式
def func_risan(xk, u, dt, car):
    xk1 = np.zeros(3)  # xk1の型を定義
    r = car['r']
    rT = car['rT']
    cos_ = np.cos(xk[2])
    sin_ = np.sin(xk[2])
    xk1[0] = xk[0] + (r * cos_ * (u[0] + u[1])) * dt
    xk1[1] = xk[1] + (r * sin_ * (u[0] + u[1])) * dt
    xk1[2] = xk[2] + (rT * (u[0] - u[1])) * dt
    return xk1



if __name__ == "__main__":
    
    # シミュレーションループ
    log = {'u': np.zeros((iLQR['len_u'] * iLQR['N'], len(tsim))),
        'x': np.zeros((iLQR['len_x'], len(tsim)))}

    for i in range(len(tsim)):
        log['u'][:, i] = iLQR['U'].reshape(iLQR['len_u'] * iLQR['N'])
        log['x'][:, i] = car['x']
        print(iLQR['u'])
        print(car['x'])

        # シミュレーション計算（最後だけは計算しない）
        if i != len(tsim) - 1:
            t_eval = [tsim[i], tsim[i + 1]]
            xi = odeint(lambda xi, t: two_wheel_car(t, xi, iLQR['u'], car), car['x'], t_eval, **opts, tfirst=True)
        else:
            break

        # iLQR法コントローラーを関数として実装
        U = iLQR_control(iLQR, car)
        iLQR['u'] = U[:, 0]
        iLQR['U'] = U

        car['x'] = xi[-1,:]

    # グラフ化
    fig, axs = plt.subplots(5, 1, figsize=(7, 10))

    for i in range(5):
        axs[i].grid(True)
        axs[i].tick_params(labelsize=9.5)

    axs[0].plot(tsim, log['x'][0, :], linewidth=1)
    axs[0].set_ylabel('x1')
    axs[0].set_xlabel('Time[s]')
    axs[0].set_ylim([0, 3])

    axs[1].plot(tsim, log['x'][1, :], linewidth=1)
    axs[1].set_ylabel('x2')
    axs[1].set_xlabel('Time[s]')
    axs[1].set_ylim([0, 2])

    axs[2].plot(tsim, log['x'][2, :], linewidth=1)
    axs[2].set_ylabel('Φ')
    axs[2].set_xlabel('Time[s]')

    axs[3].plot(tsim, log['u'][0, :], linewidth=1)
    axs[3].set_ylabel('u1')
    axs[3].set_xlabel('Time[s]')

    axs[4].plot(tsim, log['u'][1, :], linewidth=1)
    axs[4].set_ylabel('u2')
    axs[4].set_xlabel('Time[s]')

    plt.tight_layout()
    plt.show()