import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#移動ロボットクラス
class car:
    def __init__(self,x_st): #引数は初期位置
        self.R = 0.05
        self.T = 0.2
        self.r = self.R/2
        self.rT = self.R/self.T

        self.x = x_st #最初は初期位置をセット

        self.len_u = 2
        self.len_x = 3

        self.umax = np.array([[15],
                              [15]],dtype=float)
        self.umin = np.array([[-15],
                              [-15]],dtype=float)

    #ダイナミクス（状態方程式）
    def func(self,x,u):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        dx = np.array([[self.r*cos_*(u[0][0]+u[1][0])],
                       [self.r*sin_*(u[0][0]+u[1][0])],
                       [self.rT*(u[0][0]-u[1][0])]])
        return dx
    
    #離散化状態方程式
    def func_risan(self,x,u,dt):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        dx = np.array([[self.r*cos_*(u[0][0]+u[1][0])],
                       [self.r*sin_*(u[0][0]+u[1][0])],
                       [self.rT*(u[0][0]-u[1][0])]])
        x_next = x + dx*dt #ここでx += としてしまうと引数のxを変えてしまう
        return x_next
    
    #df/dx（離散）
    def CalcA_risan(self,x,u,dt):
        Ak = np.eye(3) + \
            np.array([[0, 0, -self.r*math.sin(x[2][0])*(u[0][0]+u[1][0])],
                      [0, 0, self.r*math.cos(x[2][0])*(u[0][0]+u[1][0])],
                      [0, 0, 0]])*dt
        return Ak
    
    #df/du（離散）
    def CalcB_risan(self,x,dt):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        Bk = np.array([[self.r*cos_, self.r*cos_],
                       [self.r*sin_, self.r*sin_],
                       [self.rT, -self.rT]])*dt
        return Bk


#コントローラークラス
class iLQR_controller:
    def __init__(self,model,x_ob): #引数は制御対象と目標姿勢
        #コントローラーのパラメータ
        self.Ts = 0.1
        self.tf = 1.0
        self.N = 10
        self.iter = 20
        self.dt = self.tf/self.N
        self.torelance = 1.0

        #操縦するモデルをセット
        self.model = model

        #入力の次元
        self.len_u = self.model.len_u
        self.len_x = self.model.len_x

        #評価関数中の重み
        self.Q = 100 * np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]],dtype=float)
        self.R = np.eye(2,dtype=float)
        self.S = 100 * np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]],dtype=float)
        
        #初期条件
        self.u = np.zeros((self.len_u,1),dtype=float)
        self.U = np.zeros((self.len_u,self.N),dtype=float)

        #目標地点
        self.x_ob = x_ob.reshape((self.len_x,1))

        #拘束条件
        self.umax = self.model.umax
        self.umin = self.model.umin

        #バリア関数用パラメータ
        self.b = 10 * np.ones((2*self.len_u)) #重み
        self.del_bar = 0.1
        self.bar_C = np.concatenate([np.eye(self.len_u),-np.eye(self.len_u)],0)
        self.bar_d = np.concatenate([self.umax,-self.umin],0)
        self.con_dim = len(self.b)

        #回避関数用パラメータ
        self.r = 0.2
        self.xe = np.array([[2.5, 0.1, 0],
                            [5, -0.4, 0],
                            [5.4, 0, 0]]).T #回避する障害物の座標を横ベクトルで入れて行く（転置するのを忘れずに）
        self.eva_dim = len(self.xe.T)
        #self.e = 0 * np.ones(self.eva_dim) #重み
        self.e = np.array([150,0,0])
        self.E = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]) #計算用の変数

    
    def iLQR_control(self):
        U = self.U #前のステップの解を初期解として利用
        X = self.Predict(U) #将来の状態変数を予測
        J = self.CalcJ(X,U) #評価関数の初期値を計算

        loop = 0 #繰り返し回数

        while True:
            K, d, dV1, dV2 = self.Backward(X,U)
            X, U, J_new = self.Forward(X,U,K,d,dV1,dV2,J)

            loop += 1
            
            if abs(J_new-J) <= self.torelance:
                break
            
            if loop == self.iter:
                break

            J = J_new
        
        self.u = U[:,0:1]
        self.U = U

        return

    
    def Predict(self,U):
        xk = self.model.x #現在の状態変数を持ってくる
        xk = self.model.func_risan(xk,U[:,0:1],self.dt) #次の制御の時の状態変数x0を計算
        X = np.zeros((self.len_x,self.N+1),dtype=float) #状態変数の予測値を入れる変数を設定
        X[:,0:1] = xk #初期値を最初に入れる

        for i in range(self.N):
            xk = self.model.func_risan(xk,U[:,i:i+1],self.dt) #次のステップを計算
            X[:,i+1:i+2] = xk #状態変数を入れる

        return X

    
    def CalcJ(self,X,U):
        #終端コストの計算
        phi = np.dot((X[:,-1:]-self.x_ob).T,np.dot(self.S,X[:,-1:]-self.x_ob))[0][0]

        #ステージコストの計算
        L = 0 #ステージコストの値を入れる変数
        
        for i in range(self.N):
            bar_val = self.barrier(U[:,i:i+1])
            eva_val = self.evasion(X[:,i:i+1])
            L += 0.5*np.dot((X[:,i:i+1] - self.x_ob).T,np.dot(self.Q,X[:,i:i+1]-self.x_ob))[0][0]\
                + 0.5*np.dot(U[:,i:i+1].T,np.dot(self.R,U[:,i:i+1]))[0][0]\
                + bar_val\
                + eva_val

        L = L * self.dt #最後にまとめて掛ける
        J = phi + L
        return J
    

    def Backward(self,X,U):
        sk = np.dot((X[:,-1:] - self.x_ob).T,self.S) #Vxの初期値
        Sk = self.S #Vxxの初期値
        Qk = self.Q
        Rk = self.R

        K = np.zeros((self.N,self.len_u,self.len_x),dtype=float)
        d = np.zeros((self.len_u,self.N),dtype=float)
        dV1 = 0 #dが一次の項を入れる変数
        dV2 = 0 #dが二次の項を入れる変数

        for i in reversed(range(self.N)):
            x = X[:,i:i+1]
            u = U[:,i:i+1]
            Ak = self.model.CalcA_risan(x,u,self.dt)
            Bk = self.model.CalcB_risan(x,self.dt)

            dEdx = self.evasion_dx(x)
            dEdxx = self.evasion_hes_x(x)

            Qx = np.dot((x-self.x_ob).T,Qk)*self.dt + np.dot(sk,Ak) + dEdx*self.dt
            Qxx = Qk*self.dt + np.dot(Ak.T,np.dot(Sk,Ak)) + dEdxx*self.dt

            dBdu = self.barrier_du(u)
            dBduu = self.barrier_hes_u(u)

            Qu = np.dot(u.T,Rk)*self.dt + np.dot(sk,Bk) + dBdu*self.dt
            Quu = Rk*self.dt + np.dot(Bk.T,np.dot(Sk,Bk)) + dBduu*self.dt

            #Quuが正定かどうかの判定
            try:
                kekka = np.linalg.cholesky(Quu)
            except:
                #もし違ったら
                #正定化の為にまず固有値の最小値を特定する
                alpa = -np.amin(np.linalg.eig(Quu))
                Quu = Quu + alpa * np.eye(self.len_u) #半正定化

            Qux = np.dot(Bk.T,np.dot(Sk,Ak))
            K_ = -np.dot(np.linalg.pinv(Quu),Qux) #閉ループゲインの計算
            #K_ = np.linalg.solve(Quu,-Qux)
            K[i,:,:] = K_
            d_ = -np.dot(np.linalg.pinv(Quu),Qu.T) #開ループゲインの計算
            #d_ = np.linalg.solve(Quu,-Qu.T)
            d[:,i:i+1] = d_
            sk = Qx + np.dot(d_.T,np.dot(Quu,K_)) + np.dot(Qu,K_) + np.dot(d_.T,Qux) #Vxの更新
            Sk = Qxx + np.dot(K_.T,np.dot(Quu,K_)) + np.dot(K_.T,Qux) + np.dot(Qux.T,K_) #Vxxの更新
            dV1 += np.dot(Qu,d_)[0][0] #値だけを取り出す
            dV2 += 0.5 * np.dot(d_.T,np.dot(Quu,d_))[0][0] #値だけを取り出す

        return K, d, dV1, dV2
    

    def Forward(self,X,U,K,d,dV1,dV2,J):
        alpha = 1.0 #直線探索の係数を初期化

        X_ = np.zeros((self.len_x,self.N+1),dtype=float) #新しいXを入れる変数を作成
        X_[:,0:1] = X[:,0:1] #最初の値だけは変わらない

        U_ = np.zeros((self.len_u,self.N),dtype=float) #新しいUを入れる変数

        #直線探索を終わらせるためのカウント
        count = 0

        #評価関数の最少記録とその周辺
        J_min = J
        U_min = U
        X_min = X

        while True:
            for i in range(self.N):
                U_[:,i:i+1] = U[:,i:i+1] + np.dot(K[i,:,:],X_[:,i:i+1]-X[:,i:i+1]) + alpha*d[:,i:i+1]
                X_[:,i+1:i+2] = self.model.func_risan(X_[:,i:i+1],U_[:,i:i+1],self.dt)

            J_new = self.CalcJ(X_,U_)
            dV1_ = alpha * dV1
            dV2_ = (alpha**2) * dV2
            z = (J_new - J)/(dV1_+dV2_)

            if 1e-4 <= z and z <= 10: #直線探索の条件を満たしたら
                J = J_new
                U = U_
                X = X_
                break #whileループも終了

            #評価関数の最小値とその時の計算結果を記録
            if J_min > J_new:
                J_min = J_new
                U_min = U_
                X_min = X_

            #直線探索回数が限界に来たら、評価関数が最小の物を使う
            if count == 10:
                J = J_min
                U = U_min
                X = X_min
                break

            alpha = alpha * 0.5 #直線探索の係数を更新

            count = count +1

        return X, U, J
    
    
    #バリア関数（全体）
    def barrier(self, u):
        zs = self.bar_d - np.dot(self.bar_C,u)

        Bar = 0

        for i in range(self.con_dim):
            Bar += self.b[i]*self.barrier_z(zs[i][0])

        return Bar
    
    #バリア関数のu微分（答えは行ベクトルになる）
    def barrier_du(self, u):
        zs = self.bar_d - np.dot(self.bar_C,u)

        dBdu = np.zeros((1,self.len_u))

        for i in range(self.con_dim):
            dBdz = self.barrier_dz(zs[i][0])
            dBdu -= self.b[i] * dBdz * self.bar_C[i:i+1,:]

        return dBdu
    
    #バリア関数のuヘッシアン（答えは行列になる）
    def barrier_hes_u(self, u):
        zs = self.bar_d - np.dot(self.bar_C,u)

        dBduu = np.zeros((self.len_u,self.len_u))

        for i in range(self.con_dim):
            dBdzz = self.barrier_hes_z(zs[i][0])
            dBduu += self.b[i] * dBdzz * np.dot(self.bar_C[i:i+1,:].T,self.bar_C[i:i+1,:])

        return dBduu
    
    #バリア関数（-logか二次関数かを勝手に切り替える）
    def barrier_z(self, z):
        if z > self.del_bar:
            value = -math.log(z)
        else:
            value = (1/2)*( ((z-2*self.del_bar)/self.del_bar)**2 - 1 ) - math.log(self.del_bar)

        return value
    
    #バリア関数の微分値（B(z)のz微分）
    def barrier_dz(self, z):
        if z > self.del_bar:
            value = -(1/z)
        else:
            value = (z-2*self.del_bar)/self.del_bar

        return value
    
    #バリア関数のz二階微分（B(z)のz二階微分）
    def barrier_hes_z(self, z):
        if z > self.del_bar:
            value = 1/(z**2)
        else:
            value = 1/self.del_bar
            
        return value
    
    #対数バリア関数型回避関数
    def evasion(self,x):
        eva = 0
        for i in range(self.eva_dim):
            z = np.dot((x-self.xe[:,i:i+1]).T, np.dot(self.E,(x-self.xe[:,i:i+1]))) - self.r**2
            eva += self.e[i] * self.barrier_z(z)

        return eva

    #対数バリア関数型回避関数の微分
    def evasion_dx(self,x):
        dedx = np.zeros((1,self.len_x))
        for i in range(self.eva_dim):
            z = np.dot((x-self.xe[:,i:i+1]).T, np.dot(self.E,(x-self.xe[:,i:i+1]))) - self.r**2
            dzdx = 2 * np.dot((x-self.xe[:,i:i+1]).T, self.E)
            dedz = self.barrier_dz(z)
            dedx += self.e[i] * dzdx *dedz

        return dedx
    
    #対数バリア関数型回避関数のヘッシアン
    def evasion_hes_x(self,x):
        dedxx = np.zeros((self.len_x,self.len_x))
        for i in range(self.eva_dim):
            z = np.dot((x-self.xe[:,i:i+1]).T, np.dot(self.E,(x-self.xe[:,i:i+1]))) - self.r**2
            dzdx = 2 * np.dot((x-self.xe[:,i:i+1]).T, self.E)
            dzdxx = 2 * self.E
            dedz = self.barrier_dz(z)
            dedzz = self.barrier_hes_z(z)
            dedxx += self.e[i] * (dedz* dzdxx + np.dot(dzdx.T, dedzz * dzdx))
        
        return dedxx




if __name__=="__main__":
    x_ob = np.array([[5, 0, 0]],dtype=float).T
    x_st = np.array([[0, 0, 0]],dtype=float).T

    nonholo_car = car(x_st)
    iLQR_cont = iLQR_controller(nonholo_car, x_ob)

    Time = 0

    x_log = []

    start = time.time()
    while Time <= 20:
        print("-------------Position-------------")
        print(iLQR_cont.model.x)
        print(iLQR_cont.u)

        x_log.append(iLQR_cont.model.x)

        x = iLQR_cont.model.x + iLQR_cont.model.func(iLQR_cont.model.x, iLQR_cont.u)*iLQR_cont.Ts

        iLQR_cont.iLQR_control()

        Time += iLQR_cont.Ts

        iLQR_cont.model.x = x

    end = time.time()

    loop_time = end - start

    print("計算時間：{}[s]".format(loop_time))

    x_log = np.array(x_log)
    fig, ax = plt.subplots()
    ax.plot(x_log[:,0],x_log[:,1])
    plt.show()