import math
import numpy as np
import time

#移動ロボットクラス
class car:
    def __init__(self,x_st): #引数は初期位置
        self.R = 0.05
        self.T = 0.2
        self.r = self.R/2
        self.rT = self.R/self.T

        self.x = x_st #初期位置をセット

        self.len_u = 2
        self.len_x = 3

        self.umax = np.array([15,15],dtype=float)
        self.umin = np.array([-15,-15],dtype=float)

    def func(self,x,u):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        dx = np.array([[self.r*cos_*(u[0][0]+u[1][0])],
                       [self.r*sin_*(u[0][0]+u[1][0])],
                       [self.rT*(u[0][0]-u[1][0])]])
        return dx
    
    def func_risan(self,x,u,dt):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        dx = np.array([[self.r*cos_*(u[0][0]+u[1][0])],
                       [self.r*sin_*(u[0][0]+u[1][0])],
                       [self.rT*(u[0][0]-u[1][0])]])
        x_next = x + dx*dt #ここでx += としてしまうと引数のxを変えてしまう
        return x_next
    
    #dfdx（状態方程式は連続）
    def Calcfx(self, x, u):
        dfdx = np.array([[0, 0, -self.r*math.sin(x[2][0])*(u[0][0]+u[1][0])],
                         [0, 0, self.r*math.cos(x[2][0])*(u[0][0]+u[1][0])],
                         [0, 0, 0]])
        return dfdx
    
    #dfdu（状態方程式は連続）
    def Calcfu(self, x, u):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        dfdu = np.array([[self.r*cos_, self.r*cos_],
                    [self.r*sin_, self.r*sin_],
                    [self.rT, -self.rT]])
        return dfdu
    
    def CalcA_risan(self,x,u,dt):
        Ak = np.eye(3) + \
            np.array([[0, 0, -self.r*math.sin(x[2][0])*(u[0][0]+u[1][0])],
                      [0, 0, self.r*math.cos(x[2][0])*(u[0][0]+u[1][0])],
                      [0, 0, 0]])*dt
        return Ak
    
    def CalcB_risan(self,x,dt):
        cos_ = math.cos(x[2][0])
        sin_ = math.sin(x[2][0])
        Bk = np.array([[self.r*cos_, self.r*cos_],
                       [self.r*sin_, self.r*sin_],
                       [self.rT, -self.rT]])*dt
        return Bk
    

#コントローラークラス
class controller:
    def __init__(self, car, x_ob):
        #コントローラーのパラメータ
        self.Ts = 0.05 #制御周期
        self.ht = self.Ts 
        self.zeta = 1.0/self.Ts #安定化係数
        self.tf = 1.0 #予測ホライズンの最終値
        self.alpha = 0.5 #予測ホライズンの変化パラメータ
        self.N = 20 #予測ホライズンの分割数
        self.Time = 0.0 #時刻を入れる変数
        self.dt = 0.0 #予測ホライズンの分割幅

        #操縦する車
        self.car = car

        #入力と状態の次元
        self.len_u = self.car.len_u #入力の次元
        self.len_x = self.car.len_x #状態変数の次元

        #評価関数の重み
        self.Q = np.array([[100, 0, 0],
                           [0, 100, 0],
                           [0, 0, 0]])
        self.R = np.array([[1, 0],
                           [0, 1]])
        self.S = np.array([[100, 0, 0],
                           [0, 100, 0],
                           [0, 0, 0]])
        
        #コントローラーの変数（状態変数は車の方に）
        self.u = np.zeros((self.len_u, 1))
        self.U = np.zeros((self.len_u * self.N, 1))
        self.dU = np.zeros((self.len_u * self.N, 1))

        #入力の制限
        self.umax = np.array([[15],
                              [15]]) #各入力の最大値
        self.umin = np.array([[-15],
                              [-15]]) #各入力の最小値
        
        #バリア関数用パラメータ
        self.r = 0.15
        self.del_bar = 0.5
        self.bar_C = np.concatenate([np.eye(self.len_u),-np.eye(self.len_u)],0)
        self.bar_d = np.concatenate([self.umax,-self.umin],0)
        self.con_dim = len(self.bar_d)
        
        #目標地点
        self.x_ob = x_ob


    def CGMRES_control(self):
        self.dt = (1-math.exp(-self.alpha*self.Time))*self.tf/self.N
        dx = self.car.func(self.car.x, self.u)

        Fux = self.CalcF(self.car.x + dx*self.ht, self.U + self.dU*self.ht)
        Fx = self.CalcF(self.car.x + dx*self.ht, self.U)
        F = self.CalcF(self.car.x, self.U)

        left = (Fux - Fx)/self.ht
        right = -self.zeta*F - (Fx - F)/self.ht
        r0 = right - left

        m = 10 #self.len_u*self.N

        Vm = np.zeros((self.len_u*self.N, m+1))
        Vm[:,0:1] = r0/np.linalg.norm(r0)

        Hm = np.zeros((m+1,m))

        for i in range(m):
            Fux = self.CalcF(self.car.x + dx*self.ht, self.U + Vm[:,i:i+1]*self.ht)
            Av = (Fux - Fx)/self.ht

            for k in range(i+1):
                Hm[k][i] = np.matmul(Av.T,Vm[:,k:k+1])

            temp_vec = np.zeros((self.len_u*self.N, 1))
            for k in range(i+1):
                temp_vec = temp_vec + Hm[k][i]*Vm[:,k:k+1]

            v_hat = Av - temp_vec

            Hm[i+1][i] = np.linalg.norm(v_hat)
            Vm[:,i+1:i+2] = v_hat/Hm[i+1][i]

        e = np.zeros((m+1, 1))
        e[0][0] = 1.0
        gm_ = np.linalg.norm(r0)*e

        UTMat, gm_ = self.ToUTMat(Hm, gm_, m)

        min_y = np.zeros((m, 1))

        for i in range(m):
            min_y[i][0] = (gm_[i][0] - np.matmul(UTMat[i:i+1,:],min_y))/UTMat[i][i]

        dU_new = self.dU + np.matmul(Vm[:,0:m], min_y)

        self.dU = dU_new
        self.U = self.U + self.dU*self.ht
        self.u = self.U[0:2,0:1]

    
    #F
    def CalcF(self, x, U):
        F = np.zeros((self.len_u*self.N, 1))
        U = U.reshape(self.len_u, self.N, order='F')
        X = self.Forward(x, U)

        Lambda = self.Backward(X, U)

        for i in range(self.N):
            F[self.len_u*i:self.len_u*(i+1), 0:1] = self.CalcHu(U[:,i:i+1], X[:,i:i+1], Lambda[:,i:i+1])

        return F
    
    #xの予測計算
    def Forward(self, x, U):
        X = np.zeros((self.len_x, self.N+1))

        X[:,0:1] = x

        for i in range(1,self.N+1):
            dx = self.car.func(X[:,i-1:i], U[:,i-1:i])
            X[:,i:i+1] = X[:,i-1:i] + dx*self.dt

        return X
    
    #随伴変数の計算
    def Backward(self, X, U):
        Lambda = np.zeros((self.len_x, self.N))
        Lambda[:,self.N-1:self.N] = np.matmul(self.S, X[:,self.N:self.N+1]-self.x_ob)

        for i in reversed(range(self.N-1)):
            Lambda[:,i:i+1] = Lambda[:,i+1:i+2] + self.CalcHx(X[:,i+1:i+2], U[:,i+1:i+2], Lambda[:,i+1:i+2])*self.dt
        
        return Lambda
    
    #dH/du
    def CalcHu(self, u, x, lambd):
        B = self.car.Calcfu(x,u)
        dHdu = np.zeros((self.len_u, 1))
        dHdu = np.matmul(self.R, u) + np.matmul(B.T, lambd)
        dBdu = self.barrier_du(u)
        dHdu += self.r * dBdu.T

        return dHdu
    
    #dHdx
    def CalcHx(self, x, u, lambd):
        dHdx = np.zeros((self.len_x, 1))
        dfdx = self.car.Calcfx(x,u)
        dHdx = np.matmul(self.Q, x-self.x_ob) + np.matmul(dfdx.T, lambd)
        return dHdx
    
    
    #Givens回転
    def ToUTMat(self, Hm, gm, m):
        for i in range(m):
            nu = math.sqrt(Hm[i][i]**2 + Hm[i+1][i]**2)
            c_i = Hm[i][i]/nu
            s_i = Hm[i+1][i]/nu
            Omega = np.eye(m+1)
            Omega[i][i] = c_i
            Omega[i][i+1] = s_i
            Omega[i+1][i] = -s_i
            Omega[i+1][i+1] = c_i

            Hm = np.matmul(Omega, Hm)
            gm = np.matmul(Omega, gm)

        return Hm, gm
    

    #バリア関数（全体）
    def barrier(self, u):
        zs = self.bar_d - np.dot(self.bar_C,u)

        Bar = 0

        for i in range(self.con_dim):
            Bar += self.barrier_z(zs[i][0])

        return Bar
    
    #バリア関数のu微分（答えは行ベクトルになる）
    def barrier_du(self, u):
        zs = self.bar_d - np.dot(self.bar_C,u)

        dBdu = np.zeros((1,self.len_u))

        for i in range(self.con_dim):
            dBdz = self.barrier_dz(zs[i][0])
            dBdu -= dBdz * self.bar_C[i:i+1,:]

        return dBdu
    
    #バリア関数のuヘッシアン（答えは行列になる）
    def barrier_hes_u(self, u):
        zs = self.bar_d - np.dot(self.bar_C,u)

        dBduu = np.zeros((self.len_u,self.len_u))

        for i in range(self.con_dim):
            dBdzz = self.barrier_hes_z(zs[i][0])
            dBduu += dBdzz * np.dot(self.bar_C[i:i+1,:].T,self.bar_C[i:i+1,:])

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

    

if __name__ == "__main__":
    x_ob = np.array([[3, 2, 0]]).T
    x_st = np.array([[0, 0, 0]],dtype=float).T

    nonholo_car = car(x_st)
    CGMRES_cont = controller(nonholo_car, x_ob)

    Time = 0

    start = time.time()
    while Time <= 20:
        print("-------------Position-------------")
        print(CGMRES_cont.car.x)

        x = CGMRES_cont.car.x + CGMRES_cont.car.func(CGMRES_cont.car.x, CGMRES_cont.u)*CGMRES_cont.Ts

        CGMRES_cont.Time = Time + CGMRES_cont.Ts

        CGMRES_cont.CGMRES_control()

        Time += CGMRES_cont.Ts

        CGMRES_cont.car.x = x

    end = time.time()

    loop_time = end - start

    print("計算時間：{}[s]".format(loop_time))