import numpy as np
import casadi

class mpc_controller:
    def __init__(self):
        self.tf = 3.0  # 予測ホライズン
        self.N = 30  # 予測ホライズンの分割幅
        self.dt = self.tf/self.N

    # 制御対象（model）を読み込み、状態変数や入力、状態方程式を設定する
    def make_model(self,model):
        x = self.model.x_node
        u = self.model.u_node
        x_dot = self.model.x_dot_node
        f = self.model.f  # 状態方程式のCasADi関数オブジェクト
        
        nx = self.model.nx
        nu = self.model.nu

        self.x_node = x
        self.u_node = u
        self.x_dot_node = x_dot
        self.f = f
        self.nx = nx
        self.nu = nu


class iLQR_controller(mpc_controller):
    # iLQR用に関数オブジェクトを追加作成する
    def make_model(self, model):
        super().make_model(model)

        x = self.x_node
        u = self.u_node
        x_dot = self.x_dot_node
        f = self.f  # 状態方程式のCasADi関数オブジェクト
        
        nx = self.nx
        nu = self.nu

        dt = self.dt

        x_next = x + x_dot*dt
        F = casadi.Function('F',[x,u],[x_next],['x','u'],['x_next'])
        dFdx_node = casadi.jacobian()


if __name__ == "__main__":
    pass