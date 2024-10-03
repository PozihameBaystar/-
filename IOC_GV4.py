import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
#import os
#rel_do_mpc_path = os.path.join('..','..','..')
#sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

_x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
_u = model.set_variable(var_type='_u', var_name='u', shape=(2,1))

A = np.array([[1,0],
              [0,1]])

B = np.array([[0.1,0],
              [0,0.1]])

x_next = A@_x + B@_u

model.set_rhs('x', x_next)

# 制御対象（モデル）の設定完了
model.setup()

# コントローラーの設定
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_robust': 0,
    'n_horizon': 30,
    't_step': 0.1,
    'state_discretization': 'discrete',
    'store_full_solution':True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

# 評価関数のパラメータ
Args = {
    'Q':np.array([[1,0],
                  [0,1]]),
    'R':np.array([[1,0],
                  [0,1]])
}

# 評価関数の設定
_x = model.x
_u = model.u

x_ob = np.array([[5],
                 [5]])

lterm = 0.5 * (_x-x_ob) @ Args['Q'] @ (_x-x_ob)
mterm = 0

mpc.set_objective(mterm=mterm, lterm=lterm)

rterm = 0.5 * _u @ Args['R'] @ _u

mpc.set_rterm(rterm=rterm)

mpc.setup()

# シミュレーターのセット（推測器は使わない）
simulator = do_mpc.simulator.Simulator(model)

# シミュレーターの設定
params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.1
}

simulator.set_param(**params_simulator)

simulator.setup()

x0 = np.array([[0],
               [0]])

# コントローラーとシミュレーターに初期値を設定
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()