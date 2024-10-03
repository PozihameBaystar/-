import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# 初期位置と角度
x, y = 0, 0
theta = 0

# 図とaxesオブジェクトの作成
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')

# 透明な円を作成
circle = Circle((x, y), radius=0.5, fill=False)
ax.add_artist(circle)

# 向きを表す線を作成
line, = ax.plot([], [], 'r-', lw=2)

# アニメーションの更新関数
def update(frame):
    global x, y, theta
    
    # 新しい位置と角度を計算
    x += 0.05 * np.cos(theta)
    y += 0.05 * np.sin(theta)
    theta += 0.1
    
    # 円の位置を更新
    circle.center = (x, y)
    
    # 線の位置を更新
    line_x = [x, x + 0.5 * np.cos(theta)]
    line_y = [y, y + 0.5 * np.sin(theta)]
    line.set_data(line_x, line_y)
    
    return circle, line

# アニメーションの作成
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# アニメーションの表示
plt.show()