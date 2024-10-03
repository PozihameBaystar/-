import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 図とaxesオブジェクトの作成
fig, ax = plt.subplots()

# プロットの範囲設定
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# 円オブジェクトの作成（最初は原点に配置）
circle, = ax.plot(0, 0, 'ro', markersize=15)

# アニメーションの更新関数
def update(frame):
    # フレーム数に基づいて円の新しい位置を計算
    x = 4 * np.cos(frame / 20)
    y = 4 * np.sin(frame / 20)
    
    # 円の位置を更新
    circle.set_data(x, y)
    
    return circle,

# アニメーションの作成
anim = FuncAnimation(fig, update, frames=200, interval=100, blit=True)

# アニメーションの表示
plt.show()