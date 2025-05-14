import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import quad # 如果需要精确计算单点，可以取消注释

# --- 常量定义 ---
a = 1.0  # 圆环半径 (单位: m)
# q = 1.0  # 可以定义 q 参数，或者直接在 C 中体现
# V(x,y,z) = q/(2*pi) * integral(...)
# C 对应 q/(2*pi)，这里设 q=1
C = 1.0 / (2 * np.pi)

# --- 计算函数 ---

def calculate_potential_on_grid(y_coords, z_coords):
    """
    在 yz 平面 (x=0) 的网格上计算电势 V(0, y, z)。
    """
    print("开始计算电势...")
    # 生成phi积分点（1000个点）
    phi = np.linspace(0, 2*np.pi, 1000)  
    d_phi = phi[1] - phi[0]
    
    # 创建网格：维度顺序为 (z, y, phi)
    z_grid, y_grid, phi_grid = np.meshgrid(z_coords, y_coords, phi, indexing='ij')
    
    # 计算圆环上电荷元坐标
    x_s = a * np.cos(phi_grid)
    y_s = a * np.sin(phi_grid)
    z_s = 0.0
    
    # 计算场点与电荷元的距离
    R = np.sqrt((0 - x_s)**2 + (y_grid - y_s)**2 + (z_grid - z_s)**2)
    
    # 避免除以零
    R[R < 1e-10] = 1e-10
    
    # 计算电势积分
    dV = C / R
    V = np.trapz(dV, x=phi, axis=2)
    
    print("电势计算完成.")
    return V, y_grid[:, :, 0], z_grid[:, :, 0]

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    计算电场分量
    """
    print("开始计算电场...")
    dy = y_coords[1] - y_coords[0]
    dz = z_coords[1] - z_coords[0]
    
    # 计算梯度（注意V的维度是 (z, y)，对应轴(0:z, 1:y)）
    # np.gradient 传入间距顺序应对应轴顺序
    grad_z, grad_y = np.gradient(-V, dz, dy, axis=(0,1))
    
    Ey = grad_y
    Ez = grad_z
    
    print("电场计算完成.")
    return Ey, Ez

# --- 可视化函数 ---

def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制等势线和电场线
    """
    print("开始绘图...")
    fig = plt.figure('Potential and Electric Field of Charged Ring (yz plane, x=0)', figsize=(12, 6))
    
    # 等势线图
    plt.subplot(1, 2, 1)
    levels = np.linspace(V.min(), V.max(), 20)
    cf = plt.contourf(y_grid/a, z_grid/a, V, levels=levels, cmap='viridis')
    plt.colorbar(cf, label='Electric Potential (V)')
    plt.contour(y_grid/a, z_grid/a, V, levels=levels, colors='k', linewidths=0.5)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Equipotential Lines')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    
    # 电场线图
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    color = np.log(E_magnitude + 1e-10)  # 对数颜色映射
    plt.streamplot(y_grid/a, z_grid/a, Ey, Ez, color=color, cmap='plasma', 
                   linewidth=1, density=1.5, arrowsize=1)
    plt.plot([0], [0], 'ro', markersize=8, label='Ring')  # 圆环截面
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Electric Field Lines')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    # 参数设置
    num_points_y = 100
    num_points_z = 100
    range_factor = 3
    y_range = np.linspace(-range_factor*a, range_factor*a, num_points_y)
    z_range = np.linspace(-range_factor*a, range_factor*a, num_points_z)
    
    # 计算电势和电场
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    
    # 可视化
    if V is not None and Ey is not None and Ez is not None:
        plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
    else:
        print("计算未完成，无法绘图。")
