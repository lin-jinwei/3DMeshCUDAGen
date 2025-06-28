import numpy as np
from numba import cuda, float32

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    from math import cos, sin, radians
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = cos(radians(theta) / 2.0)
    b, c, d = -axis * sin(radians(theta) / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],
                     [0, 0, 0, 1]], dtype=np.float32)

@cuda.jit(device=True)
def taylor_cos(x):
    # Taylor series expansion for cosine up to the 6th term
    term = 1.0
    result = term
    x_squared = x * x
    factorial = 2.0
    
    for i in range(1, 4):  # 4 terms gives us up to x^6
        term *= -x_squared / (factorial * (factorial + 1))
        result += term
        factorial += 2.0
    
    return result

@cuda.jit(device=True)
def taylor_sin(x):
    # Taylor series expansion for sine up to the 5th term
    term = x
    result = term
    x_cubed = x * x * x
    factorial = 3.0
    
    for i in range(1, 4):  # 4 terms gives us up to x^5
        term *= -x_cubed / (factorial * (factorial + 1) * (factorial + 2))
        result += term
        factorial += 2.0
    
    return result

@cuda.jit
def compute_rotation_matrices(rotation_x, rotation_y, rotation_z, rotation_matrices):
    idx = cuda.grid(1)
    if idx == 0:
        # Convert angles to radians manually
        pi_over_180 = 3.141592653589793 / 180.0
        rx_rad = rotation_x * pi_over_180 / 2.0
        ry_rad = rotation_y * pi_over_180 / 2.0
        rz_rad = rotation_z * pi_over_180 / 2.0
        
        # Compute trigonometric values using Taylor series
        ax = taylor_cos(rx_rad)
        bx = -taylor_sin(rx_rad)
        ay = taylor_cos(ry_rad)
        by = -taylor_sin(ry_rad)
        az = taylor_cos(rz_rad)
        bz = -taylor_sin(rz_rad)
        
        # Compute squared terms
        aax, bbx, ccx, ddx = ax*ax, bx*bx, 0*0, 0*0
        aay, bby, ccy, ddy = ay*ay, by*by, 0*0, 0*0
        aaz, bbz, ccz, ddz = az*az, bz*bz, 0*0, 0*0
        
        # Compute mixed terms
        bcx, adx, acx, abx, bdx, cdx = bx*0, ax*0, ax*0, ax*bx, bx*0, 0*0
        bcy, ady, acy, aby, bdy, cdy = by*0, ay*0, ay*0, ay*by, by*0, 0*0
        bcz, adz,acz, abz, bdz, cdz = bz*0, az*0, az*0, az*bz, bz*0, 0*0
        
        # Rotation matrices
        rotation_x_mat = np.array([
            [aax + bbx - ccx - ddx, 2*(bcx + adx), 2*(bdx - acx), 0],
            [2*(bcx - adx), aax + ccx - bbx - ddx, 2*(cdx + abx), 0],
            [2*(bdx + acx), 2*(cdx - abx), aax + ddx - bbx - ccx, 0],
            [0, 0, 0, 1]
        ], dtype=float32)
        
        rotation_y_mat = np.array([
            [aay + bby - ccy - ddy, 2*(bcy + ady), 2*(bdy - acy), 0],
            [2*(bcy - ady), aay + ccy - bby - ddy, 2*(cdy + aby), 0],
            [2*(bdy + acy), 2*(cdy - aby), aay + ddy - bby - ccy, 0],
            [0, 0, 0, 1]
        ], dtype=float32)
        
        rotation_z_mat = np.array([
            [aaz + bbz - ccz - ddz, 2*(bcz + adz), 2*(bdz - acz), 0],
            [2*(bcz - adz), aaz + ccz - bbz - ddz, 2*(cdz + abz), 0],
            [2*(bdz + acz), 2*(cdz - abz), aaz + ddz - bbz - ccz, 0],
            [0, 0, 0, 1]
        ], dtype=float32)
        
        # Combine rotation matrices
        rotation_xy = np.zeros((4, 4), dtype=float32)
        for i in range(4):
            for j in range(4):
                rotation_xy[i, j] = rotation_x_mat[i, 0] * rotation_y_mat[0, j] + \
                                    rotation_x_mat[i, 1] * rotation_y_mat[1, j] + \
                                    rotation_x_mat[i, 2] * rotation_y_mat[2, j] + \
                                    rotation_x_mat[i, 3] * rotation_y_mat[3, j]
        
        rotation_xyz = np.zeros((4, 4), dtype=float32)
        for i in range(4):
            for j in range(4):
                rotation_xyz[i, j] = rotation_xy[i, 0] * rotation_z_mat[0, j] + \
                                     rotation_xy[i, 1] * rotation_z_mat[1, j] + \
                                     rotation_xy[i, 2] * rotation_z_mat[2, j] + \
                                     rotation_xy[i, 3] * rotation_z_mat[3, j]
        
        # Store the final rotation matrix
        for i in range(4):
            for j in range(4):
                rotation_matrices[i, j] = rotation_xyz[i, j]

@cuda.jit
def apply_rotation_kernel(pose, rotation_matrices, rotated_pose):
    idx = cuda.grid(1)
    if idx == 0:
        # Manual matrix multiplication: rotated_pose = rotation_matrices @ pose
        for i in range(4):
            for j in range(4):
                sum_val = 0
                for k in range(4):
                    sum_val += rotation_matrices[i, k] * pose[k, j]
                rotated_pose[i, j] = sum_val

def apply_rotation(pose, rotation_x, rotation_y, rotation_z):
    """Apply rotations around X, Y, and Z axes to a pose matrix."""
    # Ensure inputs are NumPy arrays
    pose = np.array(pose, dtype=np.float32)
    
    # Allocate device memory for rotation matrices
    d_rotation_matrices = cuda.device_array((4, 4), dtype=np.float32)
    
    # Allocate device memory for input and output pose matrices
    d_pose = cuda.to_device(pose)
    d_rotated_pose = cuda.device_array_like(d_pose)
    
    # Configure CUDA kernel for computing rotation matrices
    threads_per_block = 1
    blocks_per_grid = 1
    
    # Launch CUDA kernel to compute rotation matrices
    compute_rotation_matrices[blocks_per_grid, threads_per_block](rotation_x, rotation_y, rotation_z, d_rotation_matrices)
    
    # Launch CUDA kernel to apply rotation matrices to pose
    apply_rotation_kernel[blocks_per_grid, threads_per_block](d_pose, d_rotation_matrices, d_rotated_pose)
    
    # Copy results back to host
    rotated_pose = d_rotated_pose.copy_to_host()
    
    return rotated_pose.tolist()

# 示例用法
if __name__ == "__main__":
    # 示例姿态矩阵
    pose = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
    
    # 旋转角度
    rotation_x = 90
    rotation_y = 0
    rotation_z = 0
    
    # 应用旋转
    rotated_pose = apply_rotation(pose, rotation_x, rotation_y, rotation_z)
    print(rotated_pose)



