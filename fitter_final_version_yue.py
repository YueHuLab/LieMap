import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import os
import datetime
import mrcfile
import re

# --- Atom Parsers ---
def parse_structure_file(file_path):
    """Robustly parses a PDB or CIF file to get structured atom data."""
    atom_data = []
    try:
        with open(file_path, 'r') as f:
            file_lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 结构文件未找到于 {file_path}"); return None

    is_cif = file_path.lower().endswith('.cif')
    if is_cif:
        header, data_lines = [], []
        in_loop, header_done = False, False
        for line in file_lines:
            s = line.strip()
            if s == 'loop_': in_loop = True; header_done = False; header = []
            elif s.startswith('_atom_site.') and in_loop: header.append(s)
            elif in_loop and s and not s.startswith('#') and not s.startswith('_'):
                header_done = True; data_lines.append(s)
        
        if not header or not data_lines: return None
        col_map = {name: i for i, name in enumerate(header)}
        x_col, y_col, z_col = col_map.get('_atom_site.Cartn_x'), col_map.get('_atom_site.Cartn_y'), col_map.get('_atom_site.Cartn_z')
        atom_col, chain_col = col_map.get('_atom_site.label_atom_id'), col_map.get('_atom_site.auth_asym_id')
        res_seq_col, res_name_col = col_map.get('_atom_site.auth_seq_id'), col_map.get('_atom_site.label_comp_id')
        if any(c is None for c in [x_col, y_col, z_col, atom_col, chain_col, res_seq_col, res_name_col]): return None
        
        for line in data_lines:
            try:
                parts = line.split()
                atom_data.append({
                    'chain': parts[chain_col], 'res_seq': int(parts[res_seq_col]), 'res_name': parts[res_name_col],
                    'atom_name': parts[atom_col].strip('"'), 'coords': [float(parts[x_col]), float(parts[y_col]), float(parts[z_col])]
                })
            except (ValueError, IndexError): continue
    else: # PDB
        for line in file_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    atom_data.append({
                        'chain': line[21], 'res_seq': int(line[22:26]), 'res_name': line[17:20].strip(),
                        'atom_name': line[12:16].strip(), 'coords': [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    })
                except (ValueError, IndexError): continue
    
    print(f"从 '{os.path.basename(file_path)}' 解析了 {len(atom_data)} 个原子。")
    if not atom_data: return None
    return atom_data

# --- Map Processing & Other Logic ---
def parse_mrc(file_path):
    try:
        with mrcfile.open(file_path, permissive=True) as mrc: 
            native_dtype = mrc.data.dtype.newbyteorder('='); data = mrc.data.astype(native_dtype)
            map_data = torch.tensor(data, dtype=torch.float32); hdr = mrc.header
            if not (hdr.mx>0 and hdr.my>0 and hdr.mz>0 and hdr.cella.x>0 and hdr.cella.y>0 and hdr.cella.z>0): return None, None, None
            vx=float(hdr.cella.x)/float(hdr.mx); vy=float(hdr.cella.y)/float(hdr.my); vz=float(hdr.cella.z)/float(hdr.mz)
            voxel_size = torch.tensor([vx, vy, vz], dtype=torch.float32)
            origin = torch.tensor([float(hdr.origin.x), float(hdr.origin.y), float(hdr.origin.z)], dtype=torch.float32)
            print(f"密度图 '{os.path.basename(file_path)}' 已加载。 维度: {map_data.shape}, 体素大小: {voxel_size.numpy()} Å, 文件头原点: {origin.numpy()} Å")
            return map_data, voxel_size, origin
    except Exception as e: print(f"解析MRC文件时发生意外错误: {e}"); return None, None, None

def get_map_mask(map_data, sigma_level=3.0):
    """Generates a binary mask for the map, keeping only values above a sigma threshold."""
    print(f"正在基于 {sigma_level}-sigma 水平生成密度图掩码...")
    print("  - Sigma值将仅基于图中的正密度值计算。  ")

    positive_densities = map_data[map_data > 0]
    if positive_densities.numel() == 0:
        print("警告: 密度图中找不到任何正值。")
        return torch.zeros_like(map_data)

    map_mean = positive_densities.mean()
    map_std = positive_densities.std()
    density_threshold = map_mean + sigma_level * map_std
    print(f"  - 正密度均值: {map_mean:.4f}, 正密度标准差 (sigma): {map_std:.4f}")
    print(f"  - {sigma_level}-sigma 阈值: {density_threshold:.4f}")

    mask = (map_data > density_threshold).float()
    print(f"  - 掩码已生成, 包含 {int(mask.sum())} 个有效体素。")
    return mask

def gaussian_blur_3d(map_tensor, sigma, kernel_size=9):
    print(f"正在对密度图进行高斯模糊 (sigma={sigma})...", end='', flush=True)
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32); g = torch.exp(-x**2 / (2 * sigma**2)); g = g / g.sum()
    k_x=g.view(1,1,-1,1,1); k_y=g.view(1,1,1,-1,1); k_z=g.view(1,1,1,1,-1)
    m = map_tensor.unsqueeze(0).unsqueeze(0); p = kernel_size // 2
    m = F.conv3d(F.conv3d(F.conv3d(m, k_x, padding=(p,0,0)), k_y, padding=(0,p,0)), k_z, padding=(0,0,p))
    print("完成。")
    return m.squeeze(0).squeeze(0)

# --- Core Algorithm ---
def get_transformation_matrix(params):
    w, u = params[:3], params[3:]; W = torch.zeros((3,3),dtype=params.dtype,device=params.device); W[0,1],W[0,2]=-w[2],w[1]; W[1,0],W[1,2]=w[2],-w[0]; W[2,0],W[2,1]=-w[1],w[0]
    return torch.linalg.matrix_exp(W), u

def get_fit_score(coords, map_tensor, voxel_size, map_origin):
    """Calculates a fit score by trilinear interpolation of density values at atom coordinates."""
    c_v = (coords - map_origin) / voxel_size
    c0 = torch.floor(c_v).long()
    c1 = c0 + 1
    f = c_v - c0.float()
    
    dims = torch.tensor(map_tensor.shape, device=c0.device, dtype=torch.long) - 1
    c0 = torch.max(torch.zeros_like(c0), torch.min(c0, dims))
    c1 = torch.max(torch.zeros_like(c1), torch.min(c1, dims))
    
    d000 = map_tensor[c0[:, 2], c0[:, 1], c0[:, 0]]; d001 = map_tensor[c0[:, 2], c0[:, 1], c1[:, 0]]
    d010 = map_tensor[c0[:, 2], c1[:, 1], c0[:, 0]]; d011 = map_tensor[c0[:, 2], c1[:, 1], c1[:, 0]]
    d100 = map_tensor[c1[:, 2], c0[:, 1], c0[:, 0]]; d101 = map_tensor[c1[:, 2], c0[:, 1], c1[:, 0]]
    d110 = map_tensor[c1[:, 2], c1[:, 1], c0[:, 0]]; d111 = map_tensor[c1[:, 2], c1[:, 1], c1[:, 0]]
    
    d00 = d000 * (1 - f[:, 0]) + d001 * f[:, 0]; d01 = d010 * (1 - f[:, 0]) + d011 * f[:, 0]
    d10 = d100 * (1 - f[:, 0]) + d101 * f[:, 0]; d11 = d110 * (1 - f[:, 0]) + d111 * f[:, 0]
    
    d0 = d00 * (1 - f[:, 1]) + d01 * f[:, 1]; d1 = d10 * (1 - f[:, 1]) + d11 * f[:, 1]
    
    return (d0 * (1 - f[:, 2]) + d1 * f[:, 2]).mean()

def calculate_rmsd(c1, c2): 
    return torch.sqrt(torch.mean(torch.sum((c1 - c2)**2, dim=1)))

# --- Main Execution ---
def main():
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="将蛋白质结构拟合到Cryo-EM密度图中的最终版本。  ")
    parser.add_argument("--mobile_structure", required=True, help="需要拟合的移动结构文件 (.cif 或 .pdb)。")
    parser.add_argument("--target_map", required=True, help="目标Cryo-EM密度图文件 (.mrc)。")
    parser.add_argument("--gold_standard_structure", required=True, help="用于计算最终RMSD的金标准结构文件。")
    parser.add_argument("--output", default=None, help="保存拟合后的PDB结构的路径。")
    parser.add_argument("--lr", type=float, default=0.1, help="优化学习率。  ")
    parser.add_argument("--steps", type=int, default=100, help="优化步数。  ")
    parser.add_argument("--sigma_level", type=float, default=3.0, help="用于生成掩码的Sigma水平。  ")
    args = parser.parse_args()

    print(f"\n--- Cryo-EM 最终拟合程序 ---\n程序开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    mobile_data = parse_structure_file(args.mobile_structure)
    gold_data = parse_structure_file(args.gold_standard_structure)
    if not mobile_data or not gold_data: print("错误: 结构文件解析失败。"); return

    all_mobile_coords = torch.tensor([a['coords'] for a in mobile_data], dtype=torch.float32)
    mobile_ca_coords = torch.tensor([a['coords'] for a in mobile_data if a['atom_name'] == 'CA'], dtype=torch.float32)
    
    mobile_ca_map = {(a['chain'], a['res_seq']): a['coords'] for a in mobile_data if a['atom_name'] == 'CA'}
    gold_ca_map = {(a['chain'], a['res_seq']): a['coords'] for a in gold_data if a['atom_name'] == 'CA'}
    common_keys = sorted(list(mobile_ca_map.keys() & gold_ca_map.keys()))
    print(f"在两个结构中找到了 {len(common_keys)} 个共同的C-alpha原子用于计算最终RMSD。")
    if not common_keys: print("错误: 找不到任何共同的C-alpha原子!"); return
    common_mobile_ca_coords = torch.tensor([mobile_ca_map[k] for k in common_keys], dtype=torch.float32)
    common_gold_ca_coords = torch.tensor([gold_ca_map[k] for k in common_keys], dtype=torch.float32)

    map_data, voxel_size, origin_from_header = parse_mrc(args.target_map)
    if map_data is None: return

    map_mask = get_map_mask(map_data, args.sigma_level)
    map_masked = map_data * map_mask
    map_blurred_masked = gaussian_blur_3d(map_masked, sigma=4.0)

    # --- Pre-alignment ---
    mask_indices = map_mask.nonzero(as_tuple=False).float()
    if mask_indices.shape[0] == 0:
        print("错误: 掩码中无有效体素，无法进行预对齐。")
        map_com_angstrom = torch.zeros(3)
    else:
        map_com_vox = mask_indices.mean(dim=0)
        map_com_angstrom = map_com_vox * voxel_size + origin_from_header

    struct_center = all_mobile_coords.mean(dim=0)
    initial_t = map_com_angstrom - struct_center
    print(f"--- 预对齐 (结构质心 -> 掩码质心) ---\n  - 结构质心 (Å): {struct_center.numpy()}\n  - 掩码质心 (Å): {map_com_angstrom.numpy()}\n  - 计算得到的初始平移 (Å): {initial_t.numpy()}")

    transform_params = torch.zeros(6, requires_grad=True)
    with torch.no_grad(): transform_params[3:] = initial_t
    
    stages = [
        {'name': '1. 粗略 (CA @ 模糊图)', 'coords': mobile_ca_coords, 'steps': 500, 'lr': 0.01},
        {'name': '2. 精细 (全部原子 @ 原图)', 'coords': all_mobile_coords, 'steps': 1800, 'lr': 0.005}
    ]

    for stage in stages:
        print(f"\n--- 正在执行阶段: {stage['name']} ({stage['steps']} 步, lr={stage['lr']}) ---")
        current_map = map_blurred_masked if '模糊' in stage['name'] else map_masked
        optimizer = optim.Adam([transform_params], lr=stage['lr'])
        for step in range(stage['steps']):
            optimizer.zero_grad()
            R, t = get_transformation_matrix(transform_params)
            
            transformed_coords = (R @ stage['coords'].T).T + t
            score = get_fit_score(transformed_coords, current_map, voxel_size, origin_from_header)
            loss = -score

            if step % 10 == 0 or step == stage['steps'] - 1:
                with torch.no_grad():
                    transformed_common_ca = (R @ common_mobile_ca_coords.T).T + t
                    current_rmsd = calculate_rmsd(transformed_common_ca, common_gold_ca_coords)
                print(f"  步骤 {step:04d}: 平均密度得分 = {score.item():.6f}, RMSD = {current_rmsd.item():.4f} Å")
            
            loss.backward(); optimizer.step()

    print("\n--- 所有阶段优化完成 ---")
    R_final, t_final = get_transformation_matrix(transform_params.detach().clone())
    
    final_coords = (R_final @ all_mobile_coords.T).T + t_final
    transformed_common_mobile_coords = (R_final @ common_mobile_ca_coords.T).T + t_final
    final_rmsd = calculate_rmsd(transformed_common_mobile_coords, common_gold_ca_coords)
    print(f"\n最终RMSD (对比 '{os.path.basename(args.gold_standard_structure)}' 的CA原子): {final_rmsd.item():.4f} Å")

    output_filename = args.output if args.output else f"{os.path.splitext(os.path.basename(args.mobile_structure))[0]}_final_version_rmsd_{final_rmsd.item():.2f}.pdb"
    print(f"\n--- 正在将拟合后的PDB写入 '{output_filename}' ---")
    with open(output_filename, 'w') as f:
        for i, atom in enumerate(mobile_data):
            x,y,z = final_coords[i]
            f.write(f"ATOM  {i+1:5d} {atom['atom_name']:<4s} {atom['res_name']:<3s} {atom['chain']}{atom['res_seq']:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

    print(f"成功！您现在可以一起查看 '{args.target_map}' 和 '{output_filename}'。")
    print(f"\n程序结束于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n总执行时间: {datetime.datetime.now() - start_time}")

if __name__ == '__main__':
    main()
