import os
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import fem2geo
from fem2geo import model_handler as mh
from fem2geo import tensor_methods as tm
from fem2geo import transform_funcs as tr

# -------------------------------
# Step 0: Parse command-line arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Generate slip, dilation and summarised tendency plots using fem2geo.")
parser.add_argument("--model_name", required=True, help="Model name for output naming")
parser.add_argument("--example_dir", required=True, help="Directory containing the VTU file")
parser.add_argument("--filename", required=True, help="VTU file name")
parser.add_argument("--points", default="fem2geo_points.csv", help="Path to points CSV file")
parser.add_argument("--outdir", default=".", help="Output directory for PNG files")
parser.add_argument("--dpi", type=int, default=300, help="Resolution for PNG output (default: 300 dpi)")
parser.add_argument("--color_min_slip", type=float, default=None, help="Min color for slip subplot (ax1)")
parser.add_argument("--color_max_slip", type=float, default=None, help="Max color for slip subplot (ax1)")
parser.add_argument("--color_min_dilation", type=float, default=None, help="Min color for dilation subplot (ax2)")
parser.add_argument("--color_max_dilation", type=float, default=None, help="Max color for dilation subplot (ax2)")
parser.add_argument("--color_min_summary", type=float, default=None, help="Min color for summary subplot (ax3)")
parser.add_argument("--color_max_summary", type=float, default=None, help="Max color for summary subplot (ax3)")

args = parser.parse_args()

model_name = args.model_name
example_dir = args.example_dir
filename = args.filename
points_file = args.points
output_dir = args.outdir
dpi = args.dpi

color_min_slip = args.color_min_slip
color_max_slip = args.color_max_slip
color_min_dilation = args.color_min_dilation
color_max_dilation = args.color_max_dilation
color_min_summary = args.color_min_summary
color_max_summary = args.color_max_summary


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Step 1: Read the VTU model
# -------------------------------
file_path = os.path.join(example_dir, filename)
full_model = pv.read(file_path)

# -------------------------------
# Step 2: Read points from CSV
# -------------------------------
points = []
with open(points_file, 'r') as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    required_cols = ['Point_label', 'X', 'Y', 'Z', 'radius']
    missing_cols = [col for col in required_cols if col not in reader.fieldnames]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for row in reader:
        points.append(row)

# -------------------------------
# Step 3: Loop through points
# -------------------------------
for row in points:
    center = (float(row['X']), float(row['Y']), float(row['Z']))
    radius = float(row['radius'])
    point_label = row['Point_label']

    print(f"Processing point: {point_label} at {center} with radius {radius}")

    sub_model = mh.get_submodel_sphere(full_model, center, radius)
    short_filename = os.path.join(example_dir, f'dilation_zone_{point_label}.vtu')
    sub_model.save(short_filename)
    sub_model = pv.read(short_filename)

    s1 = [tr.line_enu2sphe(i) for i in sub_model.cell_data['dir_DevStress1_(-)']]
    s2 = [tr.line_enu2sphe(i) for i in sub_model.cell_data['dir_DevStress2_(-)']]
    s3 = [tr.line_enu2sphe(i) for i in sub_model.cell_data['dir_DevStress3_(-)']]

    avg_stress = mh.get_stress_weightedavg(sub_model)
    val, vec = np.linalg.eig(avg_stress)
    vec = vec[:, np.argsort(val)]
    val = np.sort(val)

    s1_avg = tr.line_enu2sphe(vec[:, 0].T)
    s2_avg = tr.line_enu2sphe(vec[:, 1].T)
    s3_avg = tr.line_enu2sphe(vec[:, 2].T)

    fig, ax1, ax2, ax3, D1, D2, D3, planes = tm.plot_slipndilationnsummarised_tendency(avg_stress)
     
    # Apply custom color limits per subplot if provided
    # Slip (ax1)
    if color_min_slip is not None and color_max_slip is not None:
        for coll in ax1.collections:
            coll.set_clim(color_min_slip, color_max_slip)

    # Dilation (ax2)
    if color_min_dilation is not None and color_max_dilation is not None:
        for coll in ax2.collections:
            coll.set_clim(color_min_dilation, color_max_dilation)

    # Summarised (ax3)
    if color_min_summary is not None and color_max_summary is not None:
        for coll in ax3.collections:
            coll.set_clim(color_min_summary, color_max_summary)

    # Add stress directions to slip tendency plot (not applied - same for 3 subplots)
    for n, i in enumerate(zip(s1, s2, s3)):
        mylabel = [None, None, None]
        if n == 0:
            mylabel = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
        ax1.line(i[0][0], i[0][1], c='r', marker='o', markeredgecolor='k', label=mylabel[0])
        ax1.line(i[1][0], i[1][1], c='g', marker='s', markeredgecolor='k', label=mylabel[1])
        ax1.line(i[2][0], i[2][1], c='b', marker='v', markeredgecolor='k', label=mylabel[2])

    ax1.line(s1_avg[0], s1_avg[1], c='w', marker='o', markeredgecolor='k', markersize=8, label=r'Average $\sigma_1$')
    ax1.line(s2_avg[0], s2_avg[1], c='w', marker='s', markeredgecolor='k', markersize=8, label=r'Average $\sigma_2$')
    ax1.line(s3_avg[0], s3_avg[1], c='w', marker='v', markeredgecolor='k', markersize=8, label=r'Average $\sigma_3$')
    #ax1.legend()
        
    ax1.set_title('Slip tendency plot \n' +
                  r'$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
                  (val[0], val[2], (val[1] - val[2]) / (val[0] - val[2])), y=1.05)

    # Add stress directions to dilation tendency plot (not applied)
    for n, i in enumerate(zip(s1, s2, s3)):
        mylabel = [None, None, None]
        if n == 0:
            mylabel = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
        ax2.line(i[0][0], i[0][1], c='r', marker='o', markeredgecolor='k', label=mylabel[0])
        ax2.line(i[1][0], i[1][1], c='g', marker='s', markeredgecolor='k', label=mylabel[1])
        ax2.line(i[2][0], i[2][1], c='b', marker='v', markeredgecolor='k', label=mylabel[2])

    ax2.line(s1_avg[0], s1_avg[1], c='w', marker='o', markeredgecolor='k', markersize=8, label=r'Average $\sigma_1$')
    ax2.line(s2_avg[0], s2_avg[1], c='w', marker='s', markeredgecolor='k', markersize=8, label=r'Average $\sigma_2$')
    ax2.line(s3_avg[0], s3_avg[1], c='w', marker='v', markeredgecolor='k', markersize=8, label=r'Average $\sigma_3$')
    #ax2.legend()
    
    ax2.set_title('Dilation tendency plot \n' +
                  r'$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
                  (val[0], val[2], (val[1] - val[2]) / (val[0] - val[2])), y=1.05)
    
    
    # Add stress directions to summarised tendency plot (applied)
    for n, i in enumerate(zip(s1, s2, s3)):
        mylabel = [None, None, None]
        if n == 0:
            mylabel = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
        ax3.line(i[0][0], i[0][1], c='r', marker='o', markeredgecolor='k', label=mylabel[0])
        ax3.line(i[1][0], i[1][1], c='g', marker='s', markeredgecolor='k', label=mylabel[1])
        ax3.line(i[2][0], i[2][1], c='b', marker='v', markeredgecolor='k', label=mylabel[2])

    ax3.line(s1_avg[0], s1_avg[1], c='w', marker='o', markeredgecolor='k', markersize=8, label=r'Average $\sigma_1$')
    ax3.line(s2_avg[0], s2_avg[1], c='w', marker='s', markeredgecolor='k', markersize=8, label=r'Average $\sigma_2$')
    ax3.line(s3_avg[0], s3_avg[1], c='w', marker='v', markeredgecolor='k', markersize=8, label=r'Average $\sigma_3$')
    #ax3.legend()
    #move legend to avoid overlap of stereonets
    ax3.legend(loc='center left', bbox_to_anchor=(1, 1))

    
    denom = (val[0] - val[2])
    phi = (val[1] - val[2]) / denom if denom != 0 else float("inf")

    ax3.set_title(
        f"Summarised tendency {model_name} {point_label}\n"
        f"$\\sigma_1={val[0]:.1f}$, $\\sigma_3={val[2]:.1f}$, $\\phi={phi:.2f}$", y=1.05
    )
    
    safe_model_name = model_name.replace('/', '_')
    safe_point_label = point_label.replace('/', '_')
    output_filename = os.path.join(output_dir, f"G_Slip_Dilatancy_Summarised_3stereoplots_{safe_model_name}_{safe_point_label}.png")
    print(f"Saving figure to: {output_filename}")
    plt.savefig(output_filename, dpi=dpi)
    plt.close(fig)

print("Processing completed. Output files saved for all points.")