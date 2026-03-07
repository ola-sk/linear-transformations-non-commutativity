import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

import linalg_3d
from linalg_3d.vector3d import Vector3D
from linalg_3d.line_segment import LineSegment
from linalg_3d.transformations import (
    rotation_matrix,
    scaling_matrix,
    shearing_matrix,
    reflection_about_axis_matrix
)

class TransformationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Multiplicaton Non-Commutativity Demo")
        self.root.geometry("1400x900")

        # Container for controls and plots
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controls Frame (Left)
        controls_frame = ttk.Frame(main_frame, width=300)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Transformation A Controls
        self.create_transform_group(controls_frame, "Transformation A", "A")

        # Transformation B Controls
        self.create_transform_group(controls_frame, "Transformation B", "B")

        # Action Buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        ttk.Button(btn_frame, text="Apply Transformations", command=self.update_plots).pack(fill=tk.X)
        ttk.Button(btn_frame, text="Reset View", command=self.reset_view).pack(fill=tk.X, pady=5)

        # Plots Frame (Right)
        plots_frame = ttk.Frame(main_frame)
        plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.init_plots()

    def create_transform_group(self, parent, title, prefix):
        group = ttk.LabelFrame(parent, text=title)
        group.pack(fill=tk.X, pady=10, padx=5)

        # Legend (explaining the dimensions)
        legend_frame = ttk.Frame(group)
        legend_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(legend_frame, text="Axis Mappings: 0=X, 1=Y, 2=Z", font=("Arial", 8, "italic"), foreground="gray").pack(anchor=tk.W)

        # Type selection
        ttk.Label(group, text="Type:").pack(anchor=tk.W)
        type_var = tk.StringVar(value="Rotation")
        type_cb = ttk.Combobox(group, textvariable=type_var, state="readonly",
                               values=["Rotation", "Scaling", "Shear", "Reflection"])
        type_cb.pack(fill=tk.X, pady=2)
        setattr(self, f"{prefix}_type", type_var)

        # Parameters frame (dynamic based on type - simplified for now with all inputs visible or reusing rows)
        # For simplicity, I'll create generic input fields 1-3 and label them dynamically or just have fixed per type
        # Let's use a parameter frame
        param_frame = ttk.Frame(group)
        param_frame.pack(fill=tk.X, pady=5)

        # Store widgets to update visibility/labels
        setattr(self, f"{prefix}_params_frame", param_frame)
        setattr(self, f"{prefix}_param_vars", [])

        for i in range(3):
            f = ttk.Frame(param_frame)
            f.pack(fill=tk.X)
            lbl = ttk.Label(f, text=f"Param {i}:", width=18)
            lbl.pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            ent = ttk.Entry(f, textvariable=var)
            ent.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            getattr(self, f"{prefix}_param_vars").append((lbl, ent, var, f))

        # Bind change event to update labels
        type_cb.bind("<<ComboboxSelected>>", lambda e, p=prefix: self.update_inputs(p))
        self.update_inputs(prefix) # Initialize

    def update_inputs(self, prefix):
        t_type = getattr(self, f"{prefix}_type").get()
        widgets = getattr(self, f"{prefix}_param_vars") # list of (lbl, ent, var, frame)

        # Helper to set label and default
        def set_row(idx, label, default, visible=True):
            lbl, ent, var, frame = widgets[idx]
            if visible:
                frame.pack(fill=tk.X)
                lbl.config(text=label)
                # Only reset value if type changed? For now, keep it simple
                var.set(default)
            else:
                frame.pack_forget()

        if t_type == "Rotation":
            set_row(0, "Angle (deg):", 45.0)
            set_row(1, "Axis 1 (0,1,2):", 0) # dim0
            set_row(2, "Axis 2 (0,1,2):", 1) # dim1
        elif t_type == "Scaling":
            set_row(0, "X Scale (dim0):", 1.0)
            set_row(1, "Y Scale (dim1):", 1.0)
            set_row(2, "Z Scale (dim2):", 1.0)
        elif t_type == "Shear":
            set_row(0, "Source (0,1,2):", 1)
            set_row(1, "Target (0,1,2):", 0)
            set_row(2, "Factor:", 1.0)
        elif t_type == "Reflection":
            set_row(0, "Axis (0,1,2):", 0)
            set_row(1, "", 0, False)
            set_row(2, "", 0, False)

    def get_matrix(self, prefix):
        t_type = getattr(self, f"{prefix}_type").get()
        vars = getattr(self, f"{prefix}_param_vars")
        values = [v[2].get() for v in vars]

        try:
            if t_type == "Rotation":
                angle_rad = np.radians(values[0])
                dim0, dim1 = int(values[1]), int(values[2])
                return rotation_matrix(angle_rad, dim0, dim1)
            elif t_type == "Scaling":
                return scaling_matrix(values[0], values[1], values[2])
            elif t_type == "Shear":
                source, target = int(values[0]), int(values[1])
                return shearing_matrix(source, target, values[2])
            elif t_type == "Reflection":
                return reflection_about_axis_matrix(int(values[0]))
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters for {prefix}: {e}")
            return np.eye(3)
        except Exception as e:
            messagebox.showerror("Error", f"Unknown error in {prefix}: {e}")
            return np.eye(3)

        return np.eye(3)

    def init_plots(self):
        # Create a sample cube
        self.original_segments = self.create_cube()
        self.update_plots()

    def create_cube(self):
        # Define 12 edges of a cube centered at origin
        points = [
            Vector3D([1, 1, 1]), Vector3D([1, 1, -1]),
            Vector3D([1, -1, 1]), Vector3D([1, -1, -1]),
            Vector3D([-1, 1, 1]), Vector3D([-1, 1, -1]),
            Vector3D([-1, -1, 1]), Vector3D([-1, -1, -1])
        ]
        segments = []
        # Connect appropriately
        # Edges along x
        segments.append(LineSegment(points[0], points[4]))
        segments.append(LineSegment(points[1], points[5]))
        segments.append(LineSegment(points[2], points[6]))
        segments.append(LineSegment(points[3], points[7]))
        # Edges along y
        segments.append(LineSegment(points[0], points[2]))
        segments.append(LineSegment(points[1], points[3]))
        segments.append(LineSegment(points[4], points[6]))
        segments.append(LineSegment(points[5], points[7]))
        # Edges along z
        segments.append(LineSegment(points[0], points[1]))
        segments.append(LineSegment(points[2], points[3]))
        segments.append(LineSegment(points[4], points[5]))
        segments.append(LineSegment(points[6], points[7]))
        return segments

    def plot_segments(self, ax, segments, color='b', title=""):
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Draw axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1.5, normalize=True)

        for seg in segments:
            # seg.start and seg.end are Vector3D (ndarray)
            xs = [seg.start[0], seg.end[0]]
            ys = [seg.start[1], seg.end[1]]
            zs = [seg.start[2], seg.end[2]]
            ax.plot(xs, ys, zs, color=color)

        # Set standardized limits
        limit = 3
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

    def update_plots(self):
        mat_a = self.get_matrix("A")
        mat_b = self.get_matrix("B")

        # Order A -> B => Apply A then B => v' = B(A(v)) = (B @ A) @ v
        # So Combined Matrix M1 = B @ A
        mat_AB = mat_b @ mat_a

        # Order B -> A => Apply B then A => v' = A(B(v)) = (A @ B) @ v
        # Combined Matrix M2 = A @ B
        mat_BA = mat_a @ mat_b

        # Transform segments
        segs_AB = [s.transform(mat_AB) for s in self.original_segments]
        segs_BA = [s.transform(mat_BA) for s in self.original_segments]

        self.plot_segments(self.ax1, segs_AB, 'purple', "Order: A then B (B@A)")
        self.plot_segments(self.ax2, segs_BA, 'orange', "Order: B then A (A@B)")

        self.canvas.draw()

    def reset_view(self):
        self.ax1.view_init(elev=30, azim=45)
        self.ax2.view_init(elev=30, azim=45)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TransformationApp(root)
    root.mainloop()

