import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from linalg_3d.transformations import (
    rotation_matrix,
    scaling_matrix,
    shearing_matrix,
    reflection_about_axis_matrix
)
from linalg_3d.figures import fish

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

        # Set meaningful defaults for Transformation A (Rotation -45° in Y-Z plane)
        self.A_type.set("Rotation")
        self.update_inputs("A")
        self.A_param_vars[0][2].set(-45.0)   # angle
        self.A_param_vars[1][2].set(1)        # dim0 = Y
        self.A_param_vars[2][2].set(2)        # dim1 = Z

        # Action Buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        ttk.Button(btn_frame, text="Apply Transformations", command=self.apply_transformations).pack(fill=tk.X)
        ttk.Button(btn_frame, text="Reset View", command=self.reset_view).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Reset Demo", command=self.reset_demo).pack(fill=tk.X, pady=5)

        # Transformation History
        history_label = ttk.LabelFrame(controls_frame, text="Transformation History")
        history_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        self.history_listbox = tk.Listbox(history_label, font=("Arial", 8))
        self.history_listbox.pack(fill=tk.BOTH, expand=True)

        # Plots Frame (Right)
        plots_frame = ttk.Frame(main_frame)
        plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(121, projection='3d', proj_type='ortho')
        self.ax2 = self.fig.add_subplot(122, projection='3d', proj_type='ortho')

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
        type_var = tk.StringVar(value="None (Identity)")
        type_cb = ttk.Combobox(
            group,
            textvariable=type_var,
            state="readonly",
            values=["None (Identity)", "Rotation", "Scaling", "Shear", "Reflection"]
        )
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

        if t_type == "None (Identity)":
            set_row(0, "", 0, False)
            set_row(1, "", 0, False)
            set_row(2, "", 0, False)
        elif t_type == "Rotation":
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

    def _describe_transform(self, prefix):
        """Return a short human-readable description of the current transform settings."""
        t_type = getattr(self, f"{prefix}_type").get()
        vars = getattr(self, f"{prefix}_param_vars")
        values = [v[2].get() for v in vars]
        axis_names = {0: "X", 1: "Y", 2: "Z"}

        if t_type == "None (Identity)":
            return "Identity"
        elif t_type == "Rotation":
            return f"Rot({values[0]}°, {axis_names.get(int(values[1]),'?')}-{axis_names.get(int(values[2]),'?')})"
        elif t_type == "Scaling":
            return f"Scale({values[0]}, {values[1]}, {values[2]})"
        elif t_type == "Shear":
            return f"Shear({axis_names.get(int(values[0]),'?')}->{axis_names.get(int(values[1]),'?')}, {values[2]})"
        elif t_type == "Reflection":
            return f"Reflect({axis_names.get(int(values[0]),'?')})"
        return t_type

    def init_plots(self):
        # Create the original cube and store it
        self.original_segments = fish(2.0)
        # Current state for each plot (start with original)
        self.current_segments_AB = list(self.original_segments)
        self.current_segments_BA = list(self.original_segments)
        # History tracking
        self.history = []
        # Show the original figure without any transformation
        self._draw_plots()

    def _draw_plots(self):
        """Redraw both plots using the current segment state."""
        self.plot_segments(self.ax1, self.current_segments_AB, 'purple', "Order: A then B (B@A)")
        self.plot_segments(self.ax2, self.current_segments_BA, 'orange', "Order: B then A (A@B)")
        self.canvas.draw()

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
            print(f"in gui: ", seg)
            xs = [seg.start[0], seg.end[0]]
            ys = [seg.start[1], seg.end[1]]
            zs = [seg.start[2], seg.end[2]]
            ax.plot3D(xs, ys, zs, color=color)

            # Draw dot at end point
            ax.scatter(seg.start[0], seg.start[1], seg.start[2], color='yellow', s=15)
            ax.scatter(seg.end[0], seg.end[1], seg.end[2], color='blue', s=15)

        # Set standardized limits
        limit = 3
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

    def apply_transformations(self):
        """Apply the current A/B transformation pair on top of the previous state."""
        mat_a = self.get_matrix("A")
        mat_b = self.get_matrix("B")

        mat_AB = mat_b @ mat_a   # A then B
        mat_BA = mat_a @ mat_b   # B then A

        # Transform from current state (cumulative)
        self.current_segments_AB = [s.transform(mat_AB) for s in self.current_segments_AB]
        self.current_segments_BA = [s.transform(mat_BA) for s in self.current_segments_BA]

        # Record history entry
        desc_a = self._describe_transform("A")
        desc_b = self._describe_transform("B")
        step = len(self.history) + 1
        entry = f"#{step}  A={desc_a}  B={desc_b}"
        self.history.append(entry)
        self.history_listbox.insert(tk.END, entry)
        self.history_listbox.see(tk.END)

        self._draw_plots()

    def reset_demo(self):
        """Reset to the original figure and clear transformation history."""
        self.current_segments_AB = list(self.original_segments)
        self.current_segments_BA = list(self.original_segments)
        self.history.clear()
        self.history_listbox.delete(0, tk.END)
        self._draw_plots()

    def reset_view(self):
        self.ax1.view_init(elev=30, azim=45)
        self.ax2.view_init(elev=30, azim=45)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TransformationApp(root)
    root.mainloop()

