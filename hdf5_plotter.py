import h5py
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np

class HDF5Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Viewer")

        # File selection button
        self.btn_open = tk.Button(root, text="Open HDF5 File", command=self.load_file)
        self.btn_open.pack(pady=5)

        # Treeview for HDF5 structure
        self.tree = ttk.Treeview(root)
        self.tree.pack(expand=True, fill="both", padx=10, pady=5)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Plot button
        self.btn_plot = tk.Button(root, text="Plot Selected Data", command=self.plot_data, state=tk.DISABLED)
        self.btn_plot.pack(pady=5)

        self.filepath = None
        self.h5_file = None

    def load_file(self):
        """Open file dialog and load HDF5 file structure into the treeview."""
        self.filepath = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5 *.doric")])
        if not self.filepath:
            return

        try:
            self.h5_file = h5py.File(self.filepath, "r")
            self.tree.delete(*self.tree.get_children())  # Clear previous tree structure
            self.insert_tree("", self.h5_file)  # Populate tree
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file:\n{str(e)}")

    def insert_tree(self, parent, obj, path=""):
        """Recursively add HDF5 groups and datasets to the treeview."""
        for name, item in obj.items():
            full_path = f"{path}/{name}"
            if isinstance(item, h5py.Group):
                node = self.tree.insert(parent, "end", text=name, open=False)
                self.insert_tree(node, item, full_path)
            elif isinstance(item, h5py.Dataset):
                node = self.tree.insert(parent, "end", text=f"{name} (Dataset)", values=[full_path])

    def on_select(self, event):
        """Enable plot button when a dataset is selected."""
        selected_item = self.tree.focus()
        if selected_item:
            item_values = self.tree.item(selected_item, "values")
            if item_values:
                self.btn_plot.config(state=tk.NORMAL)
            else:
                self.btn_plot.config(state=tk.DISABLED)

    def plot_data(self):
        """Retrieve and plot the selected dataset."""
        selected_item = self.tree.focus()
        if not selected_item:
            return

        dataset_path = self.tree.item(selected_item, "values")
        if not dataset_path:
            return
        
        dataset_path = dataset_path[0]  # Get actual path from tree values
        
        try:
            data = self.h5_file[dataset_path][:]
            
            # 1D plot (Time Series)
            if data.ndim == 1:
                plt.figure(figsize=(8, 4))
                plt.plot(data, label=dataset_path)
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title(f"Plot of {dataset_path}")
                plt.legend()
                plt.show()
            
            # 2D plot (Heatmap)
            elif data.ndim == 2:
                plt.figure(figsize=(6, 5))
                plt.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
                plt.colorbar(label="Value")
                plt.title(f"Heatmap of {dataset_path}")
                plt.show()
            
            else:
                messagebox.showinfo("Unsupported Data", "Only 1D and 2D datasets can be plotted.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot data:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5Viewer(root)
    root.mainloop()
