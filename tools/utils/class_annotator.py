from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from scipy.ndimage.filters import gaussian_filter


def create_heatmap(
    x,
    y,
    sigma,
    bins=1000,
    extent=None,
):
    if extent is None:
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    else:
        hist_range = [[extent[0], extent[1]], [extent[2], extent[3]]]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=hist_range)

    heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap.T, extent


class Class_Annotator:
    def __init__(
        self,
        data: List[np.ndarray],
        display_std: bool = True,
        plot_alpha: float = 1,
        plot_s: float = 4,
        upper_bounds=[0.1, 0.1, 0.1],
        lower_bounds=[-1, -1, -1],
        axis_names=["Blue Contrast", "Green Contrast", "Red Contrast"],
    ):
        self.data = data
        self.mean_data = np.array([np.mean(d, axis=0) for d in data])
        self.std_data = np.array([np.std(d, axis=0) for d in data])
        self.data_flat = np.concatenate(data, axis=0)

        self.display_std = display_std
        self.point_cluster_ids = np.zeros(len(data), dtype=int)
        self.point_ids_in_legend = []
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.axis_names = axis_names

        self.current_cluster_id = 1
        self.colors = np.array(
            [
                "gray",
                "blue",
                "red",
                "green",
                "yellow",
                "purple",
                "orange",
                "pink",
                "brown",
                "gray",
                "olive",
                "cyan",
            ]
        )

        self.lassos = []
        self.axs = []
        self.figs = []

        self.scatter_args = {"alpha": plot_alpha, "s": plot_s}

    def update_all_figures(self):
        for fig in self.figs:
            fig.canvas.draw_idle()

    def update_titles(self):
        for idx, ax in enumerate(self.axs):
            ax.set_title(
                f"{idx} Current cluster ID: {self.current_cluster_id} (press 'a' to decrease, 'd' to increase)"
            )

    def handle_keypresses(self, event):
        if event.key == "d":
            self.current_cluster_id += 1
            self.update_titles()
            self.update_all_figures()

        elif event.key == "a":
            if self.current_cluster_id > 0:
                self.current_cluster_id -= 1
                self.update_titles()
                self.update_all_figures()

        elif event.key == "x":
            self.current_cluster_id = 0
            self.update_titles()
            self.update_all_figures()

        elif event.key == "q":
            plt.close("all")

    def onselect(self, verts, ax_index):
        projected_data = self.mean_data[:, [ax_index, (ax_index + 1) % 3]]
        path = Path(verts)
        is_point_selected = path.contains_points(projected_data)
        selected_point_indices = np.where(is_point_selected)[0]

        # Assign cluster ID to selected points
        self.point_cluster_ids[selected_point_indices] = self.current_cluster_id

        for i in range(3):
            idx_1 = i
            idx_2 = (i + 1) % 3
            updated_1 = self.mean_data[:, idx_1][selected_point_indices]
            updated_2 = self.mean_data[:, idx_2][selected_point_indices]

            if self.display_std:
                updated_std_1 = self.std_data[:, idx_1][selected_point_indices]
                updated_std_2 = self.std_data[:, idx_2][selected_point_indices]
                self.axs[i].errorbar(
                    updated_1,
                    updated_2,
                    xerr=updated_std_1,
                    yerr=updated_std_2,
                    linestyle="None",
                    marker="o",
                    color=self.colors[self.current_cluster_id],
                    alpha=self.scatter_args["alpha"],
                    # **self.scatter_args,
                )
            else:
                self.axs[i].scatter(
                    updated_1,
                    updated_2,
                    color=self.colors[self.current_cluster_id],
                    **self.scatter_args,
                )

        self.add_legend()
        self.update_all_figures()

    def plot_scatter(self, ax, idx_1, idx_2):
        if self.display_std:
            ax.errorbar(
                self.mean_data[:, idx_1],
                self.mean_data[:, idx_2],
                xerr=self.std_data[:, idx_1],
                yerr=self.std_data[:, idx_2],
                linestyle="None",
                marker="o",
                color=self.colors[0],
                alpha=self.scatter_args["alpha"],
                # **self.scatter_args,
            )
        else:
            ax.scatter(
                self.mean_data[:, idx_1],
                self.mean_data[:, idx_2],
                color=self.colors[0],
                **self.scatter_args,
            )

    def plot_heatmap(self, ax, idx_1, idx_2):
        heatmap, extent = create_heatmap(
            self.data_flat[:, idx_1],
            self.data_flat[:, idx_2],
            sigma=1,
            bins=100,
            extent=[
                self.lower_bounds[idx_1],
                self.upper_bounds[idx_1],
                self.lower_bounds[idx_2],
                self.upper_bounds[idx_2],
            ],
        )

        ax.imshow(heatmap, extent=extent, origin="lower", cmap=cm.plasma, aspect="auto")

    def add_legend(self):
        unique_cluster_ids = np.unique(self.point_cluster_ids)
        cluster_ids = [
            c for c in unique_cluster_ids if c not in self.point_ids_in_legend
        ]
        if len(cluster_ids) == 0:
            return

        for ax in self.axs:
            for cluster_id in cluster_ids:
                label = f"Cluster ID: {cluster_id}"
                if cluster_id == 0:
                    label = "No cluster ID"

                ax.scatter([], [], color=self.colors[cluster_id], label=label)
                self.point_ids_in_legend.append(cluster_id)

            ax.legend()

    def run(self):
        # Three different figures for three 2D projections

        for i in range(3):
            idx_1 = i
            idx_2 = (i + 1) % 3
            fig, ax = plt.subplots()
            ax.set_title(
                f"{i} Current cluster ID: {self.current_cluster_id} (press 'a' to decrease, 'd' to increase)"
            )
            self.plot_heatmap(ax, idx_1, idx_2)
            self.plot_scatter(ax, idx_1, idx_2)

            upper_bound_x = self.upper_bounds[idx_1]
            upper_bound_y = self.upper_bounds[idx_2]
            lower_bound_x = self.lower_bounds[idx_1]
            lower_bound_y = self.lower_bounds[idx_2]

            ax.axhline(0, color="black", lw=2, linestyle="--")
            ax.axvline(0, color="black", lw=2, linestyle="--")

            ax.set_xlabel(self.axis_names[idx_1])
            ax.set_ylabel(self.axis_names[idx_2])
            ax.set_xlim(lower_bound_x, upper_bound_x)
            ax.set_ylim(lower_bound_y, upper_bound_y)
            ax.set_aspect("equal")
            self.lassos.append(
                LassoSelector(
                    ax, lambda verts, current_i=i: self.onselect(verts, current_i)
                )
            )
            fig.canvas.mpl_connect(
                "key_press_event", lambda event: self.handle_keypresses(event)
            )
            self.figs.append(fig)
            self.axs.append(ax)

        self.add_legend()
        plt.show()

    def get_results(self):
        return self.point_cluster_ids
