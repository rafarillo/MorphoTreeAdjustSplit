"""Edit a 1D grayscale signal and explore its max-tree with Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import MaxNLocator

import morphoTreeAdjust as mta


MAX_LEVEL = 10


def main():
    signal = np.array([0, 3, 1, 4, 0, 2, 5, 5, 2, 0], dtype=np.uint8)
    samples = np.arange(signal.size)
    fig, (signal_ax, tree_ax) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained", sharey=True
    )
    (signal_line,) = signal_ax.plot(samples, signal, color="0.6", drawstyle="steps-mid")
    signal_points = signal_ax.scatter(samples, signal, color="tab:blue", zorder=3)
    component_points = signal_ax.scatter([], [], color="tab:orange", s=65, zorder=4)
    signal_ax.set(
        title="Sinal 1D",
        xlabel="Amostra",
        ylabel="Nivel de cinza",
        xticks=samples,
        xlim=(-0.5, signal.size - 0.5),
        ylim=(-0.5, MAX_LEVEL + 0.5),
    )
    signal_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    active_sample = None
    nodes = []
    points = None
    selected_point = None
    tree = None

    def draw_tree():
        nonlocal tree, nodes, points, selected_point
        tree = mta.DynamicComponentTree(signal.reshape(1, -1), True, 1.0)
        nodes = list(tree.nodes)
        positions = {}
        for node in nodes:
            pixels = tree.getPixelsOfCC(node)
            # A connected component in 1D is an interval.
            positions[node] = ((min(pixels) + max(pixels)) / 2, tree.getAltitude(node))

        tree_ax.clear()
        tree_ax.set(
            title="Max-tree",
            xlabel="Centro do componente",
            xlim=signal_ax.get_xlim(),
        )
        tree_ax.grid(axis="y", alpha=0.25)
        for node in nodes:
            if node != tree.root:
                parent = tree.getNodeParent(node)
                x, y = positions[node]
                px, py = positions[parent]
                tree_ax.plot([px, x], [py, y], color="0.45", zorder=1)

        x, y = zip(*(positions[node] for node in nodes))
        points = tree_ax.scatter(x, y, color="tab:blue", s=45, zorder=2)
        selected_point = tree_ax.scatter([], [], color="tab:orange", s=65, zorder=3)
        component_points.set_offsets(np.empty((0, 2)))

    def drag(event):
        if active_sample is None or event.inaxes != signal_ax or event.ydata is None:
            return
        level = min(MAX_LEVEL, max(0, round(event.ydata)))
        if level == signal[active_sample]:
            return
        signal[active_sample] = level
        signal_line.set_ydata(signal)
        signal_points.set_offsets(np.column_stack((samples, signal)))
        draw_tree()
        fig.canvas.draw_idle()

    def press(event):
        nonlocal active_sample
        if event.button != MouseButton.LEFT or fig.canvas.widgetlock.locked():
            return
        if event.inaxes == signal_ax and event.xdata is not None:
            active_sample = min(signal.size - 1, max(0, round(event.xdata)))
            drag(event)
        elif (
            event.inaxes == tree_ax
            and tree is not None
            and points is not None
            and selected_point is not None
        ):
            hit, details = points.contains(event)
            if hit:
                index = details["ind"][0]
                pixels = tree.getPixelsOfCC(nodes[index])
                component_points.set_offsets(np.column_stack((pixels, signal[pixels])))
                selected_point.set_offsets(points.get_offsets()[[index]])
            else:
                component_points.set_offsets(np.empty((0, 2)))
                selected_point.set_offsets(np.empty((0, 2)))
            fig.canvas.draw_idle()

    def release(event):
        nonlocal active_sample
        if event.button == MouseButton.LEFT:
            active_sample = None

    draw_tree()
    fig.canvas.mpl_connect("button_press_event", press)
    fig.canvas.mpl_connect("motion_notify_event", drag)
    fig.canvas.mpl_connect("button_release_event", release)
    plt.show()


if __name__ == "__main__":
    main()
