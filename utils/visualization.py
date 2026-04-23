import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import json
import pdb

# Try to import wandb, but don't fail if it's not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from . import file_logger


def log_node_data_to_wandb(e_mon, n_eval: int, use_wandb: bool = True):
    """Log node data to wandb as a table and create task graphs"""
    import pandas as pd

    # Create wandb table
    if use_wandb and WANDB_AVAILABLE:
        table = wandb.Table(columns=["task_id", "x", "y", "task_vec", "solution_id", "solution", "fitness",
                                     "individual_learning_success", "individual_learning_fail",
                                     "social_learning_success", "social_learning_fail"])
    else:
        table = None

    # Create DataFrame with all data at once instead of using concat
    data_rows = []
    for task_id in e_mon.nodes():
        node_data = e_mon.get_node(task_id)
        if node_data is not None:
            task_vec = node_data["task_config"]["task_vec"]
            if len(task_vec) == 12:
                x = np.sum(task_vec[:6])
                y = np.sum(task_vec[6:])
            else:
                x, y = task_vec[0], task_vec[1]
            solution_id = node_data.get("solution_id", -1)
            solution_vec = node_data.get("solution", None)
            fitness = node_data.get("fitness", float('-inf'))

            # Add to table
            if table is not None:
                table.add_data(
                    task_id,
                    x,
                    y,
                    task_vec,
                    solution_id,
                    solution_vec,
                    fitness,
                    node_data.get('individual_learning_success', 0),
                    node_data.get('individual_learning_fail', 0),
                    node_data.get('social_learning_success', 0),
                    node_data.get('social_learning_fail', 0),
                )

            # Collect data for DataFrame
            data_rows.append({
                "task_id": task_id,
                "task_vec": task_vec,
                "x": x,
                "y": y,
                "solution_id": solution_id,
                "solution": solution_vec,
                "fitness": fitness,
                'individual_learning_success': node_data.get('individual_learning_success', 0),
                'individual_learning_fail': node_data.get('individual_learning_fail', 0),
                'social_learning_success': node_data.get('social_learning_success', 0),
                'social_learning_fail': node_data.get('social_learning_fail', 0),
            })

    # Create DataFrame from collected data instead of using concat
    df = pd.DataFrame(data_rows) if data_rows else pd.DataFrame(columns=["task_id", "x", "y", "task_vec", "solution_id", "solution_vec", "fitness", "individual_learning_success", "individual_learning_fail", "social_learning_success", "social_learning_fail"])

    # Log the table
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({f"node_data/n_eval_{n_eval}": table}, step=n_eval)
    else:
        # Save table data to CSV file
        df_table = table.to_dataframe() if hasattr(table, 'to_dataframe') else df
        table_file = os.path.join(file_logger._logger.run_dir if file_logger._logger else "logs", f"node_data_n_eval_{n_eval}.csv")
        df_table.to_csv(table_file, index=False)

    # Create visualization graphs
    _create_task_graphs(df, n_eval, use_wandb)



def _create_task_graphs(df, n_eval, use_wandb: bool = True):
    """Create and log task visualization graphs."""
    # Create a graph
    G = nx.Graph()

    # Add nodes with positions
    for _, row in df.iterrows():
        G.add_node(int(row['task_id']), pos=(row['x'], row['y']),
                  solution_id=int(row['solution_id']), fitness=row['fitness'])

    # Get positions for plotting (ordered by x and y)
    pos = nx.get_node_attributes(G, 'pos')

    # Create graphs colored by solution_id and fitness
    fig_solution = _create_solution_colored_graph(G, pos)
    fig_fitness = _create_fitness_colored_graph(G, pos)

    # Save or log the graphs
    if use_wandb and WANDB_AVAILABLE:
        # fig_solution.write_image("task_graph.png")
        # fig_fitness.write_image("task_graph_fitness.png")

        wandb.log({
            "task_graph_html": wandb.Html(fig_solution.to_html(include_plotlyjs="cdn")),
            # "task_graph_image": wandb.Image("task_graph.png"),
            # "task_graph_fitness_image": wandb.Image("task_graph_fitness.png"),
            "task_graph_fitness_html": wandb.Html(fig_fitness.to_html(include_plotlyjs="cdn"))
        }, step=n_eval)
    else:
        # Save graphs to files
        if file_logger._logger:
            graphs_dir = os.path.join(file_logger._logger.run_dir, "graphs")
            os.makedirs(graphs_dir, exist_ok=True)

            # Save plotly figures as HTML
            fig_solution.write_html(os.path.join(graphs_dir, f"solution_graph_n_eval_{n_eval}.html"))
            fig_fitness.write_html(os.path.join(graphs_dir, f"fitness_graph_n_eval_{n_eval}.html"))


def _create_solution_colored_graph(G, pos):
    """Create graph colored by solution_id."""
    solution_ids = [G.nodes[node]['solution_id'] for node in G.nodes]
    solution_id_counts = Counter(solution_ids)

    single_occurrence_ids = {sol for sol, count in solution_id_counts.items() if count == 1}
    multiple_occurrence_ids = {sol for sol, count in solution_id_counts.items() if count > 1}

    multiple_occurrence_counts = [solution_id_counts[sol] for sol in multiple_occurrence_ids]
    distinct_colors, distinct_opacities = _generate_distinct_colors_with_saturation_and_opacity(multiple_occurrence_counts)

    color_map = {sol: "rgb(169, 169, 169)" for sol in single_occurrence_ids}
    opacity_map = {sol: 0.25 for sol in single_occurrence_ids}

    if distinct_colors:
        color_map.update({sol: distinct_colors[i] for i, sol in enumerate(multiple_occurrence_ids)})
        opacity_map.update({sol: distinct_opacities[i] for i, sol in enumerate(multiple_occurrence_ids)})

    node_colors = []
    node_opacities = []
    for node in G.nodes():
        solution_id = G.nodes[node]['solution_id']
        node_colors.append(color_map[solution_id])
        node_opacities.append(opacity_map[solution_id])

    return _create_plotly_figure(G, pos, node_colors, node_opacities,
                                "Task Graph Ordered by angle and L")


def _create_fitness_colored_graph(G, pos):
    """Create graph colored by fitness."""
    fitness_values = [G.nodes[node]['fitness'] for node in G.nodes()]
    fitness_values = [fit if fit is not None else -1 for fit in fitness_values]
    fitness_colors, fitness_opacities = _create_colors_and_opacities_based_on_fitness(fitness_values)

    return _create_plotly_figure(G, pos, fitness_colors, fitness_opacities,
                                "Task Graph Ordered by angle and L, colored by fitness")


def _create_plotly_figure(G, pos, node_colors, node_opacities, title):
    """Create a plotly figure with given parameters."""
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_x = []
    node_y = []
    node_hover_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_hover_text.append(f"Task ID: {node}<br>Solution ID: {G.nodes[node]['solution_id']}<br>Fitness: {G.nodes[node]['fitness']}<br>task_vec: ({x}, {y})")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition="top center",
        marker=dict(
            size=10,
            color=node_colors,
            opacity=node_opacities,
            line=dict(width=2)
        ),
        text=None,
        hoverinfo='text'
    )
    node_trace.hovertext = node_hover_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            width=1600,
            height=1200
        )
    )

    return fig


def _generate_distinct_colors_with_saturation_and_opacity(counts):
    """Generate distinct colors with saturation and opacity adjusted based on counts."""
    from colorsys import hsv_to_rgb

    if not counts:
        return [], []

    n = len(counts)
    hues = [i / n for i in range(n)]
    max_count = max(counts)
    min_count = min(counts)
    saturation_range = (0.4, 1.0)

    def scale_value(count, value_range):
        if max_count == min_count:
            return value_range[1]
        return value_range[0] + (count - min_count) / (max_count - min_count) * (value_range[1] - value_range[0])

    value = 1.0
    colors = []
    opacities = []
    opacity_range = (0.2, 1.0)
    for h, count in zip(hues, counts):
        saturation = scale_value(count, saturation_range)
        opacity = scale_value(count, opacity_range)
        r, g, b = hsv_to_rgb(h, saturation, value)
        colors.append(f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})")
        opacities.append(opacity)
    return colors, opacities


def _create_colors_and_opacities_based_on_fitness(fitness_values):
    """Create a color map and opacity values based on fitness values using a colormap (viridis)."""
    if not fitness_values:
        return [], []

    valid_fitness_values = [fit for fit in fitness_values if fit != float('-inf')]

    if valid_fitness_values:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        colormap = cm.get_cmap('viridis')

        colors = []
        opacities = []
        for fit in fitness_values:
            if fit > 0:
                color = mpl.colors.rgb2hex(colormap(norm(fit, clip=True)))
                opacity = norm(max(0.2, fit), clip=True)
            else:
                color = "#000000"
                opacity = 0.1
            colors.append(color)
            opacities.append(float(opacity))
    else:
        colors = ["#000000"] * len(fitness_values)
        opacities = [0.1] * len(fitness_values)
    return colors, opacities


def log_mtme_archive_to_wandb(archive, n_eval: int, use_wandb: bool = True):
    """Log MT-ME archive data to wandb as plots and tables"""
    import pandas as pd

    if not archive:
        return

    # Extract data from archive
    data_rows = []
    task_id = 0
    for centroid_key, species in archive.items():
        task_vec = species.desc.task_vec
        if len(task_vec) == 12:
            x = np.sum(task_vec[:6])
            y = np.sum(task_vec[6:])
        elif len(task_vec) == 36:
            x = np.sum(task_vec[:18])
            y = np.sum(task_vec[18:])
        else:
            x, y = task_vec[0], task_vec[1] if len(task_vec) > 1 else (task_vec[0], 0)

        data_rows.append({
            "centroid_x": centroid_key[0] if len(centroid_key) > 0 else x,
            "centroid_y": centroid_key[1] if len(centroid_key) > 1 else y,
            "task_x": x,
            "task_y": y,
            "task_vec": task_vec,
            "fitness": species.fitness,
            "solution_dim": len(species.x),
            "solution": species.x,
            "task_id": task_id
        })
        task_id += 1

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # sort DataFrame by task_x and task_y
    df = df.sort_values(by=['task_x', 'task_y']).reset_index(drop=True)

    # Create wandb table
    if use_wandb and WANDB_AVAILABLE:
        table = wandb.Table(dataframe=df)
        wandb.log({f"mtme_archive/n_eval_{n_eval}": table}, step=n_eval)
    else:
        # Save to CSV file
        table_file = os.path.join(file_logger._logger.run_dir if file_logger._logger else "logs",
                                 f"mtme_archive_n_eval_{n_eval}.csv")
        df.to_csv(table_file, index=False)

    # Create visualizations
    _create_mtme_archive_plots(df, n_eval, use_wandb)


def _create_mtme_archive_plots(df, n_eval, use_wandb: bool = True):
    """Create MT-ME archive visualization plots."""
    if df.empty:
        return

    # Create fitness-colored scatter plot
    fig_fitness = _create_mtme_fitness_plot(df)

    # Create task-ID colored scatter plot
    fig_tasks = _create_mtme_task_plot(df)

    # Create centroid vs task space comparison
    fig_comparison = _create_mtme_comparison_plot(df)

    # Save or log the plots
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "mtme_archive_fitness_html": wandb.Html(fig_fitness.to_html(include_plotlyjs="cdn")),
            "mtme_archive_tasks_html": wandb.Html(fig_tasks.to_html(include_plotlyjs="cdn")),
            "mtme_archive_comparison_html": wandb.Html(fig_comparison.to_html(include_plotlyjs="cdn"))
        }, step=n_eval)
    else:
        # Save plots to files
        if file_logger._logger:
            plots_dir = os.path.join(file_logger._logger.run_dir, "mtme_plots")
            os.makedirs(plots_dir, exist_ok=True)

            fig_fitness.write_html(os.path.join(plots_dir, f"archive_fitness_n_eval_{n_eval}.html"))
            fig_tasks.write_html(os.path.join(plots_dir, f"archive_tasks_n_eval_{n_eval}.html"))
            fig_comparison.write_html(os.path.join(plots_dir, f"archive_comparison_n_eval_{n_eval}.html"))


def _create_mtme_fitness_plot(df):
    """Create scatter plot colored by fitness values."""
    fitness_values = df['fitness'].values
    # colors, opacities = _create_colors_and_opacities_based_on_fitness(fitness_values)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['task_x'],
        y=df['task_y'],
        mode='markers',
        marker=dict(
            size=8,
            color=fitness_values,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Fitness"),
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=[f"Task ID: {tid}<br>Fitness: {fit:.4f}<br>Centroid: ({cx:.2f}, {cy:.2f})"
              for tid, fit, cx, cy in zip(df['task_id'], df['fitness'], df['centroid_x'], df['centroid_y'])],
        hoverinfo='text',
        name='Archive Solutions'
    ))

    fig.update_layout(
        title="MT-ME Archive - Fitness Distribution",
        xaxis_title="Task X",
        yaxis_title="Task Y",
        width=800,
        height=600
    )

    return fig


def _create_mtme_task_plot(df):
    """Create scatter plot colored by task ID."""
    unique_tasks = df['task_id'].unique()
    n_tasks = len(unique_tasks)

    # Generate distinct colors for each task
    colors = [f"hsl({i * 360 / n_tasks}, 70%, 50%)" for i in range(n_tasks)]
    color_map = {task_id: colors[i] for i, task_id in enumerate(unique_tasks)}

    fig = go.Figure()

    for task_id in unique_tasks:
        task_data = df[df['task_id'] == task_id]
        fig.add_trace(go.Scatter(
            x=task_data['task_x'],
            y=task_data['task_y'],
            mode='markers',
            marker=dict(
                size=8,
                color=color_map[task_id],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"Task ID: {tid}<br>Fitness: {fit:.4f}<br>Centroid: ({cx:.2f}, {cy:.2f})"
                  for tid, fit, cx, cy in zip(task_data['task_id'], task_data['fitness'],
                                             task_data['centroid_x'], task_data['centroid_y'])],
            hoverinfo='text',
            name=f'Task {task_id}'
        ))

    fig.update_layout(
        title="MT-ME Archive - Task Distribution",
        xaxis_title="Task X",
        yaxis_title="Task Y",
        width=800,
        height=600
    )

    return fig


def _create_mtme_comparison_plot(df):
    """Create comparison plot showing centroid space vs task space."""
    fig = go.Figure()

    # Task space points
    fig.add_trace(go.Scatter(
        x=df['task_x'],
        y=df['task_y'],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            opacity=0.6,
            symbol='circle'
        ),
        name='Task Space',
        text=[f"Task Space<br>Task ID: {tid}<br>Fitness: {fit:.4f}"
              for tid, fit in zip(df['task_id'], df['fitness'])],
        hoverinfo='text'
    ))

    # Centroid space points
    fig.add_trace(go.Scatter(
        x=df['centroid_x'],
        y=df['centroid_y'],
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            opacity=0.6,
            symbol='x'
        ),
        name='Centroid Space',
        text=[f"Centroid Space<br>Task ID: {tid}<br>Fitness: {fit:.4f}"
              for tid, fit in zip(df['task_id'], df['fitness'])],
        hoverinfo='text'
    ))

    fig.update_layout(
        title="MT-ME Archive - Task Space vs Centroid Space",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=800,
        height=600
    )

    return fig


def log_ptme_archive_to_wandb(archive, n_eval: int, use_wandb: bool = True):
    """Log PT-ME archive data to wandb as plots and tables"""
    import pandas as pd

    if not hasattr(archive, 'elites') or not archive.elites:
        return

    # Extract data from archive elites
    data_rows = []
    for idx, elite in enumerate(archive.elites):
        if elite is not None:
            situation = elite.get("situation", [])
            if len(situation) == 12:
                x = np.sum(situation[:6])
                y = np.sum(situation[6:])
            elif len(situation) == 36:
                x = np.sum(situation[:18])
                y = np.sum(situation[18:])
            else:
                x, y = situation[0], situation[1] if len(situation) > 1 else (situation[0], 0)

            centroid = archive.centroids[idx] if hasattr(archive, 'centroids') else [0, 0]

            data_rows.append({
                "elite_idx": idx,
                "centroid_x": centroid[0] if len(centroid) > 0 else x,
                "centroid_y": centroid[1] if len(centroid) > 1 else y,
                "situation_x": x,
                "situation_y": y,
                "situation_vec": situation,
                "command": elite.get("command", []),
                "fitness": elite.get("reward", float('-inf')),
                "kind": elite.get("kind", "unknown"),
                "iteration": elite.get("it", -1)
            })

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    if df.empty:
        return

    # Sort DataFrame by situation_x and situation_y
    df = df.sort_values(by=['situation_x', 'situation_y']).reset_index(drop=True)

    # Create wandb table
    if use_wandb and WANDB_AVAILABLE:
        table = wandb.Table(dataframe=df)
        wandb.log({f"ptme_archive/n_eval_{n_eval}": table}, step=n_eval)
    else:
        # Save to CSV file
        table_file = os.path.join(file_logger._logger.run_dir if file_logger._logger else "logs",
                                 f"ptme_archive_n_eval_{n_eval}.csv")
        df.to_csv(table_file, index=False)

    # Create visualizations
    _create_ptme_archive_plots(df, n_eval, use_wandb)


def _create_ptme_archive_plots(df, n_eval, use_wandb: bool = True):
    """Create PT-ME archive visualization plots."""
    if df.empty:
        return

    # Create fitness-colored scatter plot
    fig_fitness = _create_ptme_fitness_plot(df)

    # Create kind-colored scatter plot
    fig_kinds = _create_ptme_kind_plot(df)

    # Create centroid vs situation space comparison
    fig_comparison = _create_ptme_comparison_plot(df)

    # Save or log the plots
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "ptme_archive_fitness_html": wandb.Html(fig_fitness.to_html(include_plotlyjs="cdn")),
            "ptme_archive_kinds_html": wandb.Html(fig_kinds.to_html(include_plotlyjs="cdn")),
            "ptme_archive_comparison_html": wandb.Html(fig_comparison.to_html(include_plotlyjs="cdn"))
        }, step=n_eval)
    else:
        # Save plots to files
        if file_logger._logger:
            plots_dir = os.path.join(file_logger._logger.run_dir, "ptme_plots")
            os.makedirs(plots_dir, exist_ok=True)

            fig_fitness.write_html(os.path.join(plots_dir, f"archive_fitness_n_eval_{n_eval}.html"))
            fig_kinds.write_html(os.path.join(plots_dir, f"archive_kinds_n_eval_{n_eval}.html"))
            fig_comparison.write_html(os.path.join(plots_dir, f"archive_comparison_n_eval_{n_eval}.html"))


def _create_ptme_fitness_plot(df):
    """Create scatter plot colored by fitness values for PT-ME archive."""
    fitness_values = df['fitness'].values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['situation_x'],
        y=df['situation_y'],
        mode='markers',
        marker=dict(
            size=8,
            color=fitness_values,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Fitness"),
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=[f"Elite {idx}<br>Fitness: {fit:.4f}<br>Kind: {kind}<br>Centroid: ({cx:.2f}, {cy:.2f})"
              for idx, fit, kind, cx, cy in zip(df['elite_idx'], df['fitness'], df['kind'],
                                                df['centroid_x'], df['centroid_y'])],
        hoverinfo='text',
        name='Archive Elites'
    ))

    fig.update_layout(
        title="PT-ME Archive - Fitness Distribution",
        xaxis_title="Situation X",
        yaxis_title="Situation Y",
        width=800,
        height=600
    )

    return fig


def _create_ptme_kind_plot(df):
    """Create scatter plot colored by variation kind."""
    unique_kinds = df['kind'].unique()
    n_kinds = len(unique_kinds)

    # Generate distinct colors for each kind
    colors = [f"hsl({i * 360 / n_kinds}, 70%, 50%)" for i in range(n_kinds)]
    color_map = {kind: colors[i] for i, kind in enumerate(unique_kinds)}

    fig = go.Figure()

    for kind in unique_kinds:
        kind_data = df[df['kind'] == kind]
        fig.add_trace(go.Scatter(
            x=kind_data['situation_x'],
            y=kind_data['situation_y'],
            mode='markers',
            marker=dict(
                size=8,
                color=color_map[kind],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"Elite {idx}<br>Fitness: {fit:.4f}<br>Kind: {k}<br>Centroid: ({cx:.2f}, {cy:.2f})"
                  for idx, fit, k, cx, cy in zip(kind_data['elite_idx'], kind_data['fitness'],
                                                 kind_data['kind'], kind_data['centroid_x'],
                                                 kind_data['centroid_y'])],
            hoverinfo='text',
            name=f'{kind}'
        ))

    fig.update_layout(
        title="PT-ME Archive - Variation Kind Distribution",
        xaxis_title="Situation X",
        yaxis_title="Situation Y",
        width=800,
        height=600
    )

    return fig


def _create_ptme_comparison_plot(df):
    """Create comparison plot showing centroid space vs situation space."""
    fig = go.Figure()

    # Situation space points
    fig.add_trace(go.Scatter(
        x=df['situation_x'],
        y=df['situation_y'],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            opacity=0.6,
            symbol='circle'
        ),
        name='Situation Space',
        text=[f"Situation Space<br>Elite {idx}<br>Fitness: {fit:.4f}"
              for idx, fit in zip(df['elite_idx'], df['fitness'])],
        hoverinfo='text'
    ))

    # Centroid space points
    fig.add_trace(go.Scatter(
        x=df['centroid_x'],
        y=df['centroid_y'],
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            opacity=0.6,
            symbol='x'
        ),
        name='Centroid Space',
        text=[f"Centroid Space<br>Elite {idx}<br>Fitness: {fit:.4f}"
              for idx, fit in zip(df['elite_idx'], df['fitness'])],
        hoverinfo='text'
    ))

    fig.update_layout(
        title="PT-ME Archive - Situation Space vs Centroid Space",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=800,
        height=600
    )

    return fig