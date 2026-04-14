import os
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

COMPOUND_COLOURS = {1: "#E8002D", 2: "#FFC700", 3: "#EEEEEE"}
COMPOUND_NAMES = {1: "Soft", 2: "Medium", 3: "Hard"}

AGENT_COLOUR = "#00FF00"
TARGET_COLOUR = "#DD00FF"

# 2024 F1 team colours
TEAM_COLOURS = {
    "VER": "#3671C6", "PER": "#3671C6",
    "HAM": "#27F4D2", "RUS": "#27F4D2",
    "LEC": "#E8002D", "SAI": "#E8002D", "BEA": "#E8002D",
    "NOR": "#FF8000", "PIA": "#FF8000",
    "ALO": "#229971", "STR": "#229971",
    "GAS": "#2293D1", "OCO": "#2293D1", "DOO": "#2293D1",
    "ALB": "#64C4FF", "SAR": "#64C4FF", "COL": "#64C4FF",
    "TSU": "#6692FF", "RIC": "#6692FF", "LAW": "#6692FF",
    "BOT": "#C92D4B", "ZHO": "#C92D4B",
    "MAG": "#B6BABD", "HUL": "#B6BABD",
}

# Helpers
def get_safety_car_periods(race_data: Dict) -> List[Tuple[int, int]]:
    return [(int(event["start_lap"]), int(event["end_lap"])) for event in race_data.get("sc_events", [])]

def set_dark_theme(enabled: bool = True) -> None:
    if enabled:
        plt.rcParams.update({
            "figure.facecolor": "#121212",
            "axes.facecolor": "#121212",
            "axes.edgecolor": "#DDDDDD",
            "axes.labelcolor": "#EEEEEE",
            "axes.titlecolor": "#FFFFFF",
            "xtick.color": "#DDDDDD",
            "ytick.color": "#DDDDDD",
            "grid.color": "#666666",
            "text.color": "#FFFFFF",
            "legend.facecolor": "#1E1E1E",
            "legend.edgecolor": "#777777",
            "savefig.facecolor": "#121212",
            "savefig.edgecolor": "#121212",
        })


# Build list of compounds used by agent in each stint based on episode log and pit stops
def build_agent_stint_compounds(episode_log: List[Dict], agent_entries: List[Dict]) -> List[int]:
    compounds = []

    first_entry = next((e for e in episode_log if e["lap"] >= 0), None)

    if first_entry:
        compounds.append(first_entry.get("compound", 2))

    for entry in agent_entries:
        if entry.get("pitted"):
            compounds.append(entry["compound"])

    return compounds or [2]

# Draws a horizontal bar representing tyre compounds used in a stint
def draw_tyre_bar(ax, row_y: float, compounds: List[int], pit_laps: List[int], total_laps: int, bar_height: float = 0.6) -> None:
    segments = []
    stint_start = 1
    compound_index = 0
    current_compound = compounds[0] if compounds else 2

    for pit_lap in pit_laps:
        segments.append((stint_start, pit_lap, current_compound))
        compound_index += 1
        if compound_index < len(compounds):
            current_compound = compounds[compound_index]
        stint_start = pit_lap

    segments.append((stint_start, total_laps, current_compound))

    for start, end, compound_id in segments:
        width = end - start + 1
        colour = COMPOUND_COLOURS.get(compound_id, "#888888")
        edge_colour = "#444444" if compound_id == 3 else "none"
        ax.barh(row_y, width, left=start, height=bar_height, color=colour, edgecolor=edge_colour, linewidth=0.5)

# Combines tyre bars and safety car shading for the race timeline
def draw_tyre_timeline(
    ax,
    safety_car_periods: List[Tuple[int, int]],
    agent_compounds: List[int],
    agent_pit_laps: List[int],
    target_compounds: List[int],
    target_pit_laps: List[int],
    total_laps: int,
    target_code: str
) -> None:

    for start, end in safety_car_periods:
        ax.axvspan(start - 0.5, end + 0.5, color="#FFA500", alpha=0.15, zorder=0)

    draw_tyre_bar(ax, 1, agent_compounds, agent_pit_laps, total_laps)
    draw_tyre_bar(ax, 0, target_compounds, target_pit_laps, total_laps)

    ax.set_yticks([0, 1])
    ax.set_yticklabels([target_code, "Agent"], fontsize=9)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0.5, total_laps + 0.5)
    ax.set_xlabel("Lap Number")
    ax.grid(False)

    compound_handles = [mpatches.Patch(color=COMPOUND_COLOURS[c], label=COMPOUND_NAMES[c]) for c in (1, 2, 3)]
    ax.legend(handles=compound_handles, loc="upper right", fontsize=7, ncol=3)

# Target driver comparison plot
def plot_race_comparison(episode_log: List[Dict], race_data: Dict, output_dir: str, dark_theme: bool = True) -> None:
    set_dark_theme(dark_theme)

    target = race_data["target_driver_strategy"]
    target_code = target.get("driver_code", "HAM")
    race_name = race_data.get("name", "Race")
    short_name = race_name.replace(" Grand Prix", " GP")
    total_laps = race_data["track"]["total_laps"]

    # Agent data
    agent_entries = [e for e in episode_log if e["lap"] > 0 and e["lap_time"]]
    agent_times = np.array([e["lap_time"] for e in agent_entries])
    agent_pit_laps = [e["lap"] for e in agent_entries if e.get("pitted")]

    # Target data
    target_times = np.array(target["lap_times"])
    target_pit_laps = [int(lap) for lap in target.get("pit_laps", [])]

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08, figure=fig)
    ax_delta = fig.add_subplot(gs[0])
    ax_stints = fig.add_subplot(gs[1], sharex=ax_delta)

    # Cumulative time for plot
    num_common_laps = min(len(agent_times), len(target_times))
    lap_deltas = agent_times[:num_common_laps] - target_times[:num_common_laps]
    cumulative_delta = np.cumsum(lap_deltas)
    common_lap_nums = np.arange(1, num_common_laps + 1)

    # Safety car shading
    safety_car_periods = get_safety_car_periods(race_data)
    for start, end in safety_car_periods:
        ax_delta.axvspan(start - 0.5, end + 0.5, color="#FFA500", alpha=0.15, zorder=0)

    ax_delta.axhline(0, color="white", linewidth=0.5, alpha=0.4, zorder=1)

    # Fill: green where agent gained time, red where agent lost time
    ax_delta.fill_between(
        common_lap_nums, cumulative_delta, 0,
        where=cumulative_delta <= 0, color="#00FF00", alpha=0.3, interpolate=True, zorder=2,
    )
    ax_delta.fill_between(
        common_lap_nums, cumulative_delta, 0,
        where=cumulative_delta > 0, color="#E8002D", alpha=0.3, interpolate=True, zorder=2,
    )
    ax_delta.plot(common_lap_nums, cumulative_delta, color="white", linewidth=1.8, zorder=5)

    # Pit lap markers
    for pit_lap in agent_pit_laps:
        ax_delta.axvline(x=pit_lap, color="#FFFFFF", linewidth=1.5, linestyle="-", alpha=0.5)
    for pit_lap in target_pit_laps:
        ax_delta.axvline(x=pit_lap, color="#FFFFFF", linewidth=1.5, linestyle="-", alpha=0.5)

    total_delta = float(cumulative_delta[-1]) if len(cumulative_delta) > 0 else 0.0
    sign = "+" if total_delta >= 0 else ""

    ax_delta.set_title(f"Cumulative Time Delta vs {target_code} — {short_name}  (Δ {sign}{total_delta:.1f}s)",
                       fontsize=13, weight="bold")
    ax_delta.set_ylabel(f"Cumulative Δ vs {target_code} (s)")
    ax_delta.set_xlim(0.5, total_laps + 0.5)
    ax_delta.grid(True, alpha=0.25)
    plt.setp(ax_delta.get_xticklabels(), visible=False)
    legend_handles = [
        mpatches.Patch(color="#00FF00", alpha=0.4, label="Time gained"),
        mpatches.Patch(color="#E8002D", alpha=0.4, label="Time lost"),
    ]
    if safety_car_periods:
        legend_handles.append(mpatches.Patch(color="#FFA500", alpha=0.3, label="Safety Car"))
    ax_delta.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # Bottom tyre timeline
    agent_stint_compounds = build_agent_stint_compounds(episode_log, agent_entries)
    draw_tyre_timeline(
        ax_stints, safety_car_periods,
        agent_stint_compounds, agent_pit_laps,
        target.get("pit_compounds", []), target_pit_laps,
        total_laps, target_code,
    )

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"pace_comparison_{race_name.replace(' ', '_').lower()}.png"
    fig.savefig(os.path.join(output_dir, file_name), dpi=200)
    plt.close(fig)

# Full Race Position Chart
def plot_position_chart(episode_log: List[Dict], race_data: Dict, output_dir: str, dark_theme: bool = True) -> None:
    set_dark_theme(dark_theme)

    target = race_data["target_driver_strategy"]
    target_code = target.get("driver_code", "HAM")
    race_name = race_data.get("name", "Race")
    short_name = race_name.replace(" Grand Prix", " GP")
    total_laps = race_data["track"]["total_laps"]
    safety_car_periods = get_safety_car_periods(race_data)

    drivers = []
    for opponent in race_data.get("opponents", []):
        code = opponent["driver_code"]
        if code == target_code:
            continue
        positions = opponent["positions"]
        num_laps = min(len(positions), total_laps)
        drivers.append((
            code,
            np.arange(1, num_laps + 1),
            np.array(positions[:num_laps], dtype=float),
            "other",
        ))

    # Target driver
    target_positions = target.get("positions", [])
    target_num_laps = min(len(target_positions), total_laps)
    target_pit_laps = [int(lap) for lap in target.get("pit_laps", [])]
    drivers.append((target_code, np.arange(1, target_num_laps + 1), np.array(target_positions[:target_num_laps], dtype=float), "target"))

    # Agent
    agent_entries = [e for e in episode_log if e["lap"] > 0]
    agent_lap_nums = np.array([e["lap"] for e in agent_entries])
    agent_positions = np.array([e["position"] for e in agent_entries], dtype=float)
    agent_pit_laps = [e["lap"] for e in agent_entries if e.get("pitted")]
    drivers.append(("Agent", agent_lap_nums, agent_positions, "agent"))

    num_grid_positions = len(drivers) - 1

    fig = plt.figure(figsize=(14, max(7, num_grid_positions * 0.35 + 1.5)))
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.08, figure=fig)
    ax_positions = fig.add_subplot(gs[0])
    ax_stints = fig.add_subplot(gs[1], sharex=ax_positions)

    # Safety car shading
    for start, end in safety_car_periods:
        ax_positions.axvspan(start - 0.5, end + 0.5, color="#FFA500", alpha=0.12, zorder=0)

    # Plot each driver with colouring
    for code, lap_nums, positions, role in drivers:
        if role == "agent":
            ax_positions.plot(lap_nums, positions, color=AGENT_COLOUR, linewidth=2.8, zorder=10)
        elif role == "target":
            ax_positions.plot(lap_nums, positions, color=TARGET_COLOUR, linewidth=2.8, zorder=9)
        else:
            team_colour = TEAM_COLOURS.get(code, "#888888")
            ax_positions.plot(lap_nums, positions, color=team_colour, linewidth=1.5, alpha=0.2, zorder=2)
    
    # Pit lap markers
    for pit_lap in agent_pit_laps:
        ax_positions.axvline(x=pit_lap, color="#FFFFFF", linewidth=1.5, linestyle="-", alpha=0.5)
    for pit_lap in target_pit_laps:
        ax_positions.axvline(x=pit_lap, color="#FFFFFF", linewidth=1.5, linestyle="-", alpha=0.5)

    ax_positions.set_ylim(num_grid_positions + 0.5, 0.5)
    ax_positions.set_yticks(range(1, num_grid_positions + 1))
    ax_positions.set_xlim(0.5, total_laps + 0.5)
    ax_positions.set_ylabel("Position")
    ax_positions.set_title(f"Race Positions — {short_name}", fontsize=13, weight="bold")
    ax_positions.grid(True, alpha=0.2)
    
    plt.setp(ax_positions.get_xticklabels(), visible=False)
    legend_handles = [
        plt.Line2D([], [], color=AGENT_COLOUR, linewidth=2.8, label="Agent"),
        plt.Line2D([], [], color=TARGET_COLOUR, linewidth=2.8, label=target_code),
    ]
    if safety_car_periods:
        legend_handles.append(mpatches.Patch(color="#FFA500", alpha=0.3, label="Safety Car"))
    ax_positions.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # Bottom tyre timeline
    agent_stint_compounds = build_agent_stint_compounds(episode_log, agent_entries)
    draw_tyre_timeline(
        ax_stints, safety_car_periods,
        agent_stint_compounds, agent_pit_laps,
        target.get("pit_compounds", []), target_pit_laps,
        total_laps, target_code,
    )

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"positions_{race_name.replace(' ', '_').lower()}.png"
    fig.savefig(os.path.join(output_dir, file_name), dpi=200)
    plt.close(fig)