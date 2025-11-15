import fastf1 as ff1
import pandas as pd

# Example script for fastf1 data loading

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    # Set up cache to not re-download data
    ff1.Cache.enable_cache("fastf1/cache")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 240)
    pd.set_option('display.max_rows', 50)

    YEAR = 2024
    EVENT = "Silverstone"
    SESSION = "R"

    # Load Session
    print_section("LOADING SESSION")
    session = ff1.get_session(YEAR, EVENT, SESSION)
    session.load()

    # Session Info
    print_section("SESSION INFO")
    print("Event:\n", session.event)
    print("\nSession Name:", session.name)
    print("Session Data:", session.date)
    print("Scheduled total laps:", session.total_laps)
    print("Drivers:", session.drivers)

    print("INFO:")
    print(session.session_info)

    # Session results / classification
    print_section("SESSION RESULTS / CLASSIFICATION")
    results = session.results
    print("Shape:", results.shape)
    print("Columns:", list(results.columns))
    print(results)

    # Laps
    print_section("LAPS DATA")
    laps = session.laps
    print("Shape:", laps.shape)
    print("Columns:", list(laps.columns))
    print(laps.head(20)) # first 20 laps

    # Laps for a single driver
    first_driver = results["Abbreviation"].iloc[0]
    driver_laps = laps.pick_driver(first_driver)
    print_section(f"LAPS DATA FOR DRIVER: {first_driver}")
    print("Shape:", driver_laps.shape)
    print("Columns:", list(driver_laps.columns))
    print(driver_laps.head(20))

    # Weather data
    print_section("WEATHER DATA")
    weather = session.weather_data
    print("Shape:", weather.shape)
    print("Columns:", list(weather.columns))
    print(weather.head(20))

    # Weather per lap
    print_section("WEATHER DATA PER LAP")
    weather_per_laps = laps.get_weather_data()
    print("Shape:", weather_per_laps.shape)
    print("Columns:", list(weather_per_laps.columns))
    print(weather_per_laps.head(20))

    # Track Status
    print_section("TRACK STATUS (Yellow/SC/VSC)")
    track_status = session.track_status
    print("Shape:", track_status.shape)
    print("Columns:", list(track_status.columns))
    print(track_status.head(20))

    # Session Status
    print_section("SESSION STATUS (Red/Green Flags)")
    session_status = session.session_status
    print("Shape:", session_status.shape)
    print("Columns:", list(session_status.columns))
    print(session_status.head(20))

    # Race control messages
    print_section("RACE CONTROL MESSAGES")
    rcm = session.race_control_messages
    print("Shape:", rcm.shape)
    print("Columns:", list(rcm.columns))
    print(rcm.head(20))

    # Circuit info
    print_section("CIRCUIT INFO")
    circuit_info = session.get_circuit_info()
    print(circuit_info)

    # Telemetry Data
    print_section("TELEMETRY CAR DATA")
    car_data = session.car_data
    print("Drivers with car data:", list(car_data.keys()))
    if car_data:
        example_driver_num = list(car_data.keys())[0]
        cd = car_data[example_driver_num]
        print(f"\nExample car data for driver number {example_driver_num}")
        print("Shape:", cd.shape)
        print("Columns:", list(cd.columns))
        print(cd.head(20))

    print_section("TELEMETRY POSITION DATA ")
    pos_data = session.pos_data
    print("Drivers with position data:", list(pos_data.keys()))
    if pos_data:
        example_driver_num = list(pos_data.keys())[0]
        pd_pos = pos_data[example_driver_num]
        print(f"\nExample position data for driver number {example_driver_num}")
        print("Shape:", pd_pos.shape)
        print("Columns:", list(pd_pos.columns))
        print(pd_pos.head(20))

    # Example full telemetry and weather
    print_section("FULL TELEMETRY WITH WEATHER FOR A DRIVER")
    fastest_lap = driver_laps.pick_fastest()
    print("Fastest lap row:")
    print(fastest_lap)

    telemetry = fastest_lap.get_telemetry()
    print("\nTelemetry shape:", telemetry.shape)
    print("Telemetry columns:", list(telemetry.columns))
    print(telemetry.head(20))

    lap_weather = fastest_lap.get_weather_data()
    print("\nWeather on that lap:")
    print(lap_weather)

if __name__ == "__main__":
    main()
