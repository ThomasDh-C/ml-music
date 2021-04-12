import py_midicsv as pm

# Load the MIDI file and parse it into CSV format
csv_string = pm.midi_to_csv("augst.mid")

with open("example_converted.csv", "w") as f:
    f.writelines(csv_string)