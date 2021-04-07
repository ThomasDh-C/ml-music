import json

# https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-analysis

f = open('audioanalysis.json')
obj = json.load(f)
# obj is a dict {meta, track, bars, beats, sections, segments, tatums}
for i in obj:
    print(i)

# meta is kinda uselss for us
# print(obj['meta']) 

# track
# obj['track'].keys()
# ['num_samples', 'duration', 'sample_md5', 'offset_seconds', 'window_seconds', 
# 'analysis_sample_rate', 'analysis_channels', 'end_of_fade_in', 'start_of_fade_out', 
# 'loudness', 'tempo', 'tempo_confidence', 'time_signature', 'time_signature_confidence', 
# 'key', 'key_confidence', 'mode', 'mode_confidence', 'codestring', 'code_version', 
# 'echoprintstring', 'echoprint_version', 'synchstring', 'synch_version', 'rhythmstring', 
# 'rhythm_version']

# bars
# array of dicts {'start': xx, 'duration': yyy, 'confidence': zz}

# beats
# array of dicts {'start': xx, 'duration': yyy, 'confidence': zz}

# sections
# array of dicts
# {'start': 150.0695,
#   'duration': 57.89036,
#   'confidence': 0.701,
#   'loudness': -1.024,
#   'tempo': 114.986,
#   'tempo_confidence': 0.39,
#   'key': 2,
#   'key_confidence': 0.043,
#   'mode': 1,
#   'mode_confidence': 0.382,
#   'time_signature': 4,
#   'time_signature_confidence': 1.0}

# segments
# array of dicts
# {'start': 202.97769,
#   'duration': 4.98218,
#   'confidence': 0.054,
#   'loudness_start': -23.839,
#   'loudness_max_time': 0.05905,
#   'loudness_max': -21.116,
#   'loudness_end': -60.0,
#   'pitches': [ array of floats 0 -1 ],
#   'timbre': [ array of positive and negative floats]}

# tatums
# array of dicts {'start': xxx, 'duration': yyy, 'confidence': zzz}