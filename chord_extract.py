from chord_extractor.extractors import Chordino

chordino = Chordino(roll_on=1)

def chord_recognition(file_path: str):
  try:
    chord_data = chordino.extract(file_path)
    # chroma = dcp(file_path)
    # chord_data = dccrp(chroma)
    # chord_data = [{'start': start, 'end': end, 'name': name} for start, end, name in chord_data]
    chord_data = [{'start': chord_data[i].timestamp, 'end': chord_data[i + 1].timestamp if i + 1 < len(chord_data) else None, 'name': chord_data[i].chord} for i in range(len(chord_data))]
    chord_data = [chord for chord in chord_data if chord['name'] != 'N']
    print(chord_data)
    return chord_data
  except:
    print('chord_recognition failed::')
    return []