import os

import music21.key
from music21.analysis import neoRiemannian

from tqdm import tqdm

class Chord:
    def __init__(self, root, is_major):
        self.root = root
        self.is_major = is_major

    def __str__(self):
        if self.is_major:
            return f'{self.root} major'
        return f'{self.root} minor'



def parse_emopia(directory) -> dict:
    files = {}
    for file in os.listdir(directory):
        filename, ext = file.split('.')
        if ext == 'mid':
            if filename not in files.keys():
                files[filename] = []
            files[filename].append('.'.join([filename, ext]))
    return files

def parse_midi(path):
    midi_file = music21.converter.parse(path)
    chords = midi_file.chordify()
    chords = [chord for chord in chords.recurse().getElementsByClass('Chord')]
    ok = 0
    for chord in chords:
        try:
            left = neoRiemannian.L(chord, True)  # second param is raiseException
            right = neoRiemannian.R(chord, True)
            parallel = neoRiemannian.P(chord, True)
            # print(f'{chord.pitchedCommonName}, {left.pitchedCommonName}, '
                  # f'{right.pitchedCommonName}, {parallel.pitchedCommonName}')
            ok += 1
        except Exception as e:
            pass
            # print('unable to transform the chord!')
    return ok, len(chords)

def parse_track(files, dirname):
    ok, total = 0, 0
    for filename in files:
        res = parse_midi('/'.join([dirname, filename]))
        ok += res[0]
        total += res[1]
    return ok, total


def pipeline(dirname, output):
    with open(output, "w") as f:
        files = parse_emopia(dirname)
        for file in tqdm(files.keys()):
            midis = files[file]
            ok, total = parse_track(midis, dirname)
            f.write(';'.join([file, str(ok), str(total)]))
            f.write('\n')



