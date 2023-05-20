import music21
from music21 import corpus


def test_chordify():
    DATA_PATH = 'EMOPIA_1.0/midis/'
    TEST_FILE = 'Q1_0vLPYiPN7qY_0.mid'

    # def process(filename = DATA_PATH + TEST_FILE):
    #     midi = MIDI.MIDIFile(filename)
    #     midi.parse()
    #     print(str(midi))
    #     for idx, track in enumerate(midi):
    #         track.parse()
    #         print(f'Track {idx}: {str(track)}')
    #
    #
    # process()

    import music21
    from music21 import corpus

    # Load MIDI file
    # midi_file = music21.converter.parse(DATA_PATH + TEST_FILE)

    midi_file = corpus.parse('bwv66.6')
    chords = midi_file.chordify()

    roots = []
    for thisChord in chords.recurse().getElementsByClass('Chord'):
        root = thisChord.root()
        common_name = thisChord.commonName
        is_major = (common_name.find('major') != -1)
        is_minor = (common_name.find('minor') != -1)
        # print(thisChord.pitchedCommonName, thisChord.root())
        roots.append((root.name, is_major, is_minor))

    for root in roots:
        print(root)


def test_keys():
    midi_file = corpus.parse('bwv66.6')
    chords = midi_file.chordify()

    keys = []
    for thisChord in chords.recurse().getElementsByClass('Chord'):
        root = thisChord.root()
        key = music21.key.Key(root)
        keys.append(key)

    for key in keys:
        print(key)


def test_neo_riemannian():
    from music21.analysis import neoRiemannian

    DATA_PATH = 'EMOPIA_1.0/midis/'
    TEST_FILE = 'Q1_0vLPYiPN7qY_0.mid'
    midi_file = music21.converter.parse(DATA_PATH + TEST_FILE)

    # midi_file = corpus.parse('bwv66.6')
    chords = midi_file.chordify()
    chords = [chord for chord in chords.recurse().getElementsByClass('Chord')]
    ok = 0
    for chord in chords:
        try:
            left = neoRiemannian.L(chord, True)  # second param is raiseException
            right = neoRiemannian.R(chord, True)
            parallel = neoRiemannian.P(chord, True)
            print(f'{chord.pitchedCommonName}, {left.pitchedCommonName}, '
                  f'{right.pitchedCommonName}, {parallel.pitchedCommonName}')
            ok += 1
        except Exception as e:
            print('unable to transform the chord!')
    print(f'ok: {ok}/{len(chords)}')



if __name__ == '__main__':
    test_neo_riemannian()