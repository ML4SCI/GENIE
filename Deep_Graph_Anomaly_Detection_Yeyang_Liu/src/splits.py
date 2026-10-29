import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SIGNAL_COUNTS = (0, 50, 100, 500, 1000)
SR_EVENT_CAP = 120_000
TEST_SIGNAL_CAP = 20_000
# KS extra-QCD SR order: test reserve, IAD reference, remaining test background.
EXTRA_QCD_TEST_RESERVE = 40_000
EXTRA_QCD_REFERENCE_STOP = 312_858
EVENT_DTYPE = np.dtype([
    ('source', 'u1'), ('row', 'i8'), ('mjj', 'f4'),
    ('truth', 'i1'), ('weak_label', 'i1'),
])


def dijet_mass(features):
    """Compute mJJ in TeV; KS casts to float32 before applying the mass cuts."""
    px1, py1, pz1, m1 = features[['pxj1', 'pyj1', 'pzj1', 'mj1']].to_numpy(dtype=np.float64).T
    px2, py2, pz2, m2 = features[['pxj2', 'pyj2', 'pzj2', 'mj2']].to_numpy(dtype=np.float64).T
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    mass = np.sqrt((e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)
    return (mass * 1e-3).astype(np.float32)


def read_events(path, source, *, background=False):
    """Read masses and truth from a published feature table, preserving row order."""
    features = pd.read_hdf(path)
    events = np.zeros(len(features), dtype=EVENT_DTYPE)
    events['source'] = source
    events['row'] = np.arange(len(features))
    events['mjj'] = dijet_mass(features)
    if not background:
        events['truth'] = features['label'].to_numpy(dtype=np.float64).astype(np.int8)
    events['weak_label'] = -1
    return events


def signal_region(events):
    return (events['mjj'] > 3.3) & (events['mjj'] < 3.7)


def mixture_seed(run, n_signal=1000):
    seed = 1 + n_signal
    for step in range(1, run + 1):
        seed += step + n_signal
    return seed


def make_samples(background, signal, extra_background, *, mode='CWoLa', n_signal=1000, run=0):
    """Inject n_signal BEFORE the mass cuts and the 120000-event SR cap.
    Weak labels: SR=1, reference=0; truth: background=0, signal=1.
    Test events have weak_label=-1.
    """

    assert n_signal in SIGNAL_COUNTS, f'n_signal must be one of {SIGNAL_COUNTS}'

    signal = signal.copy()
    np.random.RandomState(934 + n_signal + run).shuffle(signal)
    mixture = np.concatenate([background, signal[:n_signal]])
    rng = np.random.RandomState(mixture_seed(run, n_signal))
    rng.shuffle(mixture)

    in_sr = signal_region(mixture)
    sr = mixture[in_sr][:SR_EVENT_CAP].copy()
    sideband = mixture[~in_sr & (mixture['mjj'] > 3.3 - 0.2) & (mixture['mjj'] < 3.7 + 0.2)]

    extra_sr = extra_background[signal_region(extra_background)]
    reserved_test_background = extra_sr[:EXTRA_QCD_TEST_RESERVE]
    iad_reference = extra_sr[EXTRA_QCD_TEST_RESERVE:EXTRA_QCD_REFERENCE_STOP]
    remaining_test_background = extra_sr[EXTRA_QCD_REFERENCE_STOP:]
    reference = {'IAD': iad_reference, 'CWoLa': sideband}[mode].copy()
    sr['weak_label'] = 1
    reference['weak_label'] = 0
    pool = np.concatenate([sr, reference])

    np.random.RandomState(rng.randint(300)).shuffle(pool)

    held_out_signal = signal[n_signal:]
    test_signal = held_out_signal[signal_region(held_out_signal)][:TEST_SIGNAL_CAP]
    test = np.concatenate([remaining_test_background, test_signal, reserved_test_background])
    return pool, test


def make_splits(background, signal, extra_background, *, mode='CWoLa', n_signal=1000):
    """Split the training pool in half using the KS run-0/member-0 split seed."""
    pool, test = make_samples(background, signal, extra_background, mode=mode, n_signal=n_signal)
    split_seed = np.random.RandomState(1 + n_signal).randint(np.iinfo(np.uint32).max, dtype='u8')
    train, validation = train_test_split(
        pool, test_size=0.5, stratify=pool['weak_label'], random_state=split_seed,
    )
    return {'train': train, 'validation': validation, 'test': test}
