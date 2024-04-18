import mne.io
import numpy as np
from moabb.datasets.base import BaseDataset


def sub_name(n_s: int) -> str:
    if n_s < 10:
        num_s = 'sub-0' + str(n_s)
    else:
        num_s = 'sub-' + str(n_s)
    return num_s


def load_events(n_s: int, n_b: int):
    num_s = sub_name(n_s)
    file_name = f"eeg_data/imagined_speech_dataset/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_events.dat"
    events = np.load(file_name, allow_pickle=True)
    return events[:, 1]


class ThinkingOutLoudDataset(BaseDataset):
    """Thinking out loud dataset for MOABB
    URL: https://www.nature.com/articles/s41597-022-01147-2
    github: https://github.com/N-Nieto/Inner_Speech_Dataset
    """

    def __init__(self):
        super().__init__(
            subjects=[i for i in range(1, 11)],
            sessions_per_subject=3,
            events={"Abajo": 1, "Arriba": 2, "Derecha": 3, "Izquierda": 4},
            code="ThinkingOutLoudDataset",
            interval=[1, 3.5],
            paradigm="imagery",
            doi="https://doi.org/10.1038/s41597-022-01147-2",
        )

    def _get_single_subject_data(self, subject):
        if (subject < 1) or (subject > 10):
            raise ValueError("Subject must be between 1 and 10. Got %d." % subject)
        file_path_list = self.data_path(subject)
        sampling_freq = 256.0
        letters = ['A', 'B', 'C', 'D']
        numbers = range(1, 33)
        ch_names = [f'{letter}{number}' for letter in letters for number in numbers] + ["stim"]
        ch_types = ["eeg" for _ in range(128)] + ["stim"]
        sessions = {"0": {}, "1": {}, "2": {}}
        for idx, epoch_file in enumerate(file_path_list):
            epochs = mne.read_epochs(epoch_file).get_data()
            batch_size = epochs.shape[0]
            eeg_length = epochs.shape[-1]
            events = load_events(subject, idx + 1)
            for i in range(batch_size):
                if eeg_length == 0:
                    continue
                stim = np.ones([1, eeg_length]) * events[i]
                stim[0][0] = 0
                temp_epochs = np.concatenate((epochs[i], stim))
                info = mne.create_info(ch_names, sampling_freq, ch_types)
                eeg_mne = mne.io.RawArray(temp_epochs, info)
                eeg_mne.set_montage(mne.channels.make_standard_montage("biosemi128"))
                sessions[str(idx)][str(i)] = eeg_mne
        return sessions

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        subject_name = sub_name(subject)
        base_path = "eeg_data/imagined_speech_dataset/derivatives/{subject_name}/ses-0{session_num}/{subject_name}_ses-0{session_num}_eeg-epo.fif"
        return [base_path.format(session_num=i, subject_name=subject_name) for i in range(1, 4)]
