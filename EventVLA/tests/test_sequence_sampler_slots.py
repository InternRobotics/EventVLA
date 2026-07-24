import importlib.util
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

SAMPLER_PATH = Path(__file__).parents[1] / "eventvla" / "dataloader" / "sequence_sampler.py"
SPEC = importlib.util.spec_from_file_location("eventvla_sequence_sampler_test", SAMPLER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SequentialEpisodeBatchSampler = MODULE.SequentialEpisodeBatchSampler


class Dataset:
    def __init__(self, lengths):
        self.trajectory_ids = [f"episode-{index}" for index in range(len(lengths))]
        self.trajectory_lengths = lengths


def make_sampler(
    lengths,
    *,
    batch_size=4,
    preserve_episode_batch_slots=True,
    rank=0,
    num_replicas=1,
    sampling_interval=1,
    action_horizon=1,
):
    dataset = SimpleNamespace(datasets=[Dataset(lengths)])
    return SequentialEpisodeBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        sampling_interval=sampling_interval,
        action_horizon=action_horizon,
        preserve_episode_batch_slots=preserve_episode_batch_slots,
        rank=rank,
        num_replicas=num_replicas,
    )


def test_default_mode_preserves_official_flat_batch_layout():
    sampler = make_sampler(
        [8, 8, 8, 8],
        preserve_episode_batch_slots=False,
    )

    first_batch = next(iter(sampler))

    assert [sample[1] for sample in first_batch] == ["episode-0"] * 4
    assert [sample[2] for sample in first_batch] == [0, 1, 2, 3]


def test_persistent_mode_keeps_episode_identity_per_slot():
    sampler = make_sampler([8, 8, 8, 8])
    batches = list(sampler)

    assert [sample[1] for sample in batches[0]] == [
        "episode-0",
        "episode-1",
        "episode-2",
        "episode-3",
    ]
    assert [sample[1] for sample in batches[1]] == [sample[1] for sample in batches[0]]
    assert [sample[2] for sample in batches[0]] == [0, 0, 0, 0]
    assert [sample[2] for sample in batches[1]] == [1, 1, 1, 1]


def test_persistent_mode_preserves_episode_boundaries_within_each_slot():
    sampler = make_sampler([3, 5, 4, 6, 2, 7], batch_size=2)
    batches = list(sampler)

    for slot_index in range(2):
        slot_samples = [batch[slot_index] for batch in batches]
        for previous, current in pairwise(slot_samples):
            same_episode = previous[1] == current[1]
            if same_episode and not previous[4]:
                assert current[2] > previous[2]
                assert not current[3]
            elif not same_episode:
                assert previous[4]
                assert current[3]


def test_persistent_mode_keeps_all_ranks_at_the_same_length():
    samplers = [
        make_sampler(
            [2, 9, 3, 8, 4, 7, 5],
            batch_size=2,
            rank=rank,
            num_replicas=3,
        )
        for rank in range(3)
    ]

    lengths = [len(list(sampler)) for sampler in samplers]

    assert lengths[0] == lengths[1] == lengths[2]
    assert lengths == [len(sampler) for sampler in samplers]


def test_persistent_mode_handles_fewer_rank_episodes_than_batch_slots():
    sampler = make_sampler([5], batch_size=4)
    batches = list(sampler)

    assert len(batches) == len(sampler) == 5
    assert all(len(batch) == 4 for batch in batches)
    for slot_index in range(4):
        assert [batch[slot_index][2] for batch in batches] == [0, 1, 2, 3, 4]


def test_persistent_mode_retains_sparse_anchor_progression():
    sampler = make_sampler(
        [260, 260],
        batch_size=2,
        sampling_interval=50,
        action_horizon=50,
    )
    batches = list(sampler)

    for slot_index in range(2):
        steps = [batch[slot_index][2] for batch in batches]
        assert steps[0] == 0
        assert all(current > previous for previous, current in pairwise(steps))
