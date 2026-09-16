from sokoban_state_modeling.data.sampler import FullCoverageSampler


def test_unique_before_padding_and_resume():
    data = list(range(10))
    sampler = FullCoverageSampler(data, batch_size=4, seed=7)
    iterator = iter(sampler)
    first = [next(iterator) for _ in range(7)]
    saved = sampler.state_dict()
    restored = FullCoverageSampler(data, batch_size=4, seed=7)
    restored.load_state_dict(saved)
    tail = list(restored)
    sequence = first + tail
    assert len(sequence) == 12
    assert len({item.task_index for item in sequence[:10]}) == 10
    assert [item.task_index for item in sequence[10:]] == [item.task_index for item in sequence[:2]]
    assert not any(item.is_padding for item in sequence[:10])
    assert all(item.is_padding for item in sequence[10:])
    assert [item.position for item in sequence] == list(range(12))


def test_formal_50000_cycle_shape():
    # PPO uses 1,024 prompts/batch and GRPO uses 256 prompts/group-batch. Both
    # intentionally publish the same 50,176-position coverage cycle.
    for batch_size in (256, 1_024):
        sampler = FullCoverageSampler(range(50_000), batch_size=batch_size, seed=11)
        indices = list(sampler)
        assert len(indices) == 50_176
        assert len({item.task_index for item in indices[:50_000]}) == 50_000
        assert [item.task_index for item in indices[50_000:]] == [
            item.task_index for item in indices[:176]
        ]
        assert all(item.is_padding for item in indices[50_000:])
